import os
import torch
import psutil
import random
import argparse
import monai.utils
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.utils import extract_image_identifier
from src.factories.registry import MetricFactoryRegistry

# This script computes similarity scores between synthetic and real image embeddings.
# It performs the following steps:
# (1) Extracts image embeddings using a specified deep learning model.
# (2) Computes a score matrix between synthetic and training embeddings.
# (3) Saves embeddings, score matrix, and indexing metadata into dedicated directories.
# Authors: Lemuel Puglisi and Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--augment',            action='store_true')
    parser.add_argument('--use_gpu',            action='store_true')
    parser.add_argument('--metric_name',        type=str, required=True, choices=['dar', 'chen', 'semdedup', 'deepssim'])
    parser.add_argument('--model_path',         type=str, required=True)
    parser.add_argument('--dataset_images_dir', type=str, required=True)
    parser.add_argument('--embeddings_dir',     type=str, required=True)
    parser.add_argument('--matrices_dir',       type=str, required=True)
    parser.add_argument('--indices_dir',        type=str, required=True)
    parser.add_argument('--batch_size',         type=int, default=32)
    parser.add_argument('--num_workers',        type=int, default=psutil.cpu_count(logical=False))
    args = parser.parse_args()

    embeddings_dir = os.path.join(args.embeddings_dir, args.metric_name)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(args.matrices_dir, exist_ok=True)
    os.makedirs(args.indices_dir, exist_ok=True)

    monai.utils.misc.set_determinism(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Instantiates the metric Concrete Creator via the Abstract Factory pattern, selecting the model dynamically.
    # Initializes the model with the given path and loads it onto the chosen device.  

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')  
    metric_factory = MetricFactoryRegistry.get_metric(args.metric_name, args.augment)
    feature_extractor = metric_factory.create_feature_extractor(args.model_path, device)
    scorer = metric_factory.create_embedding_scorer()
    
    # For each image UID folder, the script constructs the full path to every contained image.
    # Each batch of images is processed to compute embeddings, which are then saved as .npz files indexed by image UID.

    image_uid_dirs = [os.path.join(args.dataset_images_dir, image_uid) for image_uid in os.listdir(args.dataset_images_dir)]
    images_paths = [os.path.join(subdir, file) for subdir in image_uid_dirs if os.path.isdir(subdir) for file in os.listdir(subdir)]
    data = [{'image': path, 'uid': extract_image_identifier(path)} for path in images_paths]

    dataloader = DataLoader(
        dataset=metric_factory.create_dataset(data),
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    for batch in tqdm(dataloader, 'Computing the embeddings', len(dataloader)):
        imgs = batch['image'].to(device)
        uids = batch['uid']
        with torch.no_grad():
            embs = feature_extractor.get_batch_embeddings(imgs).cpu().numpy()
            for uid, emb in zip(uids, embs):
                np.savez(os.path.join(embeddings_dir, uid + '.npz'), emb)

    # Classifies embeddings as training/synthetic based on filename.
    # Converts the stored embeddings into tensors and computes the score matrix between training and synthetic embeddings.
    # Saves the computed score matrix along with the corresponding indices as compressed .npz files.
    # Each matrix entry (i, j) stores the score between the i-th synthetic and j-th training embedding.

    emb_paths = [os.path.join(embeddings_dir, f) for f in os.listdir(embeddings_dir)]
    train_embs, synth_embs = [], []
    train_ids, synth_ids = [], []

    for path in tqdm(emb_paths, 'Loading the embeddings', len(emb_paths)):
        emb = np.load(path)['arr_0']
        uid = os.path.basename(path).replace('.npz', '')
        (train_embs if path.endswith('training.npz') else synth_embs).append(emb)
        (train_ids if path.endswith('training.npz') else synth_ids).append(uid)

    print('Computing the score matrix...')
    train_embs_tensor = torch.cat([torch.from_numpy(e).unsqueeze(0) for e in train_embs], dim=0)
    synth_embs_tensor = torch.cat([torch.from_numpy(e).unsqueeze(0) for e in synth_embs], dim=0)
    score_matrix = scorer.compute_matrix(train_embs_tensor, synth_embs_tensor)

    np.savez(os.path.join(args.matrices_dir, args.metric_name + '.npz'), data=score_matrix.astype(np.float16))
    np.savez(os.path.join(args.indices_dir, 'real.npz'), data=np.array(train_ids))
    np.savez(os.path.join(args.indices_dir, 'synth.npz'), data=np.array(synth_ids))