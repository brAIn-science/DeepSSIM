import os
import torch
import psutil
import random
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.nn import MSELoss
from torch.optim import AdamW
from monai.data import DataLoader

from src.utils.log import log_metrics
from src.train.dataset import ImagePairDataset
from src.utils.meter import AverageMetricsMeter
from src.train.similarity_net import SimilarityNet
from src.utils.model import save_model_and_optimizer

# This script trains the DeepSSIM model to approximate the SSIM score.
# It supports configurable training hyper-parameters. The script also allows enabling GPU acceleration.
# Upon execution, it sets up directories for storing training logs and model checkpoints.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu',            action='store_true')
    parser.add_argument('--exp_name',           type=str, required=True)
    parser.add_argument('--dataset_csv',        type=str, required=True)
    parser.add_argument('--dataset_images_dir', type=str, required=True)
    parser.add_argument('--num_workers',        type=int, default=psutil.cpu_count(logical=False))
    parser.add_argument('--emb_dim',            type=int, default=256)
    parser.add_argument('--epochs',             type=int, default=40)
    parser.add_argument('--batch_size',         type=int, default=32)
    parser.add_argument('--learning_rate',      type=float, default=1e-3)
    parser.add_argument('--weight_decay',       type=float, default=1e-3)
    parser.add_argument('--dropout_prob',       type=float, default=0.33)
    args = parser.parse_args()

    BASE_LOG_PATH = os.path.join('..', 'logs', args.exp_name)
    OUTPUT_PATH = os.path.join(BASE_LOG_PATH, 'checkpoints')
    for path in [BASE_LOG_PATH, OUTPUT_PATH]:
        os.makedirs(path, exist_ok=True)

    # Sets fixed seeds to ensure reproducibility.
    # This removes randomness and ensures consistent results across runs.

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Loads the dataset CSV file containing image pairs and their split information.
    # Initializes DataLoader for the Training Set with shuffling and parallel data loading enabled.
    # Initializes DataLoader for the Validation Set without shuffling to maintain consistent evaluation.
    # Combines the data loaders into a dictionary for easy access during training and validation.
    
    dataset = pd.read_csv(args.dataset_csv)
    train_data = dataset[dataset['split'] == 'train']
    valid_data = dataset[dataset['split'] == 'valid']

    train_dataset = ImagePairDataset(train_data, args.dataset_images_dir)
    valid_dataset = ImagePairDataset(valid_data, args.dataset_images_dir)

    train_loader = DataLoader(
        dataset=train_dataset,
        persistent_workers=True, 
        pin_memory=True,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    # Selects the device: uses GPU if requested and available, otherwise fallbacks to CPU.
    # Initializes the neural network, the optimizer, and defines the loss function.

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    model = SimilarityNet(args.emb_dim, args.dropout_prob).to(device)
    optimizer = AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    criterion = MSELoss()

    # It is the main training and validation loop.
    # For each epoch, the model alternates between training and validation modes.
    # An AverageMetricsMeter tracks the loss (MSE) and the performance (MAE) for each phase.
    # The best model is saved after each epoch if it improves over previous results.
    # All progress is logged to TensorBoard, and GPU memory is cleared after each phase to prevent memory issues.

    best_valid_loss = float('inf')
    meter = AverageMetricsMeter()

    for epoch in range(args.epochs):
        for mode in ['train', 'valid']:
            model.train() if mode == 'train' else model.eval()
            meter.reset()

            description = 'Epoch [%d] in [%s]' % (epoch, mode)
            for batch in tqdm(loaders[mode], description):
                with torch.set_grad_enabled(mode == 'train'):
                    
                    img1 = batch['img1'].to(device).float()
                    img2 = batch['img2'].to(device).float()
                    y_true = batch['ssim'].to(device).float().unsqueeze(1)
                    y_pred = model(img1, img2)
                    
                    loss = criterion(y_pred, y_true)
                    mae = torch.mean(torch.abs(y_pred - y_true)).item()
                    meter.add(loss.item(), mae, len(batch))

                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

            log_metrics(mode, epoch, meter.loss_mean(), meter.loss_std(), meter.performance_mean())
            torch.cuda.empty_cache()

            if mode == 'valid' and meter.loss_mean() < best_valid_loss:
                best_valid_loss = meter.loss_mean()
                save_model_and_optimizer(model.embedding_net, optimizer, OUTPUT_PATH)