import os
import wandb
import monai
import torch
import psutil
import random
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.nn import MSELoss
from torch.amp import autocast
from torch.amp import GradScaler
from monai.data import DataLoader

from src.train.dataset import ImagePairDataset
from src.train.similarity_net import SimilarityNet

from src.utils.log import log_metrics
from src.utils.llrd import build_llrd_optimizer
from src.utils.meter import AverageMetricsMeter
from src.utils.model import save_model_and_optimizer

# This script trains the DeepSSIM network to approximate the SSIM score between image pairs.
# It configures the training environment, logs metrics to Weights & Biases (WandB), and saves the best model checkpoints.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu',            action='store_true',       help='Enable GPU acceleration for training.')
    parser.add_argument('--exp_name',           type=str,   required=True, help='Name of the experiment for logging and saving checkpoints.')
    parser.add_argument('--dataset_csv',        type=str,   required=True, help='Path to the CSV file containing image pairs and splits.')
    parser.add_argument('--dataset_images_dir', type=str,   required=True, help='Directory containing the dataset images.')
    parser.add_argument('--base_lr',            type=float, default=1e-3,  help='Base learning rate for the optimizer.')
    parser.add_argument('--llrd_decay',         type=float, default=3e-1,  help='Layer-wise learning rate decay factor.')
    parser.add_argument('--weight_decay',       type=float, default=1e-3,  help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--dropout_prob',       type=float, default=1e-1,  help='Dropout probability for the network.')
    parser.add_argument('--emb_dim',            type=int,   default=256,   help='Dimensionality of the extracted image embeddings.')
    parser.add_argument('--epochs',             type=int,   default=65,    help='Total number of training epochs.')
    parser.add_argument('--batch_size',         type=int,   default=128,   help='Number of samples per training batch.')
    parser.add_argument('--num_workers',        type=int,   default=psutil.cpu_count(logical=False), help='Number of subprocesses for data loading.')
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'logs', args.exp_name, 'checkpoints')
    OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    wandb.init(
        project='DeepSSIM',
        name=args.exp_name,
        settings=wandb.Settings(x_disable_stats=True, x_disable_meta=True)
    )

    wandb.define_metric("epoch", hidden=True) 
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("valid/*", step_metric="epoch")

    # Sets fixed seeds to ensure reproducibility.
    # This removes randomness and ensures consistent results across runs.

    monai.utils.set_determinism(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Splits the dataset into training and validation subsets based on the CSV metadata.
    # Initializes the corresponding DataLoaders, enabling shuffling exclusively for the training phase.
    
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

    # Allocates the model on the selected hardware and configures the loss function.
    # Initializes the optimizer with layer-wise learning rate decay and the AMP GradScaler.

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    is_enabled = device.type == 'cuda'

    model = SimilarityNet(args.emb_dim, args.dropout_prob).to(device)
    optimizer = build_llrd_optimizer(model, args.base_lr, args.llrd_decay, args.weight_decay)
    scaler = GradScaler(device.type, enabled=is_enabled)
    criterion = MSELoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('Trainable Parameters:', trainable_params, 'out of', total_params)

    # Executes the main loop, alternating between optimization and validation phases.
    # Computes MSE and MAE metrics, applying gradient scaling and clipping during backpropagation.
    # Logs the results to WandB and saves the embedder weights if the validation loss improves.

    best_valid_loss = float('inf')
    meter = AverageMetricsMeter()

    for epoch in range(args.epochs):
        for mode in ['train', 'valid']:
            model.train() if mode == 'train' else model.eval()
            meter.reset()

            description = 'Epoch [%d] in [%s]' % (epoch, mode)
            for batch in tqdm(loaders[mode], description, unit='batch'):

                with torch.set_grad_enabled(mode == 'train'):
                    img1 = batch['img1'].to(device).float()
                    img2 = batch['img2'].to(device).float()
                    y_true = batch['ssim'].to(device).float().unsqueeze(1)

                    with autocast(device.type, enabled=is_enabled):
                        y_pred = model(img1, img2)
                        loss = criterion(y_pred, y_true)

                    if torch.isnan(loss):
                        print('NaN loss detected during', mode, 'at epoch', epoch)
                        wandb.finish()
                        raise SystemExit(0)
                    
                    mae = torch.mean(torch.abs(y_pred - y_true)).item()
                    meter.add(loss.item(), mae, len(batch))

                    if mode == 'train':
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

            log_metrics(mode, epoch, meter.loss_mean(), meter.loss_std(), meter.performance_mean())
            torch.cuda.empty_cache()

            if mode == 'valid' and meter.loss_mean() < best_valid_loss:
                print('Saved at epoch', epoch, 'with MSE', meter.loss_mean())
                best_valid_loss = meter.loss_mean()
                save_model_and_optimizer(model.embedding_net, optimizer, OUTPUT_PATH)

    wandb.finish()