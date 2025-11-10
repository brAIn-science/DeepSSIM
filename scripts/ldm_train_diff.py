"""
This script trains a Latent Diffusion Model (LDM) with configurable conditioning.
It uses an autoencoder for VAE encoding/decoding and a diffusion model for generative modeling in the latent space.
This code is adapted from the Brain Latent Progression (BrLP) codebase: 
https://github.com/LemuelPuglisi/BrLP
Author: Lemuel Puglisi
"""

import os
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler

from src.diffusion.transforms import get_transforms
from src.diffusion.networks import init_autoencoder, init_latent_diffusion
from src.utils.diffusion import HistoryLogger, to_ldm_space, duplicate, save_examples


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditioning',    type=str, required=True, choices=['none', 'variables', 'conditioning'])
    parser.add_argument('--duplication',     type=str, required=True, choices=['none', 'low', 'high'])
    parser.add_argument('--dataset_csv',     type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--output_dir',      type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--aekl_ckpt',       type=str, required=True, help='Path to the autoencoder checkpoint file.')
    parser.add_argument('--diff_ckpt',       type=str, default=None, help='[OPTIONAL] Path to the diffusion checkpoint file.')
    parser.add_argument('--latent_shape',    type=str, default='3x36x28')
    parser.add_argument('--vae_space_shape', type=str, default='3x34x28')
    parser.add_argument('--batch_size',      type=int, default=8)
    parser.add_argument('--n_workers',       type=int, default=4)
    parser.add_argument('--n_epochs',        type=int, default=100)
    parser.add_argument('--lr',              type=float, default=2.5e-5)
    args = parser.parse_args()

    plots_dir = os.path.join(args.output_dir, 'plots')
    models_dir = os.path.join(args.output_dir, 'models')
    historypath = os.path.join(args.output_dir, 'history.csv')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    set_determinism(0)

    # Device and model initialization.
    # Sets the computation device and initialize the autoencoder and the LDM.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autoencoder = init_autoencoder(args.aekl_ckpt).to(device)
    diffusion = init_latent_diffusion(args.diff_ckpt, args.conditioning).to(device)

    # Loads the dataset CSV file containing the split information.
    # Initializes DataLoader for the Training Set with shuffling and parallel data loading enabled.
    # Initializes DataLoader for the Validation Set without shuffling to maintain consistent evaluation.

    latent_shape    = [ int(d) for d in args.latent_shape.split('x') ]
    vae_space_shape = [ int(d) for d in args.vae_space_shape.split('x') ]

    dataset = pd.read_csv(args.dataset_csv)
    train_df = dataset[dataset.split == 'train']
    train_df = duplicate(train_df, args.duplication)
    train_df.to_csv(os.path.join(args.output_dir, 'train-used.csv'), index=False)
    train_data = train_df.to_dict(orient='records')
    valid_data = dataset[dataset.split == 'valid'].to_dict(orient='records')

    transforms_fn = get_transforms(args.conditioning)
    trainset = Dataset(data=train_data, transform=transforms_fn)
    validset = Dataset(data=valid_data, transform=transforms_fn)

    train_loader = DataLoader(
        dataset=trainset, 
        batch_size=args.batch_size,
        num_workers=args.n_workers, 
        shuffle=True, 
        persistent_workers=True,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        dataset=validset, 
        batch_size=args.batch_size,
        num_workers=args.n_workers, 
        shuffle=False, 
        persistent_workers=True,
        pin_memory=True
    )
        
    logger = HistoryLogger(historypath, ['epoch', 'mode', 'loss'])

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule='scaled_linear_beta', 
        beta_start=0.0015, 
        beta_end=0.0205
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    scaler = GradScaler()
    scale_factor = 1

    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }
    datasets        = { 'train': trainset, 'valid': validset }
    best_valid_loss = float('inf')

    for epoch in range(args.n_epochs):
        
        for mode in loaders.keys():
            
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f'Epoch {epoch}')
            
            for step, batch in progress_bar:
                            
                with autocast(enabled=True):
                    
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)

                    latents = None                    
                    with torch.no_grad():
                        images = batch['image'].to(device)
                        latents = autoencoder.encode(images)[0]                        
                        latents = to_ldm_space(latents) * scale_factor                    
                    
                    if args.conditioning == 'variables':        
                        condition = batch['context'].to(device)
                        
                    elif args.conditioning == 'text':
                        condition = batch['caption_embedding'].to(device)
                    
                    elif args.conditioning == 'none':
                        condition = None
                    
                    else: raise Exception('Invalid conditioning.')
                    
                    n = latents.shape[0]
                                                            
                    with torch.set_grad_enabled(mode == 'train'):
                        noise = torch.randn_like(latents).to(device)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()
                        noise_pred = inferer(
                            inputs=latents, 
                            diffusion_model=diffusion, 
                            noise=noise, 
                            timesteps=timesteps,
                            condition=condition
                        )
                            
                        loss = F.mse_loss( noise.float(), noise_pred.float() )

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': epoch_loss / (step + 1)})
                global_counter[mode] += 1
        
            # end of epoch
            epoch_loss = epoch_loss / len(loader)
                        
            # log loss
            logger.log({'epoch': epoch, 'mode': mode, 'loss': epoch_loss})

            figure_path = os.path.join(plots_dir, f'gen_{epoch}_{mode}.png')
            save_examples(
                output_path=figure_path, 
                vae=autoencoder, 
                ldm=diffusion, 
                condition=(None if condition is None else condition[0]), 
                device=device,
                vae_space_shape=vae_space_shape,
                latent_shape=latent_shape
            )

            # Save the model if validation loss decreases or if 20 epochs have been done
            if mode == 'valid' and (epoch_loss < best_valid_loss or epoch % 20 == 0):
                print(f'new model saved: best validation loss is {best_valid_loss:.4f}')
                best_valid_loss = epoch_loss
                savepath = os.path.join(models_dir, f'unet-ep-{epoch}.pth')
                torch.save(diffusion.state_dict(), savepath)