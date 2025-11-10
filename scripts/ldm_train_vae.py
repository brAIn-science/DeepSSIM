"""
This script trains a Variational Autoencoder (VAE) with a patch discriminator.
The training objective combines reconstruction, perceptual, KL, and adversarial losses.
This code is adapted from the Brain Latent Progression (BrLP) codebase: 
https://github.com/LemuelPuglisi/BrLP
Author: Lemuel Puglisi
"""

import os
import argparse
import warnings

import torch
import pandas as pd
from tqdm import tqdm
from torch.nn import L1Loss
from monai.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss

from src.diffusion.transforms import get_transforms
from src.diffusion.networks import init_autoencoder, init_patch_discriminator
from src.utils.diffusion import KLDivergenceLoss, GradientAccumulation, AverageLoss, save_reconstruction


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    type=str, required=True, help='Path to the dataset csv file.')
    parser.add_argument('--output_dir',     type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--n_workers',      type=int, default=4)
    parser.add_argument('--batch_size',     type=int, default=16)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--n_epochs',       type=int, default=100)
    parser.add_argument('--lr',             type=float, default=1e-4)
    args = parser.parse_args()

    plots_dir = os.path.join(args.output_dir, 'plots')
    models_dir = os.path.join(args.output_dir, 'models')
    historypath = os.path.join(args.output_dir, 'history.csv')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Device and model initialization.
    # Sets the computation device and initialize the autoencoder and the path discriminator.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autoencoder = init_autoencoder().to(device)
    discriminator = init_patch_discriminator().to(device)
    
    # Loads the dataset CSV file containing the split information.
    # Initializes DataLoader for the Training Set with shuffling and parallel data loading enabled.
    # Initializes DataLoader for the Validation Set without shuffling to maintain consistent evaluation.

    dataset = pd.read_csv(args.dataset_csv)
    train_data = dataset[dataset.split == 'train'].to_dict(orient='records')
    valid_data = dataset[dataset.split == 'valid'].to_dict(orient='records')
    transforms_fn = get_transforms('none')

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
        
    adv_weight = 0.125
    perceptual_weight = 0.1
    kl_weight = 1e-6

    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion='least_squares')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        perc_loss_fn = PerceptualLoss(spatial_dims=2, 
                                      network_type='squeeze', 
                                      is_fake_3d=True, 
                                      fake_3d_ratio=0.2).to(device)
    
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_g, 
                                     grad_scaler=GradScaler())

    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_d, 
                                     grad_scaler=GradScaler())

    avgloss = AverageLoss()
    total_counter = 0
    
    for epoch in range(args.n_epochs):
        
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:

            with autocast(enabled=True):

                images = batch['image'].to(device)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # we use [-1] here because the discriminator also returns 
                # intermediate outputs and we want only the final one.
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # Computing the loss for the generator. In the Adverarial loss, 
                # if the discriminator works well then the logits are close to 0.
                # Since we use `target_is_real=True`, then the target tensor used
                # for the MSE is a tensor of 1, and minizing this loss will make 
                # the generator better at fooling the discriminator (the discriminator
                # weights are not optimized here).

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                
                loss_g = rec_loss + kld_loss + per_loss + gen_loss
                
            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):

                # Here we compute the loss for the discriminator. Keep in mind that
                # the loss used is an MSE between the output logits and the expected logits.
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # Logging.
            avgloss.put('Generator/reconstruction_loss',    rec_loss.item())
            avgloss.put('Generator/perceptual_loss',        per_loss.item())
            avgloss.put('Generator/adverarial_loss',        gen_loss.item())
            avgloss.put('Generator/kl_regularization',      kld_loss.item())
            avgloss.put('Discriminator/adverarial_loss',    loss_d.item())
            
            if total_counter % 50 == 0:
                                
                step = total_counter // 10
                avgloss.to_csv(historypath, step)
            
                with torch.no_grad():
                    determ_reconstruction = autoencoder.reconstruct(images)    
                    save_reconstruction(
                        os.path.join(plots_dir, f'{step}-recon.png'), 
                        images[0].detach().cpu(), 
                        determ_reconstruction[0].detach().cpu()
                    )
        
            total_counter += 1

        # Save the model after each epoch.
        torch.save(discriminator.state_dict(), os.path.join(models_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(),   os.path.join(models_dir, f'autoencoder-ep-{epoch}.pth'))