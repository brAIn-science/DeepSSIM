"""
This code is adapted from the Brain Latent Progression (BrLP) codebase: 
https://github.com/LemuelPuglisi/BrLP
Author: Lemuel Puglisi
"""
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.cuda.amp import autocast
from generative.networks.schedulers import DDIMScheduler

from ..utils.diffusion import to_vae_space


@torch.no_grad()
def sample_using_diffusion(
    autoencoder: nn.Module, 
    diffusion: nn.Module,
    condition: torch.Tensor, 
    device: str, 
    scale_factor: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True,
    latent_shape: tuple = (3, 36, 28),
    vae_space_shape: tuple = (3, 34, 28)
) -> torch.Tensor: 
    """
    This function samples a single synthetic brain MRI using a diffusion model.
    It applies DDIM sampling (Song et al., 2020) for efficient and deterministic denoising.
    The output is a reconstructed MRI image consistent with the provided conditioning variables.
    """

    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False
    )

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    z = torch.randn(latent_shape).unsqueeze(0).to(device)

    if condition is not None:
        condition = condition.unsqueeze(0).to(device)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                timestep = torch.tensor([t]).to(device)
                noise_pred = diffusion(x=z.float(), timesteps=timestep, context=condition)
                z, _ = scheduler.step(noise_pred, t, z)
    
    z = z / scale_factor
    z = to_vae_space(z, unpadded_z_shape=vae_space_shape)
    return autoencoder.decode_stage_2_outputs(z)


@torch.no_grad()
def sample_multiple_using_diffusion(
    autoencoder: nn.Module, 
    diffusion: nn.Module, 
    condition: torch.Tensor,
    device: str, 
    scale_factor: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True,
    latent_shape: tuple = (3, 36, 28),
    vae_space_shape: tuple = (3, 34, 28),
    batch_size=1
) -> torch.Tensor: 
    """
    This function samples multiple synthetic brain MRIs simultaneously using a diffusion model.
    It extends the single-sample version to support batched sampling for improved efficiency.
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False
    )

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    z = torch.randn([batch_size] + list(latent_shape)).to(device)
    
    if condition is not None:
        condition = condition.to(device)
    
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                timestep = torch.tensor([t]).repeat(batch_size).to(device)
                noise_pred = diffusion(x=z.float(), timesteps=timestep, context=condition)
                z, _ = scheduler.step(noise_pred, t, z)
    
    z = z / scale_factor
    z = to_vae_space(z, unpadded_z_shape=vae_space_shape)
    return autoencoder.decode_stage_2_outputs(z)