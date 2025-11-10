"""
This code is adapted from the Brain Latent Progression (BrLP) codebase: 
https://github.com/LemuelPuglisi/BrLP
Author: Lemuel Puglisi
"""
import os
from typing import Optional

import torch
import torch.nn as nn
from generative.networks.nets import (
    AutoencoderKL, 
    PatchDiscriminator, 
    DiffusionModelUNet
)


def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    This function loads pretrained weights into a neural network if a checkpoint path is provided.
    If no checkpoint is given, the network is returned with its default initialization.
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        network.load_state_dict(torch.load(checkpoints_path))
    return network


def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    This function initializes a KL autoencoder.
    If a checkpoint path is provided, it loads pretrained weights from the specified file.
    """
    autoencoder = AutoencoderKL(
        spatial_dims=2, 
        in_channels=1, 
        out_channels=1, 
        latent_channels=3,
        num_channels=(64, 128, 128),
        num_res_blocks=2, 
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=(False, False, False), 
        with_decoder_nonlocal_attn=False, 
        with_encoder_nonlocal_attn=False
    )
    return load_if(checkpoints_path, autoencoder)


def init_patch_discriminator(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    This function initializes a Patch Discriminator for adversarial training.
    If a checkpoint path is provided, pretrained weights are loaded automatically.
    """
    patch_discriminator = PatchDiscriminator(
        spatial_dims=2,
        num_layers_d=3, 
        num_channels=32, 
        in_channels=1, 
        out_channels=1
    )
    return load_if(checkpoints_path, patch_discriminator)


def init_latent_diffusion(checkpoints_path: Optional[str] = None, conditioning: str = 'none') -> nn.Module:
    """
    This function initializes the U-Net backbone for the latent diffusion model.
    It adapts its configuration based on the conditioning type ("none", "variables", "text").
    If a checkpoint path is provided, pretrained weights are loaded automatically.
    """

    if conditioning not in ['variables', 'text', 'none']:
        raise Exception('Invalid conditioning.')

    if conditioning in ['variables', 'text']:
        cross_attn_dim = 512 if conditioning == 'text' else 6    
        latent_diffusion = DiffusionModelUNet(
            spatial_dims=2, 
            in_channels=3, 
            out_channels=3, 
            num_res_blocks=2, 
            num_channels=(256, 512, 768), 
            attention_levels=(False, True, True), 
            norm_num_groups=32, 
            norm_eps=1e-6, 
            resblock_updown=True, 
            num_head_channels=(0, 512, 768), 
            transformer_num_layers=1,
            with_conditioning=True,
            cross_attention_dim=cross_attn_dim,
            num_class_embeds=None, 
            upcast_attention=True, 
            use_flash_attention=False
        )
    elif conditioning == 'none':
        latent_diffusion = DiffusionModelUNet(
            spatial_dims=2, 
            in_channels=3, 
            out_channels=3, 
            num_res_blocks=2, 
            num_channels=(256, 512, 768), 
            attention_levels=(False, True, True), 
            norm_num_groups=32, 
            norm_eps=1e-6, 
            resblock_updown=True, 
            num_head_channels=(0, 512, 768), 
            transformer_num_layers=1,
            num_class_embeds=None, 
            upcast_attention=True, 
            use_flash_attention=False
        )

    return load_if(checkpoints_path, latent_diffusion)