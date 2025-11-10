"""
This script generates images using a pretrained autoencoder and Latent Diffusion Model.
It iterates over a dataset and produces multiple (30 by default) generations per record (real image).
The generation can be conditioned on variables, text embeddings, or none.
Author: Lemuel Puglisi
"""

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from tqdm import tqdm
from skimage.io import imsave

from src.utils.diffusion import image_quantize
from src.diffusion.transforms import get_transforms
from src.diffusion.sampling import sample_multiple_using_diffusion
from src.diffusion.networks import init_autoencoder, init_latent_diffusion



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conditioning',     type=str, required=True, choices=['none', 'variables', 'conditioning'])
    parser.add_argument('--dataset_csv',      type=str, required=True)
    parser.add_argument('--aekl_ckpt',        type=str, required=True)
    parser.add_argument('--diff_ckpt',        type=str, required=True)
    parser.add_argument('--output_path',      type=str, required=True)
    parser.add_argument('--latent_shape',     type=str, default='3x36x28')
    parser.add_argument('--vae_space_shape',  type=str, default='3x34x28')
    parser.add_argument('--n_gen_per_record', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Sets the computation device and loads the models.
    # Prepares preprocessing transforms for the dataset according to the conditioning type.
    # Loads the dataset and parses LDM and VAE shapes.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = init_autoencoder(args.aekl_ckpt).to(device).eval()
    ldm = init_latent_diffusion(args.diff_ckpt, args.conditioning).to(device)
    transforms_fn = get_transforms(args.conditioning)

    dataset = pd.read_csv(args.dataset_csv)
    latent_shape = [ int(d) for d in args.latent_shape.split('x') ]
    vae_space_shape = [ int(d) for d in args.vae_space_shape.split('x') ]
    
    # Iterates over each record in the dataset to generate multiple images per record.
    # Each record is preprocessed using the appropriate transforms.
    # The generation can be conditioned on variables (context), text (caption embedding), or none.
    # If conditioning is applied, it is expanded to match the number of generations per record.
    # The diffusion model generates images, which are then saved along with the original image.

    for _dict in tqdm(dataset.to_dict(orient='records'), 'Generating', len(dataset)):

        _data = transforms_fn(_dict)
        img = _data['image'].squeeze(0)

        if args.conditioning == 'variables':        
            condition = _data['context'].to(device)
            condition_rep = condition.unsqueeze(0).repeat(args.n_gen_per_record, 1, 1)
            
        elif args.conditioning == 'text':
            condition = _data['caption_embedding'].to(device)
            condition_rep = condition.unsqueeze(0).repeat(args.n_gen_per_record, 1, 1)
        
        elif args.conditioning == 'none':
            condition = None
            condition_rep = None

        generations = sample_multiple_using_diffusion(
            autoencoder=vae,
            diffusion=ldm,
            condition=condition_rep,
            batch_size=args.n_gen_per_record,
            latent_shape=latent_shape,
            vae_space_shape=vae_space_shape,
            device=device,
            verbose=False
        )

        img_output_dir = os.path.join(args.output_path, _dict['image_uid'])
        os.makedirs(img_output_dir, exist_ok=True)
        imsave(os.path.join(img_output_dir, 'training.png'), image_quantize(img))

        for ith in range(args.n_gen_per_record):
            ith_img = generations[ith].squeeze(0).cpu().numpy()
            ith_img = image_quantize(ith_img)
            imsave(os.path.join(img_output_dir, f'gen-{ith}.png'), ith_img)