import os
import PIL.Image
import numpy as np
import monai.transforms

from skimage.metrics import structural_similarity as ssim

# Define a sequence of stochastic augmentations to apply to grayscale images.
# These transformations simulate common brain MRI variations for testing.
# Author: Antonio Scardace

prob = 0.8
transforms = monai.transforms.Compose([ 
    monai.transforms.RandFlip(spatial_axis=0, prob=prob),
    monai.transforms.RandFlip(spatial_axis=1, prob=prob),
    monai.transforms.RandRotate(range_x=0.1745, prob=prob),
    monai.transforms.RandAdjustContrast(gamma=(0.5, 1.5), prob=prob),
    monai.transforms.EnsureType(dtype=np.float32, data_type='numpy')
])

# These functions compute the Structural Similarity Index (SSIM) between two images.
# They include grayscale conversion, optional intensity normalization, and similarity computation.
# Normalization affects brightness but does not modify contrast.
# Author: Antonio Scardace

def load_grayscale_image(path: str, normalise: bool) -> np.array:
    image = np.array(PIL.Image.open(path).convert('L'))
    return (image - np.mean(image)) / np.std(image) if normalise else image

def get_ssim(image_1: np.array, image_2: np.array, data_range: float) -> float:
    similarity_index, _ = ssim(image_1, image_2, full=True, data_range=data_range)
    return similarity_index

def calculate_ssim(image_1_path: str, image_2_path: str, normalise: bool) -> float:
    image_1 = load_grayscale_image(image_1_path, normalise)
    image_2 = load_grayscale_image(image_2_path, normalise)
    data_range = image_1.max() - image_1.min() if normalise else None
    return get_ssim(image_1, image_2, data_range)

# These functions compute the Structural Similarity Index (SSIM) between two images.
# They include grayscale conversion, optional intensity normalization, and similarity computation.
# Normalization affects brightness but does not modify contrast.
# Also apply a series of random affine augmentations using MONAI transforms.
# Author: Antonio Scardace

def load_and_augment(path: str, normalise: bool) -> np.ndarray:
    image = load_grayscale_image(path, normalise)
    image = transforms(np.expand_dims(image, axis=0))
    return np.squeeze(image, axis=0)

def calculate_ssim_augmented(image_1_path: str, image_2_path: str, normalise: bool) -> float:
    image_1 = load_and_augment(image_1_path, normalise)
    image_2 = load_and_augment(image_2_path, normalise)
    data_range = image_1.max() - image_1.min() if normalise else None
    return get_ssim(image_1, image_2, data_range)

# This function processes a batch for SSIM matrix computation.
# It computes SSIM scores between one real image and a list of synthetic images,
# then saves the results as a NumPy compressed array in a temporary directory.
# Author: Antonio Scardace

def process_batch(batch: tuple) -> int:
    ridx, real_key, synth_indices, images_dir, tmp_dir = batch
    real_path = os.path.join(images_dir, real_key.replace('___', '/') + '.png')

    ssim_values = []
    for _, synth_key in enumerate(synth_indices):
        synth_path = os.path.join(images_dir, synth_key.replace('___', '/') + '.png')
        ssim_values.append(calculate_ssim(real_path, synth_path, normalise=True))

    output_path = os.path.join(tmp_dir, str(ridx) + '.npz')
    np.savez(output_path, data=np.array(ssim_values, dtype=np.float16))
    return ridx