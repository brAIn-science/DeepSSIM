import os
import PIL.Image
import numpy as np

# This function extracts the UID and name of an image from its file path.
# The UID corresponds to the parent directory, while the name is the filename without the ".png" extension.
# The extracted values are concatenated using "___" as a separator.
# Author: Lemuel Puglisi

def extract_image_identifier(path: str) -> str:
    uid = path.split('/')[-2]
    name = path.split('/')[-1].replace('.png', '')
    return uid + '___' + name

# This function reconstructs the file path of an image from its identifier.
# The identifier follows the "UID___FILENAME" format.
# The function returns the full path by joining the base directory, UID, and filename.
# Author: Antonio Scardace

def get_image_path(identifier: str, base_path: str) -> str:
    uid, name = identifier.split('___')
    return os.path.join(base_path, uid, name + '.png')

# These functions handle the end-to-end computation of the Structural Similarity Index (SSIM) between image pairs.
# They provide utilities for grayscale image loading, optional stochastic augmentation, and Z-score intensity normalization.
# The normalization step aligns the intensity distributions without altering the underlying image contrast.
# Author: Antonio Scardace

def load_grayscale_image(path: str, normalise: bool) -> np.array:
    image = np.array(PIL.Image.open(path).convert('L'))
    return (image - np.mean(image)) / np.std(image) if normalise else image