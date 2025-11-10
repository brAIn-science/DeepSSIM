"""
Author: Lemuel Puglisi
"""

from monai import transforms
from monai.data.image_reader import NumpyReader

from ..utils.diffusion import concat_covariates


def get_transforms(conditioning_type: str):
    """
    This function returns the correct set of MONAI transforms 
    depending on the selected conditioning type ('none', 'variables', or 'text').
    """
    if conditioning_type == 'none': return BaseTransforms().get()
    elif conditioning_type == 'variables': return VariablesTransforms().get()
    elif conditioning_type == 'text': return TextTransforms().get()
    else: raise Exception('Invalid conditioning.')


class BaseTransforms:
    """
    It loads the image from disk, ensures the channel-first format, and rescales intensity to [0, 1].
    """

    def __init__(self):
        self.transforms = [
            transforms.CopyItemsD(keys=['image_path'], names=['image']),
            transforms.LoadImageD(keys=['image']),
            transforms.EnsureChannelFirstd(keys=['image']),
            transforms.ScaleIntensityD(keys=['image'], minv=0, maxv=1),
        ]
        
    def get(self):
        return transforms.Compose(self.transforms)
        
        
class VariablesTransforms(BaseTransforms):
    """
    It extends BaseTransforms to include conditioning variables (e.g. age, sex)
    as an additional context vector for cross-attention conditioning.
    """

    def __init__(self):
        super().__init__()
        self.transforms += [transforms.Lambda(func=concat_covariates) ]

        
class TextTransforms(BaseTransforms):
    """
    It extends BaseTransforms to include text conditioning.
    It loads a precomputed PubMedBERT embedding from an .npz file and formats it as a tensor.
    """
    def __init__(self):
        super().__init__()
        self.transforms += [
            transforms.CopyItemsD(keys=['caption_emb_path'], names=['caption_embedding']),
            transforms.LoadImageD(keys=['caption_embedding'], reader=NumpyReader(npz_keys='data')),
            transforms.ToTensorD(keys=['caption_embedding']),
            transforms.LambdaD(keys=['caption_embedding'], func=lambda x: x.view(1, -1))
        ]