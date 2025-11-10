"""
Author: Lemuel Puglisi
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from monai import transforms
from typing import Optional, Union
from torch.cuda.amp import GradScaler
from monai.data.meta_tensor import MetaTensor
from torch.utils.tensorboard import SummaryWriter

from src.diffusion.sampling import sample_using_diffusion


def image_quantize(img: torch.Tensor) -> np.ndarray:
    """
    This function rescales a tensor to the [0, 255] range and converts it to an unsigned 8-bit numpy array.
    It is used for visualizing or saving images generated as tensors.
    """
    return (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)


def to_ldm_space(z: torch.Tensor) -> torch.Tensor:
    """
    This function pads tensors to make their spatial dimensions divisible by 4,
    which is required for compatibility with latent diffusion models (LDMs).
    """
    padder = transforms.DivisiblePad(k=4)
    return torch.cat([ padder(z[i]).unsqueeze(0) for i in range(z.shape[0]) ])


def concat_covariates(sample: dict) -> dict:
    """
    This function prepares contextual conditioning inputs for cross-attention layers
    by concatenating normalized covariates (e.g. age, sex) along the channel dimension.
    """
    conditioning_vars = ['age_norm', 'sex_norm', 'sequence_norm', 'csf_perc', 'wm_perc', 'gm_perc']
    sample['context'] = torch.tensor([ sample[c] for c in conditioning_vars ]).unsqueeze(0)
    return sample


def to_vae_space(z: torch.Tensor, unpadded_z_shape: tuple = (3, 28, 34)) -> torch.Tensor:
    """
    This function pads and unpads tensors to convert from the LDM space back to the VAE space.
    It ensures that images reconstructed by the VAE maintain their original spatial resolution.
    """
    padder = transforms.DivisiblePad(k=4)
    batch = []
    for i in range(z.shape[0]):
        _z = z[i]
        _z = padder(MetaTensor(torch.zeros(unpadded_z_shape))).to(_z.device) + _z
        _z = padder.inverse(_z)
        batch.append(_z.unsqueeze(0))
    return torch.cat(batch)


def duplicate(df: pd.DataFrame, duplication_mode: str) -> pd.DataFrame:
    """
    This function duplicates rows in a DataFrame to artificially increase the dataset size.
    The duplication range depends on the selected duplication mode ("none", "low", "high").
    """
    if duplication_mode == 'none':
        return df.copy()

    duplication_range = (1, 5) if duplication_mode == 'low' else (1, 15)
    df = df.groupby('image_uid', as_index=False).first()
    counts = np.random.randint(*duplication_range, size=len(df))
    data = [deepcopy(row) for row, n in zip(df.to_dict(orient='records'), counts) for _ in range(n)]
    return pd.DataFrame(data)
    

def save_reconstruction(outpath: str, image: torch.Tensor, recon: torch.Tensor) -> None:
    """
    This function saves a side-by-side comparison between an original image and its reconstruction.
    It is typically used during autoencoder training for visual monitoring via TensorBoard.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=2, figsize=(7, 5))
    for _ax in ax.flatten(): _ax.set_axis_off()

    if len(image.shape) == 4:
        image = image.squeeze(0) 
    if len(recon.shape) == 4:
        recon = recon.squeeze(0)

    ax[0].set_title('original image', color='cyan')
    ax[0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[1].set_title('reconstructed image', color='magenta')
    ax[1].imshow(recon[recon.shape[0] // 2, :, :], cmap='gray')
    plt.tight_layout()
    plt.savefig(outpath, dpi=50)
    plt.close()


@torch.inference_mode()
def save_examples(output_path, vae, ldm, condition, device, latent_shape, vae_space_shape) -> None:
    """
    This function samples new images using a diffusion model and saves an example output.
    It combines VAE decoding and diffusion sampling to visualize generative results.
    """
    img = sample_using_diffusion(vae, ldm, condition, device, latent_shape, vae_space_shape)
    img = img[0,0].cpu().numpy()
    _, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()
    ax.imshow(img, cmap='gray')
    plt.savefig(output_path, dpi=50)
    plt.close()


class KLDivergenceLoss:
    """
    This class computes the Kullback–Leibler (KL) divergence loss used in VAEs.
    It measures how much the learned latent distribution deviates from the standard normal prior.
    """
    def __call__(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        return torch.sum(kl_loss) / kl_loss.shape[0]


class HistoryLogger:
    """
    This class logs training metrics to a CSV file for persistent tracking of model performance.
    It initializes the log file with specified column headers and appends new entries at each step.
    """

    def __init__(self, filepath: str, columns: list):
        self.filepath = filepath
        self.columns = columns
        self._initialize()
        
    def _initialize(self) -> None:
        with open(self.filepath, 'w+') as f:
            f.write(','.join(self.columns) + '\n')

    def log(self, values: dict) -> None:
        with open(self.filepath, 'a+') as f:
            line = ','.join([str(values[c]) for c in self.columns]) + '\n'
            f.write(line)


class AverageLoss:
    """
    This class tracks, averages, and logs multiple training losses or metrics.
    It supports both TensorBoard logging and CSV export for flexible monitoring.
    """

    def __init__(self):
        self.losses_accumulator = {}
    
    def put(self, loss_key:str, loss_value:Union[int,float]) -> None:
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)
    
    def pop_avg(self, loss_key:str) -> float:
        if loss_key not in self.losses_accumulator: return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)
    
    def to_tensorboard(self, writer: SummaryWriter, step: int) -> None:
        for metric_key in self.losses_accumulator.keys():
            writer.add_scalar(metric_key, self.pop_avg(metric_key), step)
            
    def to_csv(self, filepath: str, step: int) -> None:
        metrics = [ str(round(self.pop_avg(k), 4)) for k in self.losses_accumulator.keys() ]
        with open(filepath, 'a+') as f:
            line = ','.join(([str(step)] + metrics)) + '\n' 
            f.write(line)
                    

class GradientAccumulation:
    """
    This class implements gradient accumulation to simulate larger effective batch sizes
    than what can be processed in memory. It optionally supports mixed-precision training
    through the use of PyTorch’s GradScaler.
    """

    def __init__(self,
                 actual_batch_size: int, 
                 expect_batch_size: int,
                 loader_len: int,
                 optimizer: torch.optim.Optimizer, 
                 grad_scaler: Optional[GradScaler] = None) -> None:
        """
        This function initializes the GradientAccumulation instance with all required parameters.
        It checks that the expected batch size is divisible by the actual batch size, 
        and calculates how many steps are needed before performing an optimizer update.
        """

        assert expect_batch_size % actual_batch_size == 0, \
            'expect_batch_size must be divisible by actual_batch_size'
        
        self.actual_batch_size = actual_batch_size
        self.expect_batch_size = expect_batch_size
        self.loader_len = loader_len
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.steps_until_update = expect_batch_size / actual_batch_size


    def step(self, loss: torch.Tensor, step: int) -> None:
        """
        This function performs a backward pass and updates the model parameters
        only when the number of accumulated steps matches the configured threshold.
        It supports both standard and mixed-precision training.
        """
        loss = loss / self.steps_until_update
        if self.grad_scaler is not None: self.grad_scaler.scale(loss).backward()
        else: loss.backward()  

        if (step + 1) % self.steps_until_update == 0 or (step + 1) == self.loader_len:
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)