import torch
import torch.nn as nn
import torch.nn.functional as F

# This class implements a Generalized Mean (GeM) pooling layer.
# It performs spatial aggregation parameterized by a learnable exponent, allowing the network
# to dynamically adapt its behavior between average pooling (when p=1) and max pooling (as p approaches infinity).
# Author: Antonio Scardace

class GeM(nn.Module):

    def __init__(self, p: float, eps: float) -> None:
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = x.clamp(min=self.eps)
        return F.avg_pool2d(x_clamped.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)