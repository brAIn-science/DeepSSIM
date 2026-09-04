import torch
import torch.nn as nn
import torchvision.models as models

from src.utils.gem import GeM

# This class implements the feature extractor for the Siamese network to generate image embeddings.
# It relies on a ConvNeXt-Tiny backbone to extract multi-scale representations across four stages.
# The stem and first stage are frozen to preserve generalized low-level visual features.
# Intermediate feature maps are aggregated via GeM pooling and fused through a funnel MLP head.
# Author: Antonio Scardace

class ImageEmbedder(nn.Module):
    
    def __init__(self, embedding_dim: int, dropout_prob: float) -> None:
        super().__init__()
        base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        layers_to_freeze = [0, 1]
        for idx in layers_to_freeze:
            for param in base_model.features[idx].parameters():
                param.requires_grad = False
        
        features = base_model.features
        self.stage_a = nn.Sequential(*features[0:2])
        self.stage_b = nn.Sequential(*features[2:4])
        self.stage_c = nn.Sequential(*features[4:6])
        self.stage_d = nn.Sequential(*features[6:8])

        self.gem_a = GeM(p=3.0, eps=1e-6)
        self.gem_b = GeM(p=3.0, eps=1e-6)
        self.gem_c = GeM(p=3.0, eps=1e-6)
        self.gem_d = GeM(p=3.0, eps=1e-6)
        
        dim_a, dim_b, dim_c, dim_d = [96, 192, 384, 768]
        total_dim = dim_a + dim_b + dim_c + dim_d
        hidden_dim = (total_dim + embedding_dim) // 2
        
        self.fc = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_a = self.stage_a(x)
        x_b = self.stage_b(x_a)
        x_c = self.stage_c(x_b)
        x_d = self.stage_d(x_c)
        
        pooled_a = self.gem_a(x_a).flatten(1)
        pooled_b = self.gem_b(x_b).flatten(1)
        pooled_c = self.gem_c(x_c).flatten(1)
        pooled_d = self.gem_d(x_d).flatten(1)
        
        x = torch.cat([pooled_a, pooled_b, pooled_c, pooled_d], dim=1).to(torch.float32)
        x = self.fc(x)
        return x