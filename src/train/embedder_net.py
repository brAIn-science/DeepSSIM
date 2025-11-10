import torch
import torch.nn as nn
import torchvision.models as models

# This class is the feature extraction network used in the Siamese model to produce image embeddings.
# It uses ConvNeXt, a modern and productive convolutional backbone well-suited for visual similarity tasks.
# The first two stages of ConvNeXt are frozen to preserve low-level features and reduce training time and overfitting.
# The extracted features are projected to produce compact embeddings of a fixed dimension.
# Author: Antonio Scardace

class ImageEmbedder(nn.Module):

    def __init__(self, embedding_dim: int, dropout_prob: float) -> None:
        super().__init__()
        base_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        for name, param in base_model.features.named_parameters():
            if '0' in name or '1' in name:
                param.requires_grad = False

        self.feature_extractor = base_model.features
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), 
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.fc(x)  
        return x