import torch.nn as nn

from torch.optim import AdamW

# This function builds an AdamW optimizer using Layer-wise Learning Rate Decay (LLRD).
# The classification head and GeM pooling layers use the base learning rate.
# Progressively lower learning rates are assigned to deeper backbone stages according to the specified decay factor.
# Weight decay is applied uniformly across all parameter groups.
# Author: Antonio Scardace

def build_llrd_optimizer(model: nn.Module, base_lr: float, decay_factor: float, weight_decay: float) -> AdamW:

    embedder = model.embedding_net
    backbone_stages = [
        embedder.stage_d, 
        embedder.stage_c, 
        embedder.stage_b, 
        embedder.stage_a
    ]

    gem_params = (
        list(embedder.gem_a.parameters()) + 
        list(embedder.gem_b.parameters()) + 
        list(embedder.gem_c.parameters()) + 
        list(embedder.gem_d.parameters())
    )
    param_groups = [
        {'params': embedder.fc.parameters(), 'lr': base_lr},
        {'params': gem_params, 'lr': base_lr},
    ]
    
    for depth, stage in enumerate(backbone_stages, start=1):
        decayed_lr = base_lr * (decay_factor ** depth)
        trainable_params = [p for p in stage.parameters() if p.requires_grad]
        if trainable_params:
            param_groups.append({ 'params': trainable_params, 'lr': decayed_lr })

    return AdamW(param_groups, weight_decay=weight_decay)