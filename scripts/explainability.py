import torch
import argparse
import matplotlib.pyplot as plt

from src.train.embedder_net import ImageEmbedder
from src.utils.layercam import (
    load_and_format,
    get_target_layers,
    generate_fused_heatmap,
    create_overlay
)

# This script visualizes the spatial regions driving the similarity prediction between two images.
# It computes multi-scale LayerCAM by backpropagating the cosine similarity through the embedding network.
# Author: Antonio Scardace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--real_image_path',  type=str,   required=True, help='Path to the real training image.')
    parser.add_argument('--synth_image_path', type=str,   required=True, help='Path to the synthetic target image.')
    parser.add_argument('--model_path',       type=str,   required=True, help='Path to the pre-trained embedder model weights.')
    parser.add_argument('--alpha',            type=float, default=6e-1,  help='Alpha transparency value for the heatmap overlay.')
    parser.add_argument('--dropout_prob',     type=float, default=1e-1,  help='Dropout probability for the embedder network.')
    parser.add_argument('--emb_dim',          type=int,   default=256,   help='Dimensionality of the extracted image embeddings.')
    args = parser.parse_args()

    # Initializes the embedder on the available hardware and loads the pre-trained weights.
    # Enforces the evaluation mode to guarantee a deterministic feature extraction.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = ImageEmbedder(embedding_dim=args.emb_dim, dropout_prob=args.dropout_prob).to(device)
    embedder.load_state_dict(torch.jit.load(args.model_path, map_location=device).state_dict())
    embedder.eval()
    
    # Prepares the input tensors and generates the fused multi-scale activation heatmap.
    # Renders the superimposed overlay alongside the original image pair.

    img_1_tensor, img_1_display = load_and_format(args.real_image_path, device)
    img_2_tensor, img_2_display = load_and_format(args.synth_image_path, device)

    target_layers = get_target_layers(embedder)
    fused_heatmap = generate_fused_heatmap(embedder, img_1_tensor, img_2_tensor, target_layers)
    overlay = create_overlay(img_1_display, fused_heatmap, alpha=args.alpha)

    plot_data = [
        (img_1_display, 'Real Image', 'gray'),
        (img_2_display, 'Synthetic Image', 'gray'),
        (overlay, 'LayerCAM', None)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7, 4), gridspec_kw={ 'wspace': 0.05 })
    for ax, (img, title, cmap) in zip(axes, plot_data):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    cax = axes[2].inset_axes([1.05, 0.0, 0.05, 1.0])
    fig.colorbar(plt.cm.ScalarMappable(cmap='turbo'), cax=cax, ticks=[0, 1]).set_ticklabels(['0', '1'])
    plt.show()
    plt.close()