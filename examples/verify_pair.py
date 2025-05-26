#!/usr/bin/env python3
"""
Example script for verifying if two face images belong to the same person.

This script demonstrates how to use the trained model for single pair verification
with attention visualization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from src.models.attention_network import SiameseLocalGlobalNet
from src.inference import TwinVerifier
from src.data.dataset import get_transform


def main():
    parser = argparse.ArgumentParser(description='Verify if two images are of the same person')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--image1', type=str, required=True,
                        help='Path to first image')
    parser.add_argument('--image2', type=str, required=True,
                        help='Path to second image')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save visualization (optional)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold for verification')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Loading model...")
    model = SiameseLocalGlobalNet(
        backbone=config['model']['backbone'],
        num_classes=config['model']['num_classes'],
        attention_type=config['model']['attention_type'],
        feature_dim=config['model']['feature_dim'],
        local_regions=config['model']['local_regions'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded from {args.model_path}")
    
    # Create verifier
    verifier = TwinVerifier(model, device=device)
    
    # Load and preprocess images
    print("Loading images...")
    transform = get_transform(config['data']['image_size'], is_training=False)
    
    try:
        img1 = Image.open(args.image1).convert('RGB')
        img2 = Image.open(args.image2).convert('RGB')
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Apply transforms
    img1_tensor = transform(img1)
    img2_tensor = transform(img2)
    
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    
    # Perform verification
    print("Performing verification...")
    similarity, attention_maps = verifier.verify_with_attention(img1_tensor, img2_tensor)
    
    # Make decision
    is_same_person = similarity > args.threshold
    
    # Print results
    print(f"\nVerification Results:")
    print(f"Similarity Score: {similarity:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Same Person: {'YES' if is_same_person else 'NO'}")
    print(f"Confidence: {similarity if is_same_person else (1 - similarity):.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')
    
    # Similarity result
    axes[0, 2].text(0.5, 0.5, f'Same Person: {"YES" if is_same_person else "NO"}\n'
                                f'Similarity: {similarity:.4f}\n'
                                f'Confidence: {similarity if is_same_person else (1 - similarity):.4f}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[0, 2].transAxes, fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor="lightgreen" if is_same_person else "lightcoral"))
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Verification Result')
    
    # Attention maps
    if attention_maps:
        global_attention = attention_maps.get('global_attention')
        local_attention = attention_maps.get('local_attention')
        
        if global_attention is not None:
            axes[1, 0].imshow(global_attention, cmap='hot', alpha=0.7)
            axes[1, 0].imshow(img1, alpha=0.3)
            axes[1, 0].set_title('Global Attention (Image 1)')
            axes[1, 0].axis('off')
        
        if local_attention is not None:
            axes[1, 1].imshow(local_attention, cmap='hot', alpha=0.7)
            axes[1, 1].imshow(img2, alpha=0.3)
            axes[1, 1].set_title('Local Attention (Image 2)')
            axes[1, 1].axis('off')
        
        # Combined attention
        if global_attention is not None and local_attention is not None:
            combined = (global_attention + local_attention) / 2
            axes[1, 2].imshow(combined, cmap='hot')
            axes[1, 2].set_title('Combined Attention')
            axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save or show visualization
    if args.output_path:
        plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    main()
