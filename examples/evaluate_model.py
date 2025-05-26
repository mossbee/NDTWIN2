#!/usr/bin/env python3
"""
Example script for evaluating a trained model on the ND_TWIN test set.

This script demonstrates how to load a trained model and evaluate its performance
with comprehensive metrics and visualizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
import json
from pathlib import Path

from src.data.dataset import NDTwinDataset
from src.models.attention_network import SiameseLocalGlobalNet
from src.inference import ModelEvaluator, TwinVerifier
from src.utils.visualization import plot_evaluation_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ND_TWIN dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for evaluation results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate attention visualizations')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create dataset
    print(f"Creating {args.split} dataset...")
    dataset = NDTwinDataset(
        data_root=args.data_root,
        split=args.split,
        **config['data']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device=device)
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate(dataset, batch_size=config['training']['batch_size'])
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"EER: {results['eer']:.4f}")
    print(f"TAR@FAR=0.1%: {results['tar_at_far_001']:.4f}")
    print(f"TAR@FAR=1%: {results['tar_at_far_01']:.4f}")
    
    # Save results
    results_file = output_dir / f'evaluation_results_{args.split}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Plot results
    plot_path = output_dir / f'evaluation_plots_{args.split}.png'
    plot_evaluation_results(results, save_path=str(plot_path))
    print(f"Evaluation plots saved to {plot_path}")
    
    # Generate attention visualizations if requested
    if args.visualize:
        print("Generating attention visualizations...")
        vis_dir = output_dir / 'attention_visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Create verifier for visualization
        verifier = TwinVerifier(model, device=device)
        
        # Sample some pairs for visualization
        num_samples = min(20, len(dataset))
        for i in range(num_samples):
            sample = dataset[i]
            img1, img2 = sample['image1'], sample['image2']
            label = sample['label']
            
            # Get attention maps
            similarity, attention_maps = verifier.verify_with_attention(img1, img2)
            
            # Save visualization
            vis_path = vis_dir / f'sample_{i:03d}_label_{label}_sim_{similarity:.3f}.png'
            verifier.visualize_attention(img1, img2, attention_maps, save_path=str(vis_path))
        
        print(f"Attention visualizations saved to {vis_dir}")
    
    # Save predictions if requested
    if args.save_predictions:
        print("Saving predictions...")
        predictions_file = output_dir / f'predictions_{args.split}.json'
        
        predictions = []
        for i in range(len(dataset)):
            sample = dataset[i]
            img1, img2 = sample['image1'], sample['image2']
            label = sample['label'].item()
            
            # Get prediction
            with torch.no_grad():
                img1_batch = img1.unsqueeze(0).to(device)
                img2_batch = img2.unsqueeze(0).to(device)
                similarity = model(img1_batch, img2_batch).item()
            
            predictions.append({
                'index': i,
                'true_label': label,
                'similarity': similarity,
                'predicted_label': int(similarity > 0.5)
            })
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to {predictions_file}")
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()
