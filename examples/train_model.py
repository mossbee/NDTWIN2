#!/usr/bin/env python3
"""
Example script for training the Local-Global Attention Network on ND_TWIN dataset.

This script demonstrates how to set up and train the model with proper configuration
and monitoring.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
from pathlib import Path

from src.data.dataset import NDTwinDataset
from src.models.attention_network import SiameseLocalGlobalNet
from src.train import Trainer
from src.utils.dataset_utils import validate_dataset


def main():
    parser = argparse.ArgumentParser(description='Train Local-Global Attention Network')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ND_TWIN dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--validate_data', action='store_true',
                        help='Validate dataset before training')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate dataset if requested
    if args.validate_data:
        print("Validating dataset...")
        validation_results = validate_dataset(args.data_root)
        if not validation_results['valid']:
            print(f"Dataset validation failed: {validation_results}")
            return
        print("Dataset validation passed!")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = NDTwinDataset(
        data_root=args.data_root,
        split='train',
        **config['data']
    )
    
    val_dataset = NDTwinDataset(
        data_root=args.data_root,
        split='val',
        **config['data']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = SiameseLocalGlobalNet(
        backbone=config['model']['backbone'],
        num_classes=config['model']['num_classes'],
        attention_type=config['model']['attention_type'],
        feature_dim=config['model']['feature_dim'],
        local_regions=config['model']['local_regions'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        save_dir=str(output_dir),
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
