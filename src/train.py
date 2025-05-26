import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple
import json
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import create_data_loaders
from models.attention_network import create_model
from models.loss_functions import create_loss_function


class Trainer:
    """Trainer class for the Local-Global Attention Network"""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Create model
        self.model = create_model(config).to(device)
        
        # Create loss function
        self.criterion = create_loss_function(config)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_aucs = []
        
        # Best model tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        
        # Create directories
        os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(config['logging']['log_dir'])
        
    def _create_optimizer(self):
        """Create optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config['training']['scheduler']
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=self.config['training']['patience'] // 2
            )
        else:
            return None
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'main': 0.0,
            'global': 0.0,
            'local': 0.0,
            'attention_reg': 0.0
        }
        
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(image1, image2)
            
            # Compute losses
            losses = self.criterion(outputs, labels)
            
            # Backward pass
            losses['total'].backward()
            self.optimizer.step()
            
            # Update metrics
            for key, loss in losses.items():
                total_losses[key] += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to tensorboard
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                step = len(train_loader) * (len(self.train_losses)) + batch_idx
                for key, loss in losses.items():
                    self.writer.add_scalar(f'train/{key}_loss', loss.item(), step)
        
        # Average losses
        avg_losses = {key: total_loss / num_batches for key, total_loss in total_losses.items()}
        return avg_losses
    
    def validate(self, val_loader) -> Tuple[Dict[str, float], float, float]:
        """Validate the model"""
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'main': 0.0,
            'global': 0.0,
            'local': 0.0,
            'attention_reg': 0.0
        }
        
        all_similarities = []
        all_labels = []
        attention_maps = []
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move data to device
                image1 = batch['image1'].to(self.device)
                image2 = batch['image2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(image1, image2)
                
                # Compute losses
                losses = self.criterion(outputs, labels)
                
                # Update metrics
                for key, loss in losses.items():
                    total_losses[key] += loss.item()
                
                # Compute similarities
                similarities = self.model.get_similarity(image1, image2)
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Store attention maps for visualization
                if len(attention_maps) < 10:  # Store first 10 batches
                    attention_maps.append(outputs['attention_map1'].cpu())
        
        # Average losses
        avg_losses = {key: total_loss / num_batches for key, total_loss in total_losses.items()}
        
        # Compute metrics
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        # ROC AUC
        auc = roc_auc_score(all_labels, all_similarities)
        
        # Find best threshold
        best_acc = 0.0
        best_threshold = 0.5
        
        thresholds = np.arange(
            self.config['validation']['threshold_start'],
            self.config['validation']['threshold_end'],
            self.config['validation']['threshold_step']
        )
        
        for threshold in thresholds:
            predictions = (all_similarities > threshold).astype(int)
            acc = accuracy_score(all_labels, predictions)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        # Visualize attention maps
        if self.config['logging']['visualize_attention'] and attention_maps:
            self._visualize_attention(attention_maps, len(self.val_losses))
        
        return avg_losses, auc, best_acc
    
    def _visualize_attention(self, attention_maps, epoch):
        """Visualize attention maps"""
        # Concatenate first few attention maps
        maps = torch.cat(attention_maps[:2], dim=0)  # Take first 2 batches
        maps = maps[:8]  # Take first 8 samples
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, attention_map in enumerate(maps):
            if i >= 8:
                break
            
            # Convert to numpy and squeeze
            attn = attention_map.squeeze().numpy()
            
            # Plot
            axes[i].imshow(attn, cmap='hot', interpolation='bilinear')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['logging']['log_dir'], f'attention_epoch_{epoch}.png'))
        plt.close()
        
        # Log to tensorboard
        self.writer.add_figure('attention_maps', fig, epoch)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_auc': self.best_val_auc,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint']['save_dir'], 
            'latest_checkpoint.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint']['save_dir'], 
                'best_checkpoint.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved with AUC: {self.best_val_auc:.4f}")
        
        # Save periodic checkpoints
        if epoch % self.config['checkpoint']['save_frequency'] == 0:
            epoch_path = os.path.join(
                self.config['checkpoint']['save_dir'], 
                f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_auc = checkpoint['best_val_auc']
        
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader, resume_from: str = None):
        """Main training loop"""
        start_epoch = 0
        
        # Resume training if checkpoint provided
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed training from epoch {start_epoch}")
        
        # Freeze backbone for initial epochs if specified
        freeze_epochs = self.config['model'].get('freeze_backbone_epochs', 0)
        if freeze_epochs > 0:
            print(f"Freezing backbone for first {freeze_epochs} epochs")
            self.model.base_network.freeze_backbone(True)
        
        patience_counter = 0
        
        for epoch in range(start_epoch, self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Unfreeze backbone after specified epochs
            if epoch == freeze_epochs:
                print("Unfreezing backbone")
                self.model.base_network.freeze_backbone(False)
            
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses, val_auc, val_acc = self.validate(val_loader)
            self.val_losses.append(val_losses)
            self.val_aucs.append(val_auc)
            self.val_accuracies.append(val_acc)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_auc)
                else:
                    self.scheduler.step()
            
            # Log metrics
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f}")
            print(f"Val AUC: {val_auc:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            
            # Tensorboard logging
            self.writer.add_scalar('val/auc', val_auc, epoch)
            self.writer.add_scalar('val/accuracy', val_acc, epoch)
            self.writer.add_scalar('val/total_loss', val_losses['total'], epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for best model
            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if self.config['checkpoint']['save_best_only']:
                if is_best:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if patience_counter >= self.config['training']['patience']:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch + 1}")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'val_accuracies': self.val_accuracies,
            'best_val_auc': self.best_val_auc,
            'best_epoch': self.best_epoch
        }
        
        history_path = os.path.join(self.config['logging']['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Local-Global Attention Network for Twin Verification')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--pairs_file', type=str, required=True, help='Path to pairs.json file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['data']['dataset_path'] = args.dataset_path
    config['data']['pairs_file'] = args.pairs_file
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=args.dataset_path,
        pairs_file=args.pairs_file,
        config=config,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create trainer
    trainer = Trainer(config, device)
    
    # Start training
    trainer.train(train_loader, val_loader, resume_from=args.resume)


if __name__ == "__main__":
    main()
