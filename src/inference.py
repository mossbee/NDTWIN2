import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import NDTwinDataset, FaceAligner
from models.attention_network import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TwinVerifier:
    """Inference class for twin verification"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: torch.device):
        self.device = device
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create model
        self.model = create_model(self.config).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize face aligner
        self.face_aligner = FaceAligner((self.config['data']['image_size'], 
                                       self.config['data']['image_size']))
        
        # Setup transforms
        self.transforms = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Best validation AUC: {checkpoint.get('best_val_auc', 'N/A')}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image for inference"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Face alignment
        aligned_face = self.face_aligner.align_face(image)
        if aligned_face is not None:
            image = aligned_face
        else:
            # Fallback to simple resize
            image = cv2.resize(image, (self.config['data']['image_size'], 
                                     self.config['data']['image_size']))
        
        # Apply transforms
        transformed = self.transforms(image=image)
        tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def verify_pair(self, image1_path: str, image2_path: str) -> Dict:
        """Verify if two images are of the same person"""
        with torch.no_grad():
            # Preprocess images
            img1 = self.preprocess_image(image1_path).to(self.device)
            img2 = self.preprocess_image(image2_path).to(self.device)
            
            # Forward pass
            outputs = self.model(img1, img2)
            
            # Compute similarity
            similarity = self.model.get_similarity(img1, img2).item()
            
            return {
                'similarity': similarity,
                'embedding1': outputs['embedding1'].cpu().numpy(),
                'embedding2': outputs['embedding2'].cpu().numpy(),
                'attention_map1': outputs['attention_map1'].cpu().numpy(),
                'attention_map2': outputs['attention_map2'].cpu().numpy(),
                'prediction': similarity > 0.5  # Default threshold
            }
    
    def batch_verify(self, image_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Verify multiple image pairs"""
        results = []
        
        for img1_path, img2_path in image_pairs:
            try:
                result = self.verify_pair(img1_path, img2_path)
                result['image1_path'] = img1_path
                result['image2_path'] = img2_path
                results.append(result)
            except Exception as e:
                print(f"Error processing pair ({img1_path}, {img2_path}): {e}")
                results.append({
                    'image1_path': img1_path,
                    'image2_path': img2_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_attention(self, image_path: str, attention_map: np.ndarray, 
                          save_path: str = None) -> None:
        """Visualize attention map overlaid on original image"""
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config['data']['image_size'], 
                                 self.config['data']['image_size']))
        
        # Resize attention map to match image
        attention = attention_map.squeeze()
        if attention.shape != image.shape[:2]:
            attention = cv2.resize(attention, (image.shape[1], image.shape[0]))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        im1 = axes[1].imshow(attention, cmap='hot', interpolation='bilinear')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        overlay = image.copy().astype(float)
        attention_3d = np.stack([attention] * 3, axis=-1)
        overlay = overlay * 0.7 + attention_3d * 255 * 0.3
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


class ModelEvaluator:
    """Evaluate model performance on test set"""
    
    def __init__(self, verifier: TwinVerifier):
        self.verifier = verifier
    
    def evaluate_dataset(self, dataset_path: str, pairs_file: str, 
                        output_dir: str = 'evaluation_results') -> Dict:
        """Evaluate model on full dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test dataset
        test_dataset = NDTwinDataset(
            dataset_path=dataset_path,
            pairs_file=pairs_file,
            mode='test',
            image_size=self.verifier.config['data']['image_size'],
            use_face_alignment=False,  # We'll handle this in verifier
            augment=False
        )
        
        print(f"Evaluating on {len(test_dataset)} pairs...")
        
        # Collect predictions
        similarities = []
        labels = []
        twin_pairs = []
        attention_maps = []
        
        for i in range(len(test_dataset)):
            try:
                sample = test_dataset[i]
                folder1, img1, folder2, img2, label = test_dataset.pairs[i]
                
                # Get image paths
                img1_path = os.path.join(dataset_path, folder1, img1)
                img2_path = os.path.join(dataset_path, folder2, img2)
                
                # Verify pair
                result = self.verifier.verify_pair(img1_path, img2_path)
                
                similarities.append(result['similarity'])
                labels.append(label)
                twin_pairs.append(sample['is_twin_pair'].item())
                
                if len(attention_maps) < 20:  # Store first 20 for visualization
                    attention_maps.append({
                        'attention1': result['attention_map1'],
                        'attention2': result['attention_map2'],
                        'image1_path': img1_path,
                        'image2_path': img2_path,
                        'label': label
                    })
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(test_dataset)} pairs")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        similarities = np.array(similarities)
        labels = np.array(labels)
        twin_pairs = np.array(twin_pairs)
        
        # Compute metrics
        metrics = self._compute_metrics(similarities, labels, twin_pairs)
        
        # Generate plots
        self._plot_results(similarities, labels, twin_pairs, output_dir)
        
        # Visualize attention maps
        self._visualize_attention_samples(attention_maps, output_dir)
        
        # Save results
        results = {
            'metrics': metrics,
            'similarities': similarities.tolist(),
            'labels': labels.tolist(),
            'twin_pairs': twin_pairs.tolist()
        }
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _compute_metrics(self, similarities: np.ndarray, labels: np.ndarray, 
                        twin_pairs: np.ndarray) -> Dict:
        """Compute evaluation metrics"""
        # Overall metrics
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find best threshold
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        predictions = (similarities > best_threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Separate metrics for twin vs non-twin pairs
        twin_mask = twin_pairs == 1
        non_twin_mask = twin_pairs == 0
        
        twin_similarities = similarities[twin_mask]
        twin_labels = labels[twin_mask]
        non_twin_similarities = similarities[non_twin_mask]
        non_twin_labels = labels[non_twin_mask]
        
        # Twin pair metrics (hardest negatives)
        twin_fpr, twin_tpr, _ = roc_curve(twin_labels, twin_similarities)
        twin_auc = auc(twin_fpr, twin_tpr)
        
        # Non-twin pair metrics
        if len(non_twin_similarities) > 0:
            non_twin_fpr, non_twin_tpr, _ = roc_curve(non_twin_labels, non_twin_similarities)
            non_twin_auc = auc(non_twin_fpr, non_twin_tpr)
        else:
            non_twin_auc = 0.0
        
        metrics = {
            'overall_auc': float(roc_auc),
            'twin_pairs_auc': float(twin_auc),
            'non_twin_pairs_auc': float(non_twin_auc),
            'best_threshold': float(best_threshold),
            'accuracy': float(np.mean(predictions == labels)),
            'precision': float(cm[1, 1] / (cm[1, 1] + cm[0, 1])) if cm[1, 1] + cm[0, 1] > 0 else 0.0,
            'recall': float(cm[1, 1] / (cm[1, 1] + cm[1, 0])) if cm[1, 1] + cm[1, 0] > 0 else 0.0,
            'f1_score': 0.0,
            'confusion_matrix': cm.tolist()
        }
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        
        return metrics
    
    def _plot_results(self, similarities: np.ndarray, labels: np.ndarray, 
                     twin_pairs: np.ndarray, output_dir: str):
        """Generate evaluation plots"""
        
        # ROC Curve
        plt.figure(figsize=(10, 8))
        
        # Overall ROC
        fpr, tpr, _ = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Overall (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Twin pairs ROC
        twin_mask = twin_pairs == 1
        if np.sum(twin_mask) > 0:
            twin_fpr, twin_tpr, _ = roc_curve(labels[twin_mask], similarities[twin_mask])
            twin_auc = auc(twin_fpr, twin_tpr)
            plt.plot(twin_fpr, twin_tpr, label=f'Twin Pairs (AUC = {twin_auc:.3f})', linewidth=2)
        
        # Non-twin pairs ROC
        non_twin_mask = twin_pairs == 0
        if np.sum(non_twin_mask) > 0:
            nt_fpr, nt_tpr, _ = roc_curve(labels[non_twin_mask], similarities[non_twin_mask])
            nt_auc = auc(nt_fpr, nt_tpr)
            plt.plot(nt_fpr, nt_tpr, label=f'Non-Twin Pairs (AUC = {nt_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Twin Verification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Similarity distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(similarities[labels == 1], bins=50, alpha=0.7, label='Same Person', density=True)
        plt.hist(similarities[labels == 0], bins=50, alpha=0.7, label='Different Person', density=True)
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Similarity Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Separate by twin pairs
        twin_pos = similarities[(labels == 1) & (twin_pairs == 1)]
        twin_neg = similarities[(labels == 0) & (twin_pairs == 1)]
        non_twin_pos = similarities[(labels == 1) & (twin_pairs == 0)]
        non_twin_neg = similarities[(labels == 0) & (twin_pairs == 0)]
        
        plt.hist(twin_pos, bins=30, alpha=0.7, label='Twin - Same Person', density=True)
        plt.hist(twin_neg, bins=30, alpha=0.7, label='Twin - Different Person', density=True)
        plt.hist(non_twin_pos, bins=30, alpha=0.7, label='Non-Twin - Same Person', density=True)
        plt.hist(non_twin_neg, bins=30, alpha=0.7, label='Non-Twin - Different Person', density=True)
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Similarity by Twin Status')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_attention_samples(self, attention_maps: List[Dict], output_dir: str):
        """Visualize attention maps for sample pairs"""
        attention_dir = os.path.join(output_dir, 'attention_visualizations')
        os.makedirs(attention_dir, exist_ok=True)
        
        for i, sample in enumerate(attention_maps[:10]):  # Visualize first 10
            try:
                # Create figure for this pair
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Load images
                img1 = cv2.imread(sample['image1_path'])
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img1 = cv2.resize(img1, (224, 224))
                
                img2 = cv2.imread(sample['image2_path'])
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                img2 = cv2.resize(img2, (224, 224))
                
                # Get attention maps
                attn1 = sample['attention1'].squeeze()
                attn2 = sample['attention2'].squeeze()
                
                # Resize attention maps
                if attn1.shape != (224, 224):
                    attn1 = cv2.resize(attn1, (224, 224))
                if attn2.shape != (224, 224):
                    attn2 = cv2.resize(attn2, (224, 224))
                
                # Plot images and attention
                axes[0, 0].imshow(img1)
                axes[0, 0].set_title('Image 1')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(attn1, cmap='hot')
                axes[0, 1].set_title('Attention Map 1')
                axes[0, 1].axis('off')
                
                # Overlay
                overlay1 = img1.copy().astype(float)
                attn1_3d = np.stack([attn1] * 3, axis=-1)
                overlay1 = overlay1 * 0.7 + attn1_3d * 255 * 0.3
                overlay1 = np.clip(overlay1, 0, 255).astype(np.uint8)
                axes[0, 2].imshow(overlay1)
                axes[0, 2].set_title('Overlay 1')
                axes[0, 2].axis('off')
                
                axes[1, 0].imshow(img2)
                axes[1, 0].set_title('Image 2')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(attn2, cmap='hot')
                axes[1, 1].set_title('Attention Map 2')
                axes[1, 1].axis('off')
                
                # Overlay
                overlay2 = img2.copy().astype(float)
                attn2_3d = np.stack([attn2] * 3, axis=-1)
                overlay2 = overlay2 * 0.7 + attn2_3d * 255 * 0.3
                overlay2 = np.clip(overlay2, 0, 255).astype(np.uint8)
                axes[1, 2].imshow(overlay2)
                axes[1, 2].set_title('Overlay 2')
                axes[1, 2].axis('off')
                
                # Add title with label information
                label_text = "Same Person" if sample['label'] == 1 else "Different Person"
                fig.suptitle(f'Sample {i+1}: {label_text}', fontsize=16)
                
                plt.tight_layout()
                plt.savefig(os.path.join(attention_dir, f'attention_sample_{i+1}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error visualizing attention for sample {i}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Twin Verification Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'evaluate'], 
                       default='single', help='Inference mode')
    parser.add_argument('--image1', type=str, help='Path to first image (single mode)')
    parser.add_argument('--image2', type=str, help='Path to second image (single mode)')
    parser.add_argument('--pairs_file', type=str, help='Path to pairs file (batch mode)')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset (evaluate mode)')
    parser.add_argument('--pairs_json', type=str, help='Path to pairs.json (evaluate mode)')
    parser.add_argument('--output_dir', type=str, default='inference_results', 
                       help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create verifier
    verifier = TwinVerifier(args.checkpoint, args.config, device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'single':
        if not args.image1 or not args.image2:
            print("Error: --image1 and --image2 required for single mode")
            return
        
        print(f"Verifying pair: {args.image1} vs {args.image2}")
        result = verifier.verify_pair(args.image1, args.image2)
        
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Prediction: {'Same Person' if result['prediction'] else 'Different Person'}")
        
        # Visualize attention
        verifier.visualize_attention(
            args.image1, 
            result['attention_map1'],
            save_path=os.path.join(args.output_dir, 'attention_image1.png')
        )
        verifier.visualize_attention(
            args.image2, 
            result['attention_map2'],
            save_path=os.path.join(args.output_dir, 'attention_image2.png')
        )
    
    elif args.mode == 'evaluate':
        if not args.dataset_path or not args.pairs_json:
            print("Error: --dataset_path and --pairs_json required for evaluate mode")
            return
        
        print("Evaluating model on test dataset...")
        evaluator = ModelEvaluator(verifier)
        results = evaluator.evaluate_dataset(
            args.dataset_path, 
            args.pairs_json, 
            args.output_dir
        )
        
        print("\nEvaluation Results:")
        print(f"Overall AUC: {results['metrics']['overall_auc']:.4f}")
        print(f"Twin Pairs AUC: {results['metrics']['twin_pairs_auc']:.4f}")
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"Best Threshold: {results['metrics']['best_threshold']:.4f}")
    
    else:
        print("Batch mode not implemented yet")


if __name__ == "__main__":
    main()
