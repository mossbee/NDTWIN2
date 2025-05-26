import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Tuple
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json


def plot_training_history(history_file: str, save_path: str = None):
    """Plot training history from JSON file"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    epochs = range(1, len(history['train_losses']) + 1)
    train_total = [loss['total'] for loss in history['train_losses']]
    val_total = [loss['total'] for loss in history['val_losses']]
    
    axes[0, 0].plot(epochs, train_total, 'b-', label='Training')
    axes[0, 0].plot(epochs, val_total, 'r-', label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Component losses
    train_global = [loss['global'] for loss in history['train_losses']]
    train_local = [loss['local'] for loss in history['train_losses']]
    train_attention = [loss['attention_reg'] for loss in history['train_losses']]
    
    axes[0, 1].plot(epochs, train_global, label='Global')
    axes[0, 1].plot(epochs, train_local, label='Local')
    axes[0, 1].plot(epochs, train_attention, label='Attention Reg')
    axes[0, 1].set_title('Training Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation metrics
    axes[1, 0].plot(epochs, history['val_aucs'], 'g-', label='AUC')
    axes[1, 0].plot(epochs, history['val_accuracies'], 'orange', label='Accuracy')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss comparison
    axes[1, 1].plot(epochs, train_total, 'b-', alpha=0.7, label='Train Total')
    axes[1, 1].plot(epochs, val_total, 'r-', alpha=0.7, label='Val Total')
    axes[1, 1].axvline(x=history['best_epoch'] + 1, color='green', linestyle='--', 
                      label=f'Best Epoch ({history["best_epoch"] + 1})')
    axes[1, 1].set_title('Loss Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_embeddings(embeddings: np.ndarray, labels: np.ndarray, 
                        method: str = 'tsne', save_path: str = None):
    """Visualize embeddings using dimensionality reduction"""
    
    # Reduce dimensions
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot different classes
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                   c=[color], label=f'Person {label}', alpha=0.7, s=50)
    
    plt.title(f'Embedding Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    # Only show legend if reasonable number of classes
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_attention_comparison(image1_path: str, image2_path: str,
                                 attention1: np.ndarray, attention2: np.ndarray,
                                 similarity: float, is_same_person: bool,
                                 save_path: str = None):
    """Compare attention maps for two images"""
    
    # Load images
    img1 = cv2.imread(image1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (224, 224))
    
    img2 = cv2.imread(image2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (224, 224))
    
    # Process attention maps
    attn1 = attention1.squeeze()
    attn2 = attention2.squeeze()
    
    if attn1.shape != (224, 224):
        attn1 = cv2.resize(attn1, (224, 224))
    if attn2.shape != (224, 224):
        attn2 = cv2.resize(attn2, (224, 224))
    
    # Create overlays
    overlay1 = img1.copy().astype(float)
    attn1_3d = np.stack([attn1] * 3, axis=-1)
    overlay1 = overlay1 * 0.7 + attn1_3d * 255 * 0.3
    overlay1 = np.clip(overlay1, 0, 255).astype(np.uint8)
    
    overlay2 = img2.copy().astype(float)
    attn2_3d = np.stack([attn2] * 3, axis=-1)
    overlay2 = overlay2 * 0.7 + attn2_3d * 255 * 0.3
    overlay2 = np.clip(overlay2, 0, 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Row 1: Original images
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')
    
    # Row 2: Attention maps
    im1 = axes[1, 0].imshow(attn1, cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title('Attention Map 1')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[1, 1].imshow(attn2, cmap='hot', interpolation='bilinear')
    axes[1, 1].set_title('Attention Map 2')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    # Row 3: Overlays
    axes[2, 0].imshow(overlay1)
    axes[2, 0].set_title('Attention Overlay 1')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(overlay2)
    axes[2, 1].set_title('Attention Overlay 2')
    axes[2, 1].axis('off')
    
    # Add overall title
    label_text = "Same Person" if is_same_person else "Different Person"
    fig.suptitle(f'{label_text}\nSimilarity: {similarity:.4f}', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_similarity_analysis(similarities: np.ndarray, labels: np.ndarray,
                           twin_pairs: np.ndarray = None, save_path: str = None):
    """Analyze similarity distributions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall distribution
    axes[0, 0].hist(similarities[labels == 1], bins=50, alpha=0.7, 
                   label='Same Person', density=True, color='green')
    axes[0, 0].hist(similarities[labels == 0], bins=50, alpha=0.7, 
                   label='Different Person', density=True, color='red')
    axes[0, 0].set_xlabel('Similarity Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Overall Similarity Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [similarities[labels == 1], similarities[labels == 0]]
    axes[0, 1].boxplot(data_to_plot, labels=['Same Person', 'Different Person'])
    axes[0, 1].set_ylabel('Similarity Score')
    axes[0, 1].set_title('Similarity Distribution Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    if twin_pairs is not None:
        # Twin vs non-twin analysis
        twin_mask = twin_pairs == 1
        
        # Same person - twin vs non-twin
        same_twin = similarities[(labels == 1) & twin_mask]
        same_non_twin = similarities[(labels == 1) & ~twin_mask]
        
        axes[1, 0].hist(same_twin, bins=30, alpha=0.7, label='Twins - Same Person', density=True)
        axes[1, 0].hist(same_non_twin, bins=30, alpha=0.7, label='Non-Twins - Same Person', density=True)
        axes[1, 0].set_xlabel('Similarity Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Same Person: Twin vs Non-Twin')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Different person - twin vs non-twin
        diff_twin = similarities[(labels == 0) & twin_mask]
        diff_non_twin = similarities[(labels == 0) & ~twin_mask]
        
        axes[1, 1].hist(diff_twin, bins=30, alpha=0.7, label='Twins - Different Person', density=True)
        axes[1, 1].hist(diff_non_twin, bins=30, alpha=0.7, label='Non-Twins - Different Person', density=True)
        axes[1, 1].set_xlabel('Similarity Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Different Person: Twin vs Non-Twin')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Cumulative distribution
        sorted_same = np.sort(similarities[labels == 1])
        sorted_diff = np.sort(similarities[labels == 0])
        
        y_same = np.arange(1, len(sorted_same) + 1) / len(sorted_same)
        y_diff = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
        
        axes[1, 0].plot(sorted_same, y_same, label='Same Person', linewidth=2)
        axes[1, 0].plot(sorted_diff, y_diff, label='Different Person', linewidth=2)
        axes[1, 0].set_xlabel('Similarity Score')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot with threshold
        threshold = 0.5
        correct_same = similarities[(labels == 1) & (similarities > threshold)]
        incorrect_same = similarities[(labels == 1) & (similarities <= threshold)]
        correct_diff = similarities[(labels == 0) & (similarities <= threshold)]
        incorrect_diff = similarities[(labels == 0) & (similarities > threshold)]
        
        axes[1, 1].scatter(correct_same, [1] * len(correct_same), 
                          alpha=0.6, label='Correct Same', color='green')
        axes[1, 1].scatter(incorrect_same, [1] * len(incorrect_same), 
                          alpha=0.6, label='Incorrect Same', color='red')
        axes[1, 1].scatter(correct_diff, [0] * len(correct_diff), 
                          alpha=0.6, label='Correct Different', color='blue')
        axes[1, 1].scatter(incorrect_diff, [0] * len(incorrect_diff), 
                          alpha=0.6, label='Incorrect Different', color='orange')
        
        axes[1, 1].axvline(x=threshold, color='black', linestyle='--', 
                          label=f'Threshold ({threshold})')
        axes[1, 1].set_xlabel('Similarity Score')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_title('Predictions vs True Labels')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_attention_heatmap_grid(attention_maps: List[np.ndarray], 
                                titles: List[str] = None,
                                save_path: str = None):
    """Create a grid of attention heatmaps"""
    
    n_maps = len(attention_maps)
    cols = min(4, n_maps)
    rows = (n_maps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, attn_map in enumerate(attention_maps):
        row = i // cols
        col = i % cols
        
        # Process attention map
        attn = attn_map.squeeze()
        
        # Plot
        im = axes[row, col].imshow(attn, cmap='hot', interpolation='bilinear')
        
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i])
        else:
            axes[row, col].set_title(f'Attention Map {i+1}')
        
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    
    # Hide unused subplots
    for i in range(n_maps, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None, save_path: str = None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = ['Different Person', 'Same Person']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return cm


def visualize_feature_importance(attention_maps: List[np.ndarray], 
                               image_paths: List[str] = None,
                               save_path: str = None):
    """Visualize average attention patterns"""
    
    # Stack and average attention maps
    stacked = np.stack([attn.squeeze() for attn in attention_maps])
    mean_attention = np.mean(stacked, axis=0)
    std_attention = np.std(stacked, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean attention
    im1 = axes[0].imshow(mean_attention, cmap='hot', interpolation='bilinear')
    axes[0].set_title('Average Attention Pattern')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Attention variance
    im2 = axes[1].imshow(std_attention, cmap='viridis', interpolation='bilinear')
    axes[1].set_title('Attention Variance')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return mean_attention, std_attention


def plot_performance_comparison(batch_size_results: Dict, save_path: str = None):
    """Plot performance comparison across different batch sizes."""
    batch_sizes = []
    throughputs = []
    
    for batch_size, results in batch_size_results.items():
        if 'error' not in results:
            batch_sizes.append(batch_size)
            throughputs.append(results['throughput'])
    
    if not batch_sizes:
        print("No valid batch size results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput vs batch size
    ax1.plot(batch_sizes, throughputs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (samples/sec)')
    ax1.set_title('Throughput vs Batch Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Time per sample vs batch size
    times_per_sample = [batch_size_results[bs]['time_per_sample'] for bs in batch_sizes]
    ax2.plot(batch_sizes, times_per_sample, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Time per Sample (s)')
    ax2.set_title('Time per Sample vs Batch Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_evaluation_results(results: Dict, save_path: str = None):
    """Plot comprehensive evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ROC Curve
    if 'fpr' in results and 'tpr' in results:
        axes[0, 0].plot(results['fpr'], results['tpr'], 'b-', linewidth=2, 
                       label=f"AUC = {results['auc']:.4f}")
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Score distribution
    if 'genuine_scores' in results and 'impostor_scores' in results:
        genuine_scores = results['genuine_scores']
        impostor_scores = results['impostor_scores']
        
        axes[0, 1].hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', 
                       color='red', density=True)
        axes[0, 1].hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', 
                       color='blue', density=True)
        axes[0, 1].axvline(x=results.get('eer_threshold', 0.5), color='black', 
                          linestyle='--', label=f"EER Threshold ({results.get('eer_threshold', 0.5):.3f})")
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics summary
    metrics = {
        'Accuracy': results.get('accuracy', 0),
        'AUC': results.get('auc', 0),
        'EER': results.get('eer', 0),
        'TAR@FAR=0.1%': results.get('tar_at_far_001', 0),
        'TAR@FAR=1%': results.get('tar_at_far_01', 0)
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = axes[1, 0].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance Metrics Summary')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # DET Curve (if available)
    if 'far' in results and 'frr' in results:
        axes[1, 1].loglog(results['far'], results['frr'], 'b-', linewidth=2)
        axes[1, 1].set_xlabel('False Accept Rate')
        axes[1, 1].set_ylabel('False Reject Rate')
        axes[1, 1].set_title('DET Curve')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark EER point
        if 'eer' in results:
            eer_value = results['eer']
            axes[1, 1].plot(eer_value, eer_value, 'ro', markersize=8, 
                           label=f'EER = {eer_value:.4f}')
            axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation results plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_model_architecture_diagram(config: Dict, save_path: str = None):
    """Create a visual diagram of the model architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Model components
    components = [
        {'name': 'Input Images', 'x': 1, 'y': 7, 'width': 2, 'height': 1, 'color': 'lightblue'},
        {'name': f"{config['model']['backbone']}\nBackbone", 'x': 1, 'y': 5.5, 'width': 2, 'height': 1, 'color': 'lightgreen'},
        {'name': 'Global Branch', 'x': 0.5, 'y': 4, 'width': 1.5, 'height': 1, 'color': 'lightcoral'},
        {'name': 'Local Branch', 'x': 2.5, 'y': 4, 'width': 1.5, 'height': 1, 'color': 'lightcoral'},
        {'name': f"{config['model']['attention_type']}\nAttention", 'x': 1, 'y': 2.5, 'width': 2, 'height': 1, 'color': 'lightyellow'},
        {'name': 'Feature Fusion', 'x': 1, 'y': 1, 'width': 2, 'height': 1, 'color': 'lightgray'},
        {'name': 'Similarity Score', 'x': 1, 'y': -0.5, 'width': 2, 'height': 1, 'color': 'lightpink'}
    ]
    
    # Draw components
    for comp in components:
        rect = plt.Rectangle((comp['x'] - comp['width']/2, comp['y'] - comp['height']/2), 
                           comp['width'], comp['height'], 
                           facecolor=comp['color'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(comp['x'], comp['y'], comp['name'], ha='center', va='center', fontweight='bold')
    
    # Draw connections
    connections = [
        (2, 6.5, 2, 6),      # Input to backbone
        (2, 5, 1.25, 4.5),   # Backbone to global
        (2, 5, 2.75, 4.5),   # Backbone to local
        (1.25, 3.5, 2, 3),   # Global to attention
        (2.75, 3.5, 2, 3),   # Local to attention
        (2, 2, 2, 1.5),      # Attention to fusion
        (2, 0.5, 2, 0)       # Fusion to output
    ]
    
    for x1, y1, x2, y2 in connections:
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.05, head_length=0.05, 
                fc='black', ec='black')
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Local-Global Attention Network Architecture', fontsize=16, fontweight='bold')
    
    # Add model parameters
    param_text = f"""Model Configuration:
• Backbone: {config['model']['backbone']}
• Attention: {config['model']['attention_type']}
• Feature Dim: {config['model']['feature_dim']}
• Local Regions: {config['model']['local_regions']}
• Dropout: {config['model']['dropout_rate']}"""
    
    ax.text(4.5, 4, param_text, fontsize=10, va='center', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Example usage functions
def main():
    """Example usage of visualization functions"""
    
    # Example data
    n_samples = 1000
    similarities = np.random.beta(2, 2, n_samples)  # Beta distribution for similarities
    labels = np.random.binomial(1, 0.5, n_samples)
    twin_pairs = np.random.binomial(1, 0.3, n_samples)
    
    # Adjust similarities based on labels to make it more realistic
    similarities[labels == 1] = np.random.beta(5, 2, np.sum(labels == 1))  # Higher for same person
    similarities[labels == 0] = np.random.beta(2, 5, np.sum(labels == 0))  # Lower for different person
    
    # Plot similarity analysis
    plot_similarity_analysis(similarities, labels, twin_pairs, 'similarity_analysis.png')
    
    # Create sample attention maps
    attention_maps = [np.random.rand(14, 14) for _ in range(8)]
    create_attention_heatmap_grid(attention_maps, save_path='attention_grid.png')
    
    # Feature importance
    visualize_feature_importance(attention_maps, save_path='feature_importance.png')
    
    # Confusion matrix
    predictions = (similarities > 0.5).astype(int)
    plot_confusion_matrix(labels, predictions, save_path='confusion_matrix.png')
    
    # Example performance comparison
    batch_size_results = {
        8: {'throughput': 150, 'time_per_sample': 0.05},
        16: {'throughput': 300, 'time_per_sample': 0.04},
        32: {'throughput': 600, 'time_per_sample': 0.03},
        64: {'throughput': 1200, 'time_per_sample': 0.025},
        128: {'throughput': 2400, 'time_per_sample': 0.02}
    }
    plot_performance_comparison(batch_size_results, save_path='performance_comparison.png')
    
    # Example evaluation results
    evaluation_results = {
        'fpr': np.linspace(0, 1, 100),
        'tpr': np.linspace(0, 1, 100),
        'auc': 0.95,
        'genuine_scores': np.random.normal(0.8, 0.1, 500),
        'impostor_scores': np.random.normal(0.2, 0.1, 500),
        'eer_threshold': 0.5,
        'accuracy': 0.92,
        'eer': 0.05,
        'tar_at_far_001': 0.85,
        'tar_at_far_01': 0.90,
        'far': np.linspace(0.0001, 0.1, 100),
        'frr': np.linspace(0.1, 0.0001, 100)
    }
    plot_evaluation_results(evaluation_results, save_path='evaluation_results.png')


if __name__ == "__main__":
    main()
