import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for siamese networks.
    Pulls positive pairs closer and pushes negative pairs apart.
    """
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embedding1: First embedding (B, D)
            embedding2: Second embedding (B, D)
            labels: Binary labels (B,) - 1 for same person, 0 for different
        """
        # Euclidean distance
        distances = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Contrastive loss
        positive_loss = labels * torch.pow(distances, 2)
        negative_loss = (1 - labels) * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss with hard negative mining.
    """
    
    def __init__(self, margin: float = 0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Feature embeddings (B, D)
            labels: Identity labels (B,)
        """
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Remove diagonal (self-comparisons)
        labels_equal = labels_equal.fill_diagonal_(False)
        
        # Hard positive mining - furthest positive
        pos_distances = distances * labels_equal.float()
        pos_distances[~labels_equal] = float('-inf')
        hardest_positive = pos_distances.max(dim=1)[0]
        
        # Hard negative mining - closest negative
        neg_distances = distances * labels_not_equal.float()
        neg_distances[~labels_not_equal] = float('inf')
        hardest_negative = neg_distances.min(dim=1)[0]
        
        # Triplet loss
        triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin)
        return triplet_loss.mean()


class AttentionRegularizationLoss(nn.Module):
    """
    Regularization loss for attention maps to encourage sparsity and diversity.
    """
    
    def __init__(self, sparsity_weight: float = 0.01, diversity_weight: float = 0.01):
        super(AttentionRegularizationLoss, self).__init__()
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        
    def forward(self, attention_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_maps: Attention maps (B, 1, H, W)
        """
        B, _, H, W = attention_maps.shape
        
        # Sparsity loss - encourage focused attention
        # L1 norm to encourage sparsity
        sparsity_loss = torch.mean(torch.sum(attention_maps.view(B, -1), dim=1))
        
        # Diversity loss - encourage different attention patterns
        # Compute pairwise similarities between attention maps in the batch
        flattened_maps = attention_maps.view(B, -1)
        similarities = F.cosine_similarity(
            flattened_maps.unsqueeze(1), 
            flattened_maps.unsqueeze(0), 
            dim=2
        )
        
        # Remove diagonal and compute mean similarity
        mask = ~torch.eye(B, dtype=torch.bool, device=attention_maps.device)
        diversity_loss = torch.mean(similarities[mask])
        
        total_loss = (self.sparsity_weight * sparsity_loss + 
                     self.diversity_weight * diversity_loss)
        
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for the Local-Global Attention Network.
    Combines contrastive/triplet loss with attention regularization.
    """
    
    def __init__(
        self,
        loss_type: str = 'contrastive',
        margin: float = 1.0,
        weight_global: float = 0.6,
        weight_local: float = 0.4,
        attention_reg_weight: float = 0.01,
        sparsity_weight: float = 0.01,
        diversity_weight: float = 0.01
    ):
        super(CombinedLoss, self).__init__()
        
        self.loss_type = loss_type
        self.weight_global = weight_global
        self.weight_local = weight_local
        self.attention_reg_weight = attention_reg_weight
        
        # Main loss function
        if loss_type == 'contrastive':
            self.main_loss = ContrastiveLoss(margin=margin)
        elif loss_type == 'triplet':
            self.main_loss = TripletLoss(margin=margin)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Attention regularization
        self.attention_reg = AttentionRegularizationLoss(
            sparsity_weight=sparsity_weight,
            diversity_weight=diversity_weight
        )
        
    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs dictionary
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Extract embeddings and features
        embedding1 = outputs['embedding1']
        embedding2 = outputs['embedding2']
        global_features1 = outputs['global_features1']
        global_features2 = outputs['global_features2']
        local_features1 = outputs['local_features1']
        local_features2 = outputs['local_features2']
        attention_map1 = outputs['attention_map1']
        attention_map2 = outputs['attention_map2']
        
        # Main loss on full embeddings
        if self.loss_type == 'contrastive':
            main_loss = self.main_loss(embedding1, embedding2, labels)
        else:  # triplet
            # For triplet loss, we need to reshape
            all_embeddings = torch.cat([embedding1, embedding2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            main_loss = self.main_loss(all_embeddings, all_labels)
        
        losses['main'] = main_loss
        
        # Separate losses for global and local branches
        if self.loss_type == 'contrastive':
            global_loss = self.main_loss(global_features1, global_features2, labels)
            local_loss = self.main_loss(local_features1, local_features2, labels)
        else:
            all_global = torch.cat([global_features1, global_features2], dim=0)
            all_local = torch.cat([local_features1, local_features2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            global_loss = self.main_loss(all_global, all_labels)
            local_loss = self.main_loss(all_local, all_labels)
        
        losses['global'] = global_loss
        losses['local'] = local_loss
        
        # Attention regularization
        all_attention = torch.cat([attention_map1, attention_map2], dim=0)
        attention_reg_loss = self.attention_reg(all_attention)
        losses['attention_reg'] = attention_reg_loss
        
        # Combined weighted loss
        total_loss = (
            self.weight_global * global_loss +
            self.weight_local * local_loss +
            self.attention_reg_weight * attention_reg_loss
        )
        
        losses['total'] = total_loss
        
        return losses


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Useful when positive and negative pairs are heavily imbalanced.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted similarities or logits (B,)
            targets: Binary targets (B,)
        """
        # Convert to probabilities if needed
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            probs = torch.sigmoid(predictions)
        else:
            probs = predictions
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for learning better embeddings.
    """
    
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Feature embeddings (B, D)
            labels: Class labels (B,)
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


def create_loss_function(config: dict) -> CombinedLoss:
    """Create loss function based on configuration"""
    
    loss_config = config['loss']
    
    return CombinedLoss(
        loss_type=loss_config['type'],
        margin=config['training']['margin'],
        weight_global=loss_config['weight_global'],
        weight_local=loss_config['weight_local'],
        attention_reg_weight=loss_config['attention_reg_weight']
    )


if __name__ == "__main__":
    # Test loss functions
    B, D = 4, 512
    
    # Test contrastive loss
    embedding1 = torch.randn(B, D)
    embedding2 = torch.randn(B, D)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    
    contrastive_loss = ContrastiveLoss()
    loss_value = contrastive_loss(embedding1, embedding2, labels)
    print(f"Contrastive loss: {loss_value.item()}")
    
    # Test attention regularization
    attention_maps = torch.rand(B, 1, 14, 14)
    attention_reg = AttentionRegularizationLoss()
    reg_loss = attention_reg(attention_maps)
    print(f"Attention regularization loss: {reg_loss.item()}")
    
    # Test combined loss
    outputs = {
        'embedding1': embedding1,
        'embedding2': embedding2,
        'global_features1': torch.randn(B, D//2),
        'global_features2': torch.randn(B, D//2),
        'local_features1': torch.randn(B, D//2),
        'local_features2': torch.randn(B, D//2),
        'attention_map1': attention_maps,
        'attention_map2': attention_maps,
    }
    
    combined_loss = CombinedLoss()
    losses = combined_loss(outputs, labels)
    print(f"Combined losses: {losses}")