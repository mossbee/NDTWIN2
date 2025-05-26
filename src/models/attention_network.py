import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Tuple, Dict, Optional
import math


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on discriminative regions"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // reduction, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Global average pooling to get channel-wise attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        channel_att = self.conv2(F.relu(self.conv1(avg_pool)))
        channel_att = self.sigmoid(channel_att)
        
        # Apply channel attention
        x_att = x * channel_att
        
        # Spatial attention
        spatial_att = torch.mean(x_att, dim=1, keepdim=True)
        spatial_att = self.sigmoid(spatial_att)
        
        # Apply spatial attention
        output = x_att * spatial_att
        
        return output, spatial_att


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(CBAM, self).__init__()
        self.channel_attention = self._channel_attention(in_channels, reduction)
        self.spatial_attention = self._spatial_attention()
        
    def _channel_attention(self, in_channels: int, reduction: int):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def _spatial_attention(self):
        return nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        
        return x, spatial_att


class TransformerAttention(nn.Module):
    """Transformer-based attention for spatial features"""
    
    def __init__(self, in_channels: int, num_heads: int = 8, dropout: float = 0.1):
        super(TransformerAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        # Project to query, key, value
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Spatial attention map generation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        # Multi-head self-attention
        qkv = self.qkv(x).view(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, C//heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention weights
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        out = self.proj(out)
        
        # Generate spatial attention map
        spatial_att = self.attention_conv(out)
        
        # Apply spatial attention
        out = out * spatial_att
        
        return out, spatial_att


class LocalGlobalAttentionNetwork(nn.Module):
    """
    Local-Global Attention Network for identical twin verification.
    Combines global face representations with attention-driven local features.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        embedding_dim: int = 512,
        attention_type: str = 'cbam',
        dropout: float = 0.3,
        num_classes: Optional[int] = None
    ):
        super(LocalGlobalAttentionNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.attention_type = attention_type
        
        # Initialize backbone
        self.backbone = self._get_backbone(backbone, pretrained)
        backbone_dim = self._get_backbone_dim(backbone)
        
        # Global branch - extracts overall face features
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_dim, embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Local branch with attention
        if attention_type == 'cbam':
            self.attention = CBAM(backbone_dim)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention(backbone_dim)
        elif attention_type == 'transformer':
            self.attention = TransformerAttention(backbone_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.local_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_dim, embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Final embedding layer
        self.embedding = nn.Linear(embedding_dim, embedding_dim)
        
        # Classification head (optional, for auxiliary loss)
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
            
    def _get_backbone(self, backbone: str, pretrained: bool):
        """Initialize backbone network"""
        if backbone.startswith('resnet'):
            if backbone == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
            elif backbone == 'resnet101':
                model = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown ResNet variant: {backbone}")
            
            # Remove final layers
            return nn.Sequential(*list(model.children())[:-2])
            
        elif backbone.startswith('efficientnet'):
            model = timm.create_model(backbone, pretrained=pretrained, features_only=True)
            return model
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def _get_backbone_dim(self, backbone: str) -> int:
        """Get backbone output dimensions"""
        if backbone == 'resnet50':
            return 2048
        elif backbone == 'resnet101':
            return 2048
        elif backbone.startswith('efficientnet'):
            # This is approximate - you might need to adjust based on specific model
            if 'b0' in backbone:
                return 1280
            elif 'b1' in backbone:
                return 1280
            elif 'b2' in backbone:
                return 1408
            else:
                return 1536
        else:
            raise ValueError(f"Unknown backbone dimension for: {backbone}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary containing:
            - embedding: Final feature embedding
            - global_features: Global branch features
            - local_features: Local branch features  
            - attention_map: Spatial attention map
            - logits: Classification logits (if classifier exists)
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Global branch - captures overall face structure
        global_features = self.global_branch(features)
        
        # Local branch with attention - focuses on discriminative regions
        attended_features, attention_map = self.attention(features)
        local_features = self.local_branch(attended_features)
        
        # Fuse global and local features
        combined = torch.cat([global_features, local_features], dim=1)
        fused = self.fusion(combined)
        
        # Final embedding
        embedding = self.embedding(fused)
        
        # L2 normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        
        output = {
            'embedding': embedding,
            'global_features': global_features,
            'local_features': local_features,
            'attention_map': attention_map
        }
        
        # Add classification logits if classifier exists
        if self.classifier is not None:
            output['logits'] = self.classifier(fused)
            
        return output
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


class SiameseNetwork(nn.Module):
    """Siamese network wrapper for twin verification"""
    
    def __init__(self, base_network: LocalGlobalAttentionNetwork):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for siamese network
        
        Args:
            x1: First image tensor
            x2: Second image tensor
            
        Returns:
            Dictionary containing outputs for both images
        """
        output1 = self.base_network(x1)
        output2 = self.base_network(x2)
        
        return {
            'embedding1': output1['embedding'],
            'embedding2': output2['embedding'],
            'global_features1': output1['global_features'],
            'global_features2': output2['global_features'],
            'local_features1': output1['local_features'],
            'local_features2': output2['local_features'],
            'attention_map1': output1['attention_map'],
            'attention_map2': output2['attention_map'],
        }
    
    def get_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two images"""
        outputs = self.forward(x1, x2)
        similarity = F.cosine_similarity(
            outputs['embedding1'], 
            outputs['embedding2'], 
            dim=1
        )
        return similarity


def create_model(config: dict) -> SiameseNetwork:
    """Create the Local-Global Attention Network model"""
    
    base_network = LocalGlobalAttentionNetwork(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        embedding_dim=config['model']['embedding_dim'],
        attention_type=config['model']['attention_type'],
        dropout=config['model']['dropout']
    )
    
    model = SiameseNetwork(base_network)
    return model


if __name__ == "__main__":
    # Test model creation
    config = {
        'model': {
            'backbone': 'resnet50',
            'pretrained': True,
            'embedding_dim': 512,
            'attention_type': 'cbam',
            'dropout': 0.3
        }
    }
    
    model = create_model(config)
    
    # Test forward pass
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    
    outputs = model(x1, x2)
    print("Model output keys:", outputs.keys())
    print("Embedding1 shape:", outputs['embedding1'].shape)
    print("Attention map1 shape:", outputs['attention_map1'].shape)
    
    similarity = model.get_similarity(x1, x2)
    print("Similarity shape:", similarity.shape)