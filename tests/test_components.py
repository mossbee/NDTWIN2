#!/usr/bin/env python3
"""
Unit tests for the Local-Global Attention Network components.

This module contains comprehensive tests for the dataset, model, and utility functions.
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import yaml

from src.data.dataset import NDTwinDataset, get_transform
from src.models.attention_network import (
    SiameseLocalGlobalNet, LocalGlobalAttentionNetwork,
    CBAMAttention, SpatialAttention, TransformerAttention
)
from src.models.loss_functions import (
    ContrastiveLoss, TripletLoss, AttentionRegularizationLoss, CombinedLoss
)
from src.utils.dataset_utils import validate_dataset_structure


class TestAttentionMechanisms(unittest.TestCase):
    """Test attention mechanism implementations."""
    
    def setUp(self):
        self.batch_size = 2
        self.channels = 512
        self.height = 7
        self.width = 7
        self.feature_maps = torch.randn(self.batch_size, self.channels, self.height, self.width)
    
    def test_cbam_attention(self):
        """Test CBAM attention mechanism."""
        cbam = CBAMAttention(self.channels)
        output = cbam(self.feature_maps)
        
        # Check output shape
        self.assertEqual(output.shape, self.feature_maps.shape)
        
        # Check that attention weights are between 0 and 1
        attention_weights = cbam.get_attention_weights(self.feature_maps)
        self.assertTrue(torch.all(attention_weights >= 0))
        self.assertTrue(torch.all(attention_weights <= 1))
    
    def test_spatial_attention(self):
        """Test spatial attention mechanism."""
        spatial_att = SpatialAttention(self.channels)
        output = spatial_att(self.feature_maps)
        
        # Check output shape
        self.assertEqual(output.shape, self.feature_maps.shape)
        
        # Check attention map shape
        attention_map = spatial_att.get_attention_map(self.feature_maps)
        expected_shape = (self.batch_size, 1, self.height, self.width)
        self.assertEqual(attention_map.shape, expected_shape)
    
    def test_transformer_attention(self):
        """Test transformer attention mechanism."""
        transformer_att = TransformerAttention(self.channels, num_heads=8)
        
        # Flatten feature maps for transformer
        flat_features = self.feature_maps.view(self.batch_size, self.channels, -1).transpose(1, 2)
        output = transformer_att(flat_features)
        
        # Check output shape
        expected_shape = (self.batch_size, self.height * self.width, self.channels)
        self.assertEqual(output.shape, expected_shape)


class TestLocalGlobalNetwork(unittest.TestCase):
    """Test the main network architecture."""
    
    def setUp(self):
        self.batch_size = 2
        self.image_size = 224
        self.config = {
            'backbone': 'resnet50',
            'num_classes': 1,
            'attention_type': 'cbam',
            'feature_dim': 512,
            'local_regions': 4,
            'dropout_rate': 0.5
        }
    
    def test_network_forward(self):
        """Test forward pass of the network."""
        network = LocalGlobalAttentionNetwork(**self.config)
        
        # Create dummy input
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        # Forward pass
        global_features, local_features, attention_maps = network(x)
        
        # Check output shapes
        self.assertEqual(global_features.shape, (self.batch_size, self.config['feature_dim']))
        self.assertEqual(local_features.shape, (self.batch_size, self.config['feature_dim']))
        self.assertIsNotNone(attention_maps)
    
    def test_siamese_network(self):
        """Test siamese wrapper."""
        siamese_net = SiameseLocalGlobalNet(**self.config)
        
        # Create dummy inputs
        img1 = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        img2 = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        # Forward pass
        similarity = siamese_net(img1, img2)
        
        # Check output shape
        self.assertEqual(similarity.shape, (self.batch_size, 1))
        
        # Check similarity range (should be between 0 and 1 after sigmoid)
        self.assertTrue(torch.all(similarity >= 0))
        self.assertTrue(torch.all(similarity <= 1))
    
    def test_different_backbones(self):
        """Test different backbone architectures."""
        backbones = ['resnet18', 'resnet50', 'efficientnet_b0']
        
        for backbone in backbones:
            config = self.config.copy()
            config['backbone'] = backbone
            
            try:
                network = LocalGlobalAttentionNetwork(**config)
                x = torch.randn(1, 3, self.image_size, self.image_size)
                global_features, local_features, attention_maps = network(x)
                
                # Check that forward pass works
                self.assertEqual(global_features.shape[0], 1)
                self.assertEqual(local_features.shape[0], 1)
                
            except Exception as e:
                self.fail(f"Failed to create network with backbone {backbone}: {e}")


class TestLossFunctions(unittest.TestCase):
    """Test loss function implementations."""
    
    def setUp(self):
        self.batch_size = 4
        self.feature_dim = 512
        
        # Create dummy features and labels
        self.features1 = torch.randn(self.batch_size, self.feature_dim)
        self.features2 = torch.randn(self.batch_size, self.feature_dim)
        self.labels = torch.randint(0, 2, (self.batch_size,)).float()
        self.similarities = torch.sigmoid(torch.randn(self.batch_size))
    
    def test_contrastive_loss(self):
        """Test contrastive loss."""
        loss_fn = ContrastiveLoss(margin=1.0)
        loss = loss_fn(self.features1, self.features2, self.labels)
        
        # Check that loss is computed
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_triplet_loss(self):
        """Test triplet loss."""
        loss_fn = TripletLoss(margin=0.5)
        
        # Create anchor, positive, negative
        anchor = self.features1[:2]
        positive = self.features2[:2]
        negative = self.features1[2:]
        
        loss = loss_fn(anchor, positive, negative)
        
        # Check that loss is computed
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_attention_regularization_loss(self):
        """Test attention regularization loss."""
        loss_fn = AttentionRegularizationLoss()
        
        # Create dummy attention maps
        attention_maps = {
            'global_attention': torch.rand(self.batch_size, 1, 7, 7),
            'local_attention': torch.rand(self.batch_size, 4, 7, 7)
        }
        
        loss = loss_fn(attention_maps)
        
        # Check that loss is computed
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_combined_loss(self):
        """Test combined loss function."""
        loss_fn = CombinedLoss()
        
        # Create dummy outputs
        global_features = self.features1
        local_features = self.features2
        attention_maps = {
            'global_attention': torch.rand(self.batch_size, 1, 7, 7),
            'local_attention': torch.rand(self.batch_size, 4, 7, 7)
        }
        similarities = self.similarities
        labels = self.labels
        
        loss_dict = loss_fn(global_features, local_features, attention_maps, similarities, labels)
        
        # Check that all loss components are computed
        expected_keys = ['global', 'local', 'attention_reg', 'total']
        for key in expected_keys:
            self.assertIn(key, loss_dict)
            self.assertIsInstance(loss_dict[key], torch.Tensor)


class TestDataset(unittest.TestCase):
    """Test dataset implementation."""
    
    def setUp(self):
        # Create temporary dataset structure
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = Path(self.temp_dir)
        
        # Create sample dataset structure
        self.create_sample_dataset()
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing."""
        # Create splits
        for split in ['train', 'val', 'test']:
            split_dir = self.data_root / split
            split_dir.mkdir(parents=True)
            
            # Create some subjects
            for subject_id in range(5):
                subject_dir = split_dir / f'subject_{subject_id:03d}'
                subject_dir.mkdir()
                
                # Create some images for each subject
                for img_id in range(3):
                    img_path = subject_dir / f'image_{img_id}.jpg'
                    
                    # Create a dummy image
                    img = Image.new('RGB', (224, 224), color='white')
                    img.save(img_path)
        
        # Create twins.txt file
        twins_file = self.data_root / 'twins.txt'
        with open(twins_file, 'w') as f:
            f.write('subject_001,subject_002\n')
            f.write('subject_003,subject_004\n')
    
    def test_dataset_validation(self):
        """Test dataset structure validation."""
        result = validate_dataset_structure(str(self.data_root))
        self.assertTrue(result['valid'])
    
    def test_dataset_creation(self):
        """Test dataset creation and loading."""
        try:
            dataset = NDTwinDataset(
                data_root=str(self.data_root),
                split='train',
                image_size=224,
                twin_negative_ratio=0.7
            )
            
            # Check dataset size
            self.assertGreater(len(dataset), 0)
            
            # Check sample format
            sample = dataset[0]
            expected_keys = ['image1', 'image2', 'label']
            for key in expected_keys:
                self.assertIn(key, sample)
            
            # Check tensor shapes
            self.assertEqual(sample['image1'].shape, (3, 224, 224))
            self.assertEqual(sample['image2'].shape, (3, 224, 224))
            self.assertIn(sample['label'].item(), [0, 1])
            
        except Exception as e:
            self.fail(f"Failed to create dataset: {e}")
    
    def test_transforms(self):
        """Test image transformations."""
        transform = get_transform(224, is_training=True)
        
        # Create dummy image
        img = Image.new('RGB', (256, 256), color='red')
        transformed = transform(img)
        
        # Check output shape and type
        self.assertEqual(transformed.shape, (3, 224, 224))
        self.assertIsInstance(transformed, torch.Tensor)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        self.config = {
            'model': {
                'backbone': 'resnet18',  # Use smaller model for testing
                'num_classes': 1,
                'attention_type': 'cbam',
                'feature_dim': 256,
                'local_regions': 2,
                'dropout_rate': 0.1
            },
            'data': {
                'image_size': 224,
                'twin_negative_ratio': 0.7
            }
        }
    
    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        # Create model
        model = SiameseLocalGlobalNet(**self.config['model'])
        
        # Create dummy batch
        batch_size = 2
        img1 = torch.randn(batch_size, 3, 224, 224)
        img2 = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        # Create loss function and optimizer
        loss_fn = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass through base network
        global_feat1, local_feat1, att_maps1 = model.base_network(img1)
        global_feat2, local_feat2, att_maps2 = model.base_network(img2)
        
        # Compute similarity
        similarity = model.similarity_head(
            torch.cat([global_feat1, local_feat1], dim=1),
            torch.cat([global_feat2, local_feat2], dim=1)
        )
        
        # Compute loss
        loss_dict = loss_fn(global_feat1, local_feat1, att_maps1, similarity.squeeze(), labels)
        
        # Backward pass
        loss_dict['total'].backward()
        optimizer.step()
        
        # Check that loss is computed and finite
        self.assertTrue(torch.isfinite(loss_dict['total']))
        self.assertGreater(loss_dict['total'].item(), 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAttentionMechanisms,
        TestLocalGlobalNetwork,
        TestLossFunctions,
        TestDataset,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
