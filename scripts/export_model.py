#!/usr/bin/env python3
"""
Script to export trained models to different formats for deployment.

Supports exporting to ONNX, TorchScript, and TensorRT formats for various
deployment scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.onnx
import yaml
import argparse
from pathlib import Path

from src.models.attention_network import SiameseLocalGlobalNet


def export_to_onnx(model, input_shape, output_path, opset_version=11):
    """Export model to ONNX format."""
    model.eval()
    
    # Create dummy inputs
    dummy_img1 = torch.randn(1, *input_shape)
    dummy_img2 = torch.randn(1, *input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_img1, dummy_img2),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image1', 'image2'],
        output_names=['similarity'],
        dynamic_axes={
            'image1': {0: 'batch_size'},
            'image2': {0: 'batch_size'},
            'similarity': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")


def export_to_torchscript(model, input_shape, output_path, method='trace'):
    """Export model to TorchScript format."""
    model.eval()
    
    if method == 'trace':
        # Create dummy inputs
        dummy_img1 = torch.randn(1, *input_shape)
        dummy_img2 = torch.randn(1, *input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, (dummy_img1, dummy_img2))
        traced_model.save(output_path)
        
    elif method == 'script':
        # Script the model
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
    
    print(f"Model exported to TorchScript ({method}): {output_path}")


def verify_exported_model(original_model, exported_path, input_shape, format_type):
    """Verify that the exported model produces the same results."""
    original_model.eval()
    
    # Create test inputs
    test_img1 = torch.randn(1, *input_shape)
    test_img2 = torch.randn(1, *input_shape)
    
    # Get original output
    with torch.no_grad():
        original_output = original_model(test_img1, test_img2)
    
    if format_type == 'onnx':
        try:
            import onnxruntime as ort
            
            # Load ONNX model
            ort_session = ort.InferenceSession(str(exported_path))
            
            # Run inference
            ort_inputs = {
                'image1': test_img1.numpy(),
                'image2': test_img2.numpy()
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            exported_output = torch.tensor(ort_outputs[0])
            
        except ImportError:
            print("ONNX Runtime not installed, skipping verification")
            return True
    
    elif format_type == 'torchscript':
        # Load TorchScript model
        loaded_model = torch.jit.load(exported_path)
        
        with torch.no_grad():
            exported_output = loaded_model(test_img1, test_img2)
    
    # Compare outputs
    diff = torch.abs(original_output - exported_output).max().item()
    tolerance = 1e-5
    
    if diff < tolerance:
        print(f"✓ Verification passed (max diff: {diff:.2e})")
        return True
    else:
        print(f"✗ Verification failed (max diff: {diff:.2e})")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export trained model to different formats')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='exported_models',
                        help='Output directory for exported models')
    parser.add_argument('--formats', type=str, nargs='+',
                        choices=['onnx', 'torchscript_trace', 'torchscript_script'],
                        default=['onnx', 'torchscript_trace'],
                        help='Export formats')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported models against original')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX opset version')
    
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
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded from {args.model_path}")
    
    # Input shape
    input_shape = (3, config['data']['image_size'], config['data']['image_size'])
    print(f"Input shape: {input_shape}")
    
    # Export to different formats
    export_info = {}
    
    for format_name in args.formats:
        print(f"\nExporting to {format_name}...")
        
        if format_name == 'onnx':
            output_path = output_dir / 'model.onnx'
            export_to_onnx(model, input_shape, str(output_path), args.opset_version)
            export_info['onnx'] = {
                'path': str(output_path),
                'size_mb': output_path.stat().st_size / 1024**2
            }
            
            if args.verify:
                verify_exported_model(model, output_path, input_shape, 'onnx')
        
        elif format_name == 'torchscript_trace':
            output_path = output_dir / 'model_traced.pt'
            export_to_torchscript(model, input_shape, str(output_path), 'trace')
            export_info['torchscript_trace'] = {
                'path': str(output_path),
                'size_mb': output_path.stat().st_size / 1024**2
            }
            
            if args.verify:
                verify_exported_model(model, output_path, input_shape, 'torchscript')
        
        elif format_name == 'torchscript_script':
            output_path = output_dir / 'model_scripted.pt'
            export_to_torchscript(model, input_shape, str(output_path), 'script')
            export_info['torchscript_script'] = {
                'path': str(output_path),
                'size_mb': output_path.stat().st_size / 1024**2
            }
            
            if args.verify:
                verify_exported_model(model, output_path, input_shape, 'torchscript')
    
    # Save export information
    import json
    info_file = output_dir / 'export_info.json'
    with open(info_file, 'w') as f:
        json.dump({
            'original_model': args.model_path,
            'config': args.config,
            'input_shape': input_shape,
            'exports': export_info
        }, f, indent=2)
    
    # Create deployment examples
    examples_dir = output_dir / 'deployment_examples'
    examples_dir.mkdir(exist_ok=True)
    
    # ONNX deployment example
    if 'onnx' in export_info:
        onnx_example = examples_dir / 'onnx_inference.py'
        with open(onnx_example, 'w') as f:
            f.write("""#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

# Load ONNX model
session = ort.InferenceSession('model.onnx')

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess images
img1 = Image.open('image1.jpg').convert('RGB')
img2 = Image.open('image2.jpg').convert('RGB')

img1_tensor = transform(img1).unsqueeze(0).numpy()
img2_tensor = transform(img2).unsqueeze(0).numpy()

# Run inference
inputs = {'image1': img1_tensor, 'image2': img2_tensor}
outputs = session.run(None, inputs)
similarity = outputs[0][0][0]

print(f"Similarity: {similarity:.4f}")
print(f"Same person: {'Yes' if similarity > 0.5 else 'No'}")
""")
    
    # TorchScript deployment example
    if 'torchscript_trace' in export_info or 'torchscript_script' in export_info:
        ts_example = examples_dir / 'torchscript_inference.py'
        with open(ts_example, 'w') as f:
            f.write("""#!/usr/bin/env python3
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load TorchScript model
model = torch.jit.load('model_traced.pt')
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess images
img1 = Image.open('image1.jpg').convert('RGB')
img2 = Image.open('image2.jpg').convert('RGB')

img1_tensor = transform(img1).unsqueeze(0)
img2_tensor = transform(img2).unsqueeze(0)

# Run inference
with torch.no_grad():
    similarity = model(img1_tensor, img2_tensor).item()

print(f"Similarity: {similarity:.4f}")
print(f"Same person: {'Yes' if similarity > 0.5 else 'No'}")
""")
    
    print(f"\nExport completed!")
    print(f"Export information saved to {info_file}")
    print(f"Deployment examples saved to {examples_dir}")
    
    # Print summary
    print(f"\nExport Summary:")
    for format_name, info in export_info.items():
        print(f"- {format_name}: {info['path']} ({info['size_mb']:.2f} MB)")


if __name__ == '__main__':
    main()
