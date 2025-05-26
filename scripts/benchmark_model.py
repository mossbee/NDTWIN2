#!/usr/bin/env python3
"""
Script to benchmark model performance and generate detailed performance reports.

This script evaluates model performance across different metrics and generates
comprehensive benchmarking reports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import NDTwinDataset
from src.models.attention_network import SiameseLocalGlobalNet
from src.inference import ModelEvaluator
from src.utils.visualization import plot_performance_comparison


def benchmark_inference_speed(model, dataset, device, num_samples=1000):
    """Benchmark inference speed."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    times = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            
            start_time = time.time()
            _ = model(img1, img2)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return {
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'fps': 1.0 / (sum(times) / len(times))
    }


def benchmark_memory_usage(model, dataset, device):
    """Benchmark memory usage."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Get one batch
    batch = next(iter(dataloader))
    img1 = batch['image1'].to(device)
    img2 = batch['image2'].to(device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(img1, img2)
        
        memory_stats = {
            'allocated': torch.cuda.memory_allocated(device) / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved(device) / 1024**2,     # MB
            'peak': torch.cuda.max_memory_allocated(device) / 1024**2   # MB
        }
    else:
        memory_stats = {'allocated': 0, 'cached': 0, 'peak': 0}
    
    return memory_stats


def main():
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ND_TWIN dataset')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Output directory for benchmark results')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for speed benchmark')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 8, 16, 32],
                        help='Batch sizes to benchmark')
    
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
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
    
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    # Create test dataset
    test_dataset = NDTwinDataset(
        data_root=args.data_root,
        split='test',
        **config['data']
    )
    
    benchmark_results = {
        'model_info': {
            'num_parameters': num_params,
            'model_size_mb': model_size,
            'backbone': config['model']['backbone'],
            'attention_type': config['model']['attention_type']
        },
        'device': str(device),
        'dataset_size': len(test_dataset)
    }
    
    # Benchmark inference speed
    print("Benchmarking inference speed...")
    speed_results = benchmark_inference_speed(model, test_dataset, device, args.num_samples)
    benchmark_results['speed'] = speed_results
    
    print(f"Average inference time: {speed_results['mean_time']:.4f}s")
    print(f"Inference FPS: {speed_results['fps']:.2f}")
    
    # Benchmark memory usage
    print("Benchmarking memory usage...")
    memory_results = benchmark_memory_usage(model, test_dataset, device)
    benchmark_results['memory'] = memory_results
    
    if device.type == 'cuda':
        print(f"GPU memory allocated: {memory_results['allocated']:.2f} MB")
        print(f"GPU memory peak: {memory_results['peak']:.2f} MB")
    
    # Benchmark different batch sizes
    print("Benchmarking different batch sizes...")
    batch_size_results = {}
    
    for batch_size in args.batch_sizes:
        print(f"Testing batch size {batch_size}...")
        try:
            dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            batch = next(iter(dataloader))
            
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            
            # Warm-up
            with torch.no_grad():
                _ = model(img1, img2)
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(img1, img2)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            throughput = batch_size / avg_time
            
            batch_size_results[batch_size] = {
                'avg_time': avg_time,
                'throughput': throughput,
                'time_per_sample': avg_time / batch_size
            }
            
            print(f"  Batch size {batch_size}: {throughput:.2f} samples/sec")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch size {batch_size}: Out of memory")
                batch_size_results[batch_size] = {'error': 'out_of_memory'}
            else:
                raise e
    
    benchmark_results['batch_sizes'] = batch_size_results
    
    # Evaluate model accuracy
    print("Evaluating model accuracy...")
    evaluator = ModelEvaluator(model, device=device)
    accuracy_results = evaluator.evaluate(test_dataset, batch_size=32)
    
    benchmark_results['accuracy'] = {
        'accuracy': accuracy_results['accuracy'],
        'auc': accuracy_results['auc'],
        'eer': accuracy_results['eer'],
        'tar_at_far_001': accuracy_results['tar_at_far_001'],
        'tar_at_far_01': accuracy_results['tar_at_far_01']
    }
    
    print(f"Model accuracy: {accuracy_results['accuracy']:.4f}")
    print(f"Model AUC: {accuracy_results['auc']:.4f}")
    
    # Save benchmark results
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Generate summary report
    summary_file = output_dir / 'benchmark_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Model Performance Benchmark Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"- Parameters: {num_params:,}\n")
        f.write(f"- Model size: {model_size:.2f} MB\n")
        f.write(f"- Backbone: {config['model']['backbone']}\n")
        f.write(f"- Attention: {config['model']['attention_type']}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"- Accuracy: {accuracy_results['accuracy']:.4f}\n")
        f.write(f"- AUC: {accuracy_results['auc']:.4f}\n")
        f.write(f"- EER: {accuracy_results['eer']:.4f}\n\n")
        
        f.write("Speed Performance:\n")
        f.write(f"- Average inference time: {speed_results['mean_time']:.4f}s\n")
        f.write(f"- Inference FPS: {speed_results['fps']:.2f}\n\n")
        
        if device.type == 'cuda':
            f.write("Memory Usage:\n")
            f.write(f"- Allocated: {memory_results['allocated']:.2f} MB\n")
            f.write(f"- Peak: {memory_results['peak']:.2f} MB\n\n")
        
        f.write("Batch Size Performance:\n")
        for batch_size, results in batch_size_results.items():
            if 'error' not in results:
                f.write(f"- Batch {batch_size}: {results['throughput']:.2f} samples/sec\n")
            else:
                f.write(f"- Batch {batch_size}: {results['error']}\n")
    
    print(f"\nBenchmark results saved to {results_file}")
    print(f"Summary report saved to {summary_file}")
    
    # Plot performance comparison if multiple results exist
    if len(batch_size_results) > 1:
        plot_path = output_dir / 'performance_comparison.png'
        try:
            plot_performance_comparison(batch_size_results, save_path=str(plot_path))
            print(f"Performance plots saved to {plot_path}")
        except Exception as e:
            print(f"Failed to generate plots: {e}")


if __name__ == '__main__':
    main()
