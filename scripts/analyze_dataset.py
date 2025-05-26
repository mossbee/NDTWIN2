#!/usr/bin/env python3
"""
Script to analyze and preprocess the ND_TWIN dataset.

This script provides utilities for dataset analysis, validation, and preprocessing
including face detection, alignment, and quality assessment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.utils.dataset_utils import (
    validate_dataset, analyze_dataset, preprocess_dataset,
    check_face_alignment, generate_dataset_report
)


def main():
    parser = argparse.ArgumentParser(description='Analyze and preprocess ND_TWIN dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ND_TWIN dataset')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--action', type=str, required=True,
                        choices=['validate', 'analyze', 'preprocess', 'check_alignment', 'report'],
                        help='Action to perform')
    parser.add_argument('--save_preprocessed', type=str, default=None,
                        help='Directory to save preprocessed images (for preprocess action)')
    parser.add_argument('--face_size', type=int, default=224,
                        help='Size for face alignment (for preprocess action)')
    parser.add_argument('--quality_threshold', type=float, default=0.8,
                        help='Quality threshold for face images')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset root: {args.data_root}")
    print(f"Action: {args.action}")
    print(f"Output directory: {output_dir}")
    
    if args.action == 'validate':
        print("Validating dataset structure...")
        results = validate_dataset(args.data_root)
        
        # Save validation results
        results_file = output_dir / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        if results['valid']:
            print("✓ Dataset validation PASSED")
            print(f"Total subjects: {results['num_subjects']}")
            print(f"Total images: {results['num_images']}")
            print(f"Twin pairs: {results['num_twin_pairs']}")
        else:
            print("✗ Dataset validation FAILED")
            for error in results['errors']:
                print(f"  - {error}")
        
        print(f"Detailed results saved to {results_file}")
    
    elif args.action == 'analyze':
        print("Analyzing dataset...")
        analysis = analyze_dataset(args.data_root)
        
        # Save analysis results
        analysis_file = output_dir / 'dataset_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print summary
        print("\nDataset Analysis Summary:")
        print(f"Subjects: {analysis['num_subjects']}")
        print(f"Images: {analysis['num_images']}")
        print(f"Twin pairs: {analysis['num_twin_pairs']}")
        print(f"Average images per subject: {analysis['avg_images_per_subject']:.1f}")
        print(f"Image size distribution: {analysis['image_size_stats']}")
        print(f"File format distribution: {analysis['file_formats']}")
        
        print(f"Detailed analysis saved to {analysis_file}")
    
    elif args.action == 'check_alignment':
        print("Checking face alignment...")
        alignment_results = check_face_alignment(args.data_root, max_samples=1000)
        
        # Save alignment results
        alignment_file = output_dir / 'alignment_check.json'
        with open(alignment_file, 'w') as f:
            json.dump(alignment_results, f, indent=2)
        
        # Print summary
        print("\nFace Alignment Check:")
        print(f"Images checked: {alignment_results['total_checked']}")
        print(f"Faces detected: {alignment_results['faces_detected']}")
        print(f"Detection rate: {alignment_results['detection_rate']:.2%}")
        print(f"Average quality score: {alignment_results['avg_quality']:.3f}")
        print(f"Images above quality threshold: {alignment_results['high_quality_count']}")
        
        if alignment_results['problematic_images']:
            print(f"Problematic images: {len(alignment_results['problematic_images'])}")
            print("First few problematic images:")
            for img in alignment_results['problematic_images'][:5]:
                print(f"  - {img}")
        
        print(f"Detailed results saved to {alignment_file}")
    
    elif args.action == 'preprocess':
        if not args.save_preprocessed:
            print("Error: --save_preprocessed is required for preprocess action")
            return
        
        print("Preprocessing dataset...")
        save_dir = Path(args.save_preprocessed)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        preprocess_results = preprocess_dataset(
            data_root=args.data_root,
            output_dir=str(save_dir),
            face_size=args.face_size,
            quality_threshold=args.quality_threshold
        )
        
        # Save preprocessing results
        preprocess_file = output_dir / 'preprocessing_results.json'
        with open(preprocess_file, 'w') as f:
            json.dump(preprocess_results, f, indent=2)
        
        # Print summary
        print("\nPreprocessing Summary:")
        print(f"Images processed: {preprocess_results['total_processed']}")
        print(f"Images saved: {preprocess_results['images_saved']}")
        print(f"Images skipped: {preprocess_results['images_skipped']}")
        print(f"Success rate: {preprocess_results['success_rate']:.2%}")
        
        if preprocess_results['failed_images']:
            print(f"Failed images: {len(preprocess_results['failed_images'])}")
        
        print(f"Preprocessed images saved to {save_dir}")
        print(f"Detailed results saved to {preprocess_file}")
    
    elif args.action == 'report':
        print("Generating comprehensive dataset report...")
        
        # Run all analyses
        validation = validate_dataset(args.data_root)
        analysis = analyze_dataset(args.data_root)
        alignment = check_face_alignment(args.data_root, max_samples=500)
        
        # Generate report
        report = generate_dataset_report(validation, analysis, alignment)
        
        # Save report
        report_file = output_dir / 'dataset_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable report
        readable_report = output_dir / 'dataset_report.txt'
        with open(readable_report, 'w') as f:
            f.write("ND_TWIN Dataset Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Dataset Overview:\n")
            f.write(f"- Status: {'VALID' if validation['valid'] else 'INVALID'}\n")
            f.write(f"- Subjects: {analysis['num_subjects']}\n")
            f.write(f"- Images: {analysis['num_images']}\n")
            f.write(f"- Twin pairs: {analysis['num_twin_pairs']}\n\n")
            
            f.write("Image Quality:\n")
            f.write(f"- Face detection rate: {alignment['detection_rate']:.2%}\n")
            f.write(f"- Average quality score: {alignment['avg_quality']:.3f}\n")
            f.write(f"- High quality images: {alignment['high_quality_count']}\n\n")
            
            f.write("Recommendations:\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
        
        print(f"Comprehensive report saved to {report_file}")
        print(f"Human-readable report saved to {readable_report}")


if __name__ == '__main__':
    main()
