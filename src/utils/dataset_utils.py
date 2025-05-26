import os
import json
import random
from typing import List, Tuple
import argparse
from collections import defaultdict


def validate_dataset_structure(dataset_path: str) -> bool:
    """Validate the dataset has the expected structure"""
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return False
    
    folders = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(folders) == 0:
        print("No folders found in dataset")
        return False
    
    print(f"Found {len(folders)} folders in dataset")
    
    # Check image counts
    image_counts = []
    for folder in folders[:10]:  # Check first 10 folders
        folder_path = os.path.join(dataset_path, folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_counts.append(len(images))
    
    print(f"Sample image counts per folder: {image_counts}")
    return True


def validate_pairs_file(pairs_file: str, dataset_path: str) -> bool:
    """Validate the pairs.json file"""
    if not os.path.exists(pairs_file):
        print(f"Pairs file does not exist: {pairs_file}")
        return False
    
    with open(pairs_file, 'r') as f:
        pairs = json.load(f)
    
    if not isinstance(pairs, list):
        print("Pairs file should contain a list")
        return False
    
    print(f"Found {len(pairs)} twin pairs")
    
    # Validate each pair
    valid_pairs = 0
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        
        folder1, folder2 = pair
        path1 = os.path.join(dataset_path, folder1)
        path2 = os.path.join(dataset_path, folder2)
        
        if os.path.exists(path1) and os.path.exists(path2):
            valid_pairs += 1
    
    print(f"Valid pairs: {valid_pairs}/{len(pairs)}")
    return valid_pairs > 0


def create_sample_pairs_file(dataset_path: str, output_file: str, num_pairs: int = 100):
    """Create a sample pairs.json file for testing"""
    folders = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(folders) < num_pairs * 2:
        print(f"Not enough folders for {num_pairs} pairs. Found {len(folders)} folders.")
        num_pairs = len(folders) // 2
    
    # Randomly pair folders
    random.shuffle(folders)
    pairs = []
    
    for i in range(0, num_pairs * 2, 2):
        pairs.append([folders[i], folders[i + 1]])
    
    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    print(f"Created {len(pairs)} pairs in {output_file}")


def analyze_dataset(dataset_path: str, pairs_file: str):
    """Analyze the dataset and provide statistics"""
    print("=== Dataset Analysis ===")
    
    # Load pairs
    with open(pairs_file, 'r') as f:
        pairs = json.load(f)
    
    # Get all folders
    all_folders = set()
    for pair in pairs:
        all_folders.update(pair)
    
    print(f"Total folders: {len(all_folders)}")
    print(f"Twin pairs: {len(pairs)}")
    
    # Analyze image counts
    image_counts = []
    total_images = 0
    
    for folder in all_folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_counts.append(len(images))
            total_images += len(images)
    
    print(f"Total images: {total_images}")
    print(f"Average images per folder: {sum(image_counts) / len(image_counts):.1f}")
    print(f"Min images per folder: {min(image_counts)}")
    print(f"Max images per folder: {max(image_counts)}")
    
    # Estimate possible pairs
    positive_pairs = 0
    for count in image_counts:
        positive_pairs += count * (count - 1) // 2  # Combinations within folder
    
    # Twin negative pairs
    twin_negative_pairs = 0
    for pair in pairs:
        folder1, folder2 = pair
        path1 = os.path.join(dataset_path, folder1)
        path2 = os.path.join(dataset_path, folder2)
        
        if os.path.exists(path1) and os.path.exists(path2):
            count1 = len([f for f in os.listdir(path1) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            count2 = len([f for f in os.listdir(path2) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            twin_negative_pairs += count1 * count2
    
    print(f"Possible positive pairs: {positive_pairs}")
    print(f"Possible twin negative pairs: {twin_negative_pairs}")
    
    # Estimate dataset split sizes
    total_pairs = positive_pairs + twin_negative_pairs
    print(f"Estimated total training pairs: {total_pairs}")
    print(f"Estimated train/val/test split (80/10/10): {int(total_pairs*0.8)}/{int(total_pairs*0.1)}/{int(total_pairs*0.1)}")


def check_image_quality(dataset_path: str, sample_size: int = 100):
    """Check image quality and properties"""
    import cv2
    from PIL import Image
    
    print("=== Image Quality Check ===")
    
    folders = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    image_paths = []
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images[:5]:  # Sample 5 images per folder
            image_paths.append(os.path.join(folder_path, img))
    
    random.shuffle(image_paths)
    image_paths = image_paths[:sample_size]
    
    widths, heights = [], []
    corrupt_count = 0
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
        except Exception:
            corrupt_count += 1
    
    if widths and heights:
        print(f"Checked {len(image_paths)} images")
        print(f"Corrupt images: {corrupt_count}")
        print(f"Width range: {min(widths)} - {max(widths)}")
        print(f"Height range: {min(heights)} - {max(heights)}")
        print(f"Average size: {sum(widths)/len(widths):.0f} x {sum(heights)/len(heights):.0f}")
        
        # Check for common resolutions
        resolutions = defaultdict(int)
        for w, h in zip(widths, heights):
            resolutions[(w, h)] += 1
        
        print("Most common resolutions:")
        sorted_res = sorted(resolutions.items(), key=lambda x: x[1], reverse=True)
        for res, count in sorted_res[:5]:
            print(f"  {res[0]}x{res[1]}: {count} images")


def generate_dataset_report(validation_results: dict, analysis_results: dict, alignment_results: dict) -> dict:
    """Generate a comprehensive dataset report."""
    report = {
        'dataset_status': 'VALID' if validation_results['valid'] else 'INVALID',
        'summary': {
            'total_subjects': analysis_results['num_subjects'],
            'total_images': analysis_results['num_images'],
            'twin_pairs': analysis_results['num_twin_pairs'],
            'face_detection_rate': alignment_results['detection_rate'],
            'avg_quality_score': alignment_results['avg_quality']
        },
        'data_distribution': analysis_results.get('split_distribution', {}),
        'image_quality': {
            'high_quality_images': alignment_results['high_quality_count'],
            'problematic_images': len(alignment_results.get('problematic_images', [])),
            'quality_distribution': alignment_results.get('quality_distribution', {})
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if not validation_results['valid']:
        report['recommendations'].append("Fix dataset structure issues before training")
    
    if alignment_results['detection_rate'] < 0.95:
        report['recommendations'].append("Consider improving face detection or filtering low-quality images")
    
    if alignment_results['avg_quality'] < 0.7:
        report['recommendations'].append("Dataset quality is low, consider preprocessing or data cleaning")
    
    if analysis_results['num_twin_pairs'] < 100:
        report['recommendations'].append("Limited twin pairs available, consider data augmentation")
    
    if len(alignment_results.get('problematic_images', [])) > analysis_results['num_images'] * 0.1:
        report['recommendations'].append("High number of problematic images, consider manual review")
    
    return report


def preprocess_dataset(data_root: str, output_dir: str, face_size: int = 224, quality_threshold: float = 0.8) -> dict:
    """Preprocess dataset with face alignment and quality filtering."""
    try:
        from mtcnn import MTCNN
        import cv2
        from PIL import Image
        import numpy as np
    except ImportError:
        return {'error': 'Required packages not installed: mtcnn, opencv-python'}
    
    detector = MTCNN()
    results = {
        'total_processed': 0,
        'images_saved': 0,
        'images_skipped': 0,
        'failed_images': [],
        'success_rate': 0.0
    }
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        split_path = Path(data_root) / split
        if split_path.exists():
            output_split_path = Path(output_dir) / split
            output_split_path.mkdir(parents=True, exist_ok=True)
            
            for subject_dir in split_path.iterdir():
                if subject_dir.is_dir():
                    output_subject_path = output_split_path / subject_dir.name
                    output_subject_path.mkdir(exist_ok=True)
                    
                    # Process images in subject directory
                    for img_path in subject_dir.glob('*.jpg'):
                        results['total_processed'] += 1
                        
                        try:
                            # Load image
                            img = cv2.imread(str(img_path))
                            if img is None:
                                results['failed_images'].append(str(img_path))
                                continue
                            
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            # Detect face
                            detections = detector.detect_faces(img_rgb)
                            
                            if len(detections) == 0:
                                results['images_skipped'] += 1
                                continue
                            
                            # Use the largest detection
                            detection = max(detections, key=lambda x: x['box'][2] * x['box'][3])
                            
                            # Check quality
                            if detection['confidence'] < quality_threshold:
                                results['images_skipped'] += 1
                                continue
                            
                            # Extract and align face
                            x, y, w, h = detection['box']
                            # Add some padding
                            padding = int(0.2 * min(w, h))
                            x = max(0, x - padding)
                            y = max(0, y - padding)
                            w = w + 2 * padding
                            h = h + 2 * padding
                            
                            face = img_rgb[y:y+h, x:x+w]
                            
                            # Resize to target size
                            face_resized = cv2.resize(face, (face_size, face_size))
                            
                            # Save processed image
                            output_path = output_subject_path / img_path.name
                            face_pil = Image.fromarray(face_resized)
                            face_pil.save(output_path, quality=95)
                            
                            results['images_saved'] += 1
                            
                        except Exception as e:
                            results['failed_images'].append(f"{img_path}: {str(e)}")
    
    # Calculate success rate
    if results['total_processed'] > 0:
        results['success_rate'] = results['images_saved'] / results['total_processed']
    
    return results


def check_face_alignment(data_root: str, max_samples: int = 1000) -> dict:
    """Check face alignment quality in the dataset."""
    try:
        from mtcnn import MTCNN
        import cv2
        import numpy as np
    except ImportError:
        return {'error': 'Required packages not installed: mtcnn, opencv-python'}
    
    detector = MTCNN()
    results = {
        'total_checked': 0,
        'faces_detected': 0,
        'detection_rate': 0.0,
        'quality_scores': [],
        'avg_quality': 0.0,
        'high_quality_count': 0,
        'problematic_images': []
    }
    
    # Collect image paths
    image_paths = []
    for split_path in [Path(data_root) / split for split in ['train', 'val', 'test']]:
        if split_path.exists():
            for subject_dir in split_path.iterdir():
                if subject_dir.is_dir():
                    image_paths.extend(list(subject_dir.glob('*.jpg')))
    
    # Sample if too many images
    if len(image_paths) > max_samples:
        image_paths = random.sample(image_paths, max_samples)
    
    for img_path in image_paths:
        results['total_checked'] += 1
        
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                results['problematic_images'].append(str(img_path))
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = detector.detect_faces(img_rgb)
            
            if len(detections) > 0:
                results['faces_detected'] += 1
                
                # Use the best detection
                best_detection = max(detections, key=lambda x: x['confidence'])
                quality_score = best_detection['confidence']
                results['quality_scores'].append(quality_score)
                
                if quality_score > 0.9:
                    results['high_quality_count'] += 1
            else:
                results['problematic_images'].append(str(img_path))
        
        except Exception as e:
            results['problematic_images'].append(f"{img_path}: {str(e)}")
    
    # Calculate statistics
    if results['total_checked'] > 0:
        results['detection_rate'] = results['faces_detected'] / results['total_checked']
    
    if results['quality_scores']:
        results['avg_quality'] = sum(results['quality_scores']) / len(results['quality_scores'])
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Dataset utilities for ND-TWIN')
    parser.add_argument('--mode', choices=['validate', 'create_pairs', 'analyze', 'check_quality'], 
                       required=True, help='Utility mode')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--pairs_file', type=str, help='Path to pairs.json file')
    parser.add_argument('--output_file', type=str, help='Output file path')
    parser.add_argument('--num_pairs', type=int, default=100, help='Number of pairs to create')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size for quality check')
    
    args = parser.parse_args()
    
    if args.mode == 'validate':
        if not args.pairs_file:
            print("--pairs_file required for validate mode")
            return
        
        print("Validating dataset structure...")
        dataset_valid = validate_dataset_structure(args.dataset_path)
        
        print("Validating pairs file...")
        pairs_valid = validate_pairs_file(args.pairs_file, args.dataset_path)
        
        if dataset_valid and pairs_valid:
            print("✓ Dataset validation passed")
        else:
            print("✗ Dataset validation failed")
    
    elif args.mode == 'create_pairs':
        if not args.output_file:
            print("--output_file required for create_pairs mode")
            return
        
        create_sample_pairs_file(args.dataset_path, args.output_file, args.num_pairs)
    
    elif args.mode == 'analyze':
        if not args.pairs_file:
            print("--pairs_file required for analyze mode")
            return
        
        analyze_dataset(args.dataset_path, args.pairs_file)
    
    elif args.mode == 'check_quality':
        check_image_quality(args.dataset_path, args.sample_size)


if __name__ == "__main__":
    main()
