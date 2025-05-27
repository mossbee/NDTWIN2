#!/usr/bin/env python3
"""
Preprocessing script for ND_TWIN dataset.

This script performs face alignment on all images in the dataset and saves them
to a processed directory structure. This preprocessing step is done once, and
during training we load from the preprocessed images instead of processing
each image on-the-fly.
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Face detection and alignment
from mtcnn import MTCNN


class FacePreprocessor:
    """Enhanced face alignment and preprocessing for consistent face representation."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 quality_threshold: float = 0.9,
                 padding_ratio: float = 0.3):
        """
        Initialize face preprocessor.
        
        Args:
            target_size: Target size for aligned faces
            quality_threshold: Minimum confidence threshold for face detection
            padding_ratio: Padding around detected face region
        """
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        self.padding_ratio = padding_ratio
        self.detector = MTCNN()
        
    def align_face(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Align face in image using facial landmarks.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Tuple of (aligned_face, metadata)
        """
        metadata = {
            'success': False,
            'confidence': 0.0,
            'face_detected': False,
            'alignment_applied': False,
            'error': None
        }
        
        try:
            # Detect faces
            result = self.detector.detect_faces(image)
            box = result[0]["box"]
            prob = result[0]["confidence"] 
            landmark = result[0]["keypoints"]
            if box is None or len(box) == 0:
                metadata['error'] = 'No face detected'
                return None, metadata
            
            metadata['face_detected'] = True
                        
            metadata['confidence'] = float(prob)
            
            # Check quality threshold
            if prob < self.quality_threshold:
                metadata['error'] = f'Low confidence: {prob:.3f} < {self.quality_threshold}'
                return None, metadata
            
            # Perform alignment if landmarks are available
            aligned_image = image.copy()
            if landmark is not None:
                aligned_image = self._align_using_landmarks(image, landmark)
                metadata['alignment_applied'] = True
            
            # Extract and crop face region
            x, y, w, h = box
            
            # Add padding
            padding_w = int(w * self.padding_ratio)
            padding_h = int(h * self.padding_ratio)
            
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(aligned_image.shape[1], x + w + padding_w)
            y2 = min(aligned_image.shape[0], y + h + padding_h)
            
            # Crop face
            face = aligned_image[y1:y2, x1:x2]
            
            # Resize to target size
            face_resized = cv2.resize(face, self.target_size, interpolation=cv2.INTER_CUBIC)
            
            metadata['success'] = True
            return face_resized, metadata
            
        except Exception as e:
            metadata['error'] = str(e)
            return None, metadata
    
    def _align_using_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face using eye landmarks."""
        try:
            # Get eye coordinates
            left_eye = landmarks[3]  # Left eye
            right_eye = landmarks[2]  # Right eye
            
            # Calculate angle between eyes
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Get rotation matrix
            center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            aligned = cv2.warpAffine(
                image, rot_matrix, 
                (image.shape[1], image.shape[0]), 
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            return aligned
        except:
            return image


def process_single_image(args):
    """Process a single image file."""
    input_path, output_path, preprocessor = args
    
    try:
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            return False, f"Failed to load image: {input_path}"
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Align face
        aligned_face, metadata = preprocessor.align_face(image_rgb)
        
        if aligned_face is None:
            return False, f"Face alignment failed: {metadata.get('error', 'Unknown error')}"
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save aligned face
        aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(output_path), aligned_face_bgr, 
                             [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not success:
            return False, f"Failed to save image: {output_path}"
        
        return True, metadata
        
    except Exception as e:
        return False, f"Processing error: {str(e)}"


def preprocess_dataset(input_dir: str, 
                      output_dir: str,
                      pairs_file: str = None,
                      target_size: Tuple[int, int] = (224, 224),
                      quality_threshold: float = 0.9,
                      num_workers: int = None) -> Dict:
    """
    Preprocess entire dataset with face alignment.
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for processed images
        pairs_file: Path to twins pairs file (optional)
        target_size: Target size for aligned faces
        quality_threshold: Minimum confidence threshold
        num_workers: Number of parallel workers
        
    Returns:
        Dictionary with processing statistics
    """
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    logging.info(f"Starting dataset preprocessing with {num_workers} workers")
    logging.info(f"Input: {input_dir}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"Target size: {target_size}")
    logging.info(f"Quality threshold: {quality_threshold}")
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(
        target_size=target_size,
        quality_threshold=quality_threshold
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all image files
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Find all images in subdirectories
    image_files = []
    for subject_dir in input_path.iterdir():
        if subject_dir.is_dir():
            for img_file in subject_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    # Maintain directory structure
                    relative_path = img_file.relative_to(input_path)
                    output_file_path = output_path / relative_path
                    image_files.append((img_file, output_file_path))
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Prepare processing arguments
    process_args = [(input_file, output_file, preprocessor) 
                   for input_file, output_file in image_files]
    
    # Process images in parallel
    results = {
        'total_images': len(image_files),
        'successful': 0,
        'failed': 0,
        'failed_files': [],
        'processing_stats': {
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0
        }
    }
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_image, args): args[1] 
            for args in process_args
        }
        
        # Process results with progress bar
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            for future in as_completed(future_to_file):
                output_file = future_to_file[future]
                try:
                    success, metadata = future.result()
                    
                    if success:
                        results['successful'] += 1
                        
                        # Track confidence statistics
                        if isinstance(metadata, dict) and 'confidence' in metadata:
                            conf = metadata['confidence']
                            if conf >= 0.95:
                                results['processing_stats']['high_confidence'] += 1
                            elif conf >= 0.90:
                                results['processing_stats']['medium_confidence'] += 1
                            else:
                                results['processing_stats']['low_confidence'] += 1
                    else:
                        results['failed'] += 1
                        results['failed_files'].append({
                            'file': str(output_file),
                            'error': metadata if isinstance(metadata, str) else str(metadata)
                        })
                        
                except Exception as e:
                    results['failed'] += 1
                    results['failed_files'].append({
                        'file': str(output_file),
                        'error': str(e)
                    })
                
                pbar.update(1)
    
    # Copy pairs file if provided
    if pairs_file and os.path.exists(pairs_file):
        output_pairs_file = output_path / 'pairs.json'
        shutil.copy2(pairs_file, output_pairs_file)
        logging.info(f"Copied pairs file to {output_pairs_file}")
    
    # Save processing report
    report_file = output_path / 'preprocessing_report.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate success rate
    success_rate = results['successful'] / results['total_images'] if results['total_images'] > 0 else 0
    
    logging.info(f"Preprocessing completed!")
    logging.info(f"Total images: {results['total_images']}")
    logging.info(f"Successful: {results['successful']}")
    logging.info(f"Failed: {results['failed']}")
    logging.info(f"Success rate: {success_rate:.2%}")
    logging.info(f"Report saved to: {report_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Preprocess ND_TWIN dataset with face alignment')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed images')
    parser.add_argument('--pairs_file', type=str, default=None,
                        help='Path to twins pairs JSON file')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Target image size (default: 224)')
    parser.add_argument('--quality_threshold', type=float, default=0.9,
                        help='Minimum face detection confidence (default: 0.9)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logging.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Run preprocessing
    results = preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pairs_file=args.pairs_file,
        target_size=(args.image_size, args.image_size),
        quality_threshold=args.quality_threshold,
        num_workers=args.num_workers
    )
    
    # Print final summary
    print(f"\n{'='*50}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total images processed: {results['total_images']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['successful']/results['total_images']:.2%}")
    print(f"Output directory: {args.output_dir}")
    
    if results['failed'] > 0:
        print(f"\nFailed files saved in preprocessing_report.json")


if __name__ == '__main__':
    main()
