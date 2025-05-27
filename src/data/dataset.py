import os
import json
import random
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from mtcnn import MTCNN
import cv2


class FaceAligner:
    """Face alignment using MTCNN for consistent face orientation"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.detector = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = target_size
    
    def align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Align face in image using eye coordinates"""
        try:
            # Detect face and landmarks
            result = self.detector.detect(image)
            if result[0] is None:
                return None
            
            boxes, probs, landmarks = result
            if len(boxes) == 0:
                return None
            
            # Use the most confident detection
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            landmark = landmarks[best_idx]
            
            # Extract eye coordinates
            left_eye = landmark[0]
            right_eye = landmark[1]
            
            # Calculate angle and align
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotate image
            center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            
            # Crop face region with some padding
            x, y, w, h = box.astype(int)
            padding = int(max(w, h) * 0.3)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face = aligned[y1:y2, x1:x2]
            face = cv2.resize(face, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            return face
            
        except Exception as e:
            print(f"Face alignment failed: {e}")
            return None


class NDTwinDataset(Dataset):
    """Dataset class for ND-TWIN identical twin verification"""
    
    def __init__(
        self,
        dataset_path: str,
        pairs_file: str,
        mode: str = 'train',
        image_size: int = 224,
        use_face_alignment: bool = True,
        augment: bool = True,
        twin_negative_ratio: float = 0.7,  # Ratio of twin pairs in negative samples
        use_preprocessed: bool = False,  # Whether to use preprocessed images
        preprocessed_path: Optional[str] = None  # Path to preprocessed images
    ):
        self.dataset_path = dataset_path
        self.mode = mode
        self.image_size = image_size
        self.use_face_alignment = use_face_alignment
        self.twin_negative_ratio = twin_negative_ratio
        self.use_preprocessed = use_preprocessed
        self.preprocessed_path = preprocessed_path
        
        # Validate preprocessing setup
        if use_preprocessed:
            if preprocessed_path is None:
                raise ValueError("preprocessed_path must be provided when use_preprocessed=True")
            if not os.path.exists(preprocessed_path):
                raise ValueError(f"Preprocessed path does not exist: {preprocessed_path}")
            self.image_root = preprocessed_path
            print(f"Using preprocessed images from: {preprocessed_path}")
        else:
            self.image_root = dataset_path
            # Initialize face aligner only for live processing
            if use_face_alignment:
                self.face_aligner = FaceAligner((image_size, image_size))
            else:
                self.face_aligner = None
            print(f"Using live processing from: {dataset_path}")
        
        # Load pairs data
        with open(pairs_file, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Get all folder names
        self.all_folders = [pair for twin_pair in self.twin_pairs for pair in twin_pair]
        self.folder_to_twin = {}
        for twin_pair in self.twin_pairs:
            self.folder_to_twin[twin_pair[0]] = twin_pair[1]
            self.folder_to_twin[twin_pair[1]] = twin_pair[0]
        
        # Build image paths dictionary
        self.folder_images = {}
        self._build_image_paths()
        
        # Create training pairs
        self.pairs = self._create_pairs()
        
        # Setup transforms
        self.transforms = self._get_transforms(augment and mode == 'train')
    
    def _build_image_paths(self):
        """Build dictionary of available images for each folder"""
        missing_folders = []
        for folder in self.all_folders:
            folder_path = os.path.join(self.image_root, folder)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    self.folder_images[folder] = images
                else:
                    missing_folders.append(folder)
            else:
                missing_folders.append(folder)
        
        if missing_folders:
            print(f"Warning: {len(missing_folders)} folders missing or empty in {self.image_root}")
            if len(missing_folders) <= 10:
                print(f"Missing folders: {missing_folders}")
            
        print(f"Found {len(self.folder_images)} folders with images out of {len(self.all_folders)} total folders")
    
    def _create_pairs(self) -> List[Tuple[str, str, str, str, int]]:
        """Create positive and negative pairs for training/validation"""
        pairs = []
        
        # Create positive pairs (same person)
        for folder in self.all_folders:
            if folder not in self.folder_images or len(self.folder_images[folder]) < 2:
                continue
            
            images = self.folder_images[folder]
            # Create all possible pairs within the same folder
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    pairs.append((folder, images[i], folder, images[j], 1))
        
        # Create negative pairs (different persons)
        num_positive = len(pairs)
        negative_pairs = []
        
        # Twin negative pairs (hardest negatives)
        twin_negatives_needed = int(num_positive * self.twin_negative_ratio)
        for _ in range(twin_negatives_needed):
            # Select a random twin pair
            twin_pair = random.choice(self.twin_pairs)
            folder1, folder2 = twin_pair
            
            if (folder1 in self.folder_images and folder2 in self.folder_images and
                len(self.folder_images[folder1]) > 0 and len(self.folder_images[folder2]) > 0):
                
                img1 = random.choice(self.folder_images[folder1])
                img2 = random.choice(self.folder_images[folder2])
                negative_pairs.append((folder1, img1, folder2, img2, 0))
        
        # Random negative pairs
        random_negatives_needed = num_positive - len(negative_pairs)
        for _ in range(random_negatives_needed):
            folder1 = random.choice(self.all_folders)
            folder2 = random.choice(self.all_folders)
            
            # Ensure they're not the same person or twins
            if (folder1 != folder2 and 
                folder1 not in self.folder_images.get(folder2, []) and
                self.folder_to_twin.get(folder1) != folder2 and
                folder1 in self.folder_images and folder2 in self.folder_images and
                len(self.folder_images[folder1]) > 0 and len(self.folder_images[folder2]) > 0):
                
                img1 = random.choice(self.folder_images[folder1])
                img2 = random.choice(self.folder_images[folder2])
                negative_pairs.append((folder1, img1, folder2, img2, 0))
        
        pairs.extend(negative_pairs)
        random.shuffle(pairs)
        return pairs
    
    def _get_transforms(self, augment: bool):
        """Get image transformations"""
        if augment:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _load_and_preprocess_image(self, folder: str, image_name: str) -> torch.Tensor:
        """Load and preprocess a single image"""
        image_path = os.path.join(self.image_root, folder, image_name)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_preprocessed:
            # Preprocessed images are already aligned and resized
            # Just ensure correct size
            if image.shape[:2] != (self.image_size, self.image_size):
                image = cv2.resize(image, (self.image_size, self.image_size))
        else:
            # Live processing: apply face alignment if enabled
            if self.face_aligner is not None:
                aligned_face = self.face_aligner.align_face(image)
                if aligned_face is not None:
                    image = aligned_face
            
            # Resize if face alignment failed or not used
            if image.shape[:2] != (self.image_size, self.image_size):
                image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Apply transforms
        transformed = self.transforms(image=image)
        return transformed['image']
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        folder1, img1, folder2, img2, label = self.pairs[idx]
        
        # Load images
        image1 = self._load_and_preprocess_image(folder1, img1)
        image2 = self._load_and_preprocess_image(folder2, img2)
        
        return {
            'image1': image1,
            'image2': image2,
            'label': torch.tensor(label, dtype=torch.float32),
            'is_twin_pair': torch.tensor(
                1 if self.folder_to_twin.get(folder1) == folder2 else 0, 
                dtype=torch.float32
            )
        }


def create_data_loaders(
    dataset_path: str,
    pairs_file: str,
    config: dict,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Extract data configuration
    data_config = config['data']
    
    # Create full dataset
    full_dataset = NDTwinDataset(
        dataset_path=dataset_path,
        pairs_file=pairs_file,
        mode='train',
        image_size=data_config['image_size'],
        twin_negative_ratio=0.7,
        use_preprocessed=data_config.get('use_preprocessed', False),
        preprocessed_path=data_config.get('preprocessed_path', None),
        use_face_alignment=data_config.get('use_face_alignment', True)
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create separate dataset instances for different modes
    train_dataset.dataset.mode = 'train'
    val_dataset.dataset.mode = 'val'
    test_dataset.dataset.mode = 'test'
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def validate_preprocessed_dataset(
    original_dataset_path: str,
    preprocessed_path: str,
    pairs_file: str
) -> Dict[str, any]:
    """
    Validate that preprocessed dataset is complete and ready for use.
    
    Args:
        original_dataset_path: Path to original dataset
        preprocessed_path: Path to preprocessed images
        pairs_file: Path to pairs.json file
        
    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check if preprocessed path exists
    if not os.path.exists(preprocessed_path):
        results['valid'] = False
        results['errors'].append(f"Preprocessed path does not exist: {preprocessed_path}")
        return results
    
    # Load pairs data
    try:
        with open(pairs_file, 'r') as f:
            twin_pairs = json.load(f)
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Failed to load pairs file: {e}")
        return results
    
    # Get all required folders
    all_folders = [pair for twin_pair in twin_pairs for pair in twin_pair]
    
    # Check original dataset statistics
    original_stats = {'folders': 0, 'images': 0, 'missing_folders': []}
    for folder in all_folders:
        original_folder_path = os.path.join(original_dataset_path, folder)
        if os.path.exists(original_folder_path):
            images = [f for f in os.listdir(original_folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                original_stats['folders'] += 1
                original_stats['images'] += len(images)
            else:
                original_stats['missing_folders'].append(folder)
        else:
            original_stats['missing_folders'].append(folder)
    
    # Check preprocessed dataset statistics
    preprocessed_stats = {'folders': 0, 'images': 0, 'missing_folders': []}
    for folder in all_folders:
        preprocessed_folder_path = os.path.join(preprocessed_path, folder)
        if os.path.exists(preprocessed_folder_path):
            images = [f for f in os.listdir(preprocessed_folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                preprocessed_stats['folders'] += 1
                preprocessed_stats['images'] += len(images)
            else:
                preprocessed_stats['missing_folders'].append(folder)
        else:
            preprocessed_stats['missing_folders'].append(folder)
    
    # Compare statistics
    results['statistics'] = {
        'original': original_stats,
        'preprocessed': preprocessed_stats,
        'total_folders_expected': len(all_folders)
    }
    
    # Check for missing folders in preprocessed dataset
    if preprocessed_stats['missing_folders']:
        if len(preprocessed_stats['missing_folders']) > len(original_stats['missing_folders']):
            results['warnings'].append(
                f"Preprocessed dataset is missing {len(preprocessed_stats['missing_folders'])} folders "
                f"that exist in original dataset"
            )
    
    # Check if preprocessed has significantly fewer images
    if original_stats['images'] > 0:
        coverage = preprocessed_stats['images'] / original_stats['images']
        if coverage < 0.8:
            results['warnings'].append(
                f"Preprocessed dataset has only {coverage*100:.1f}% of original images "
                f"({preprocessed_stats['images']} vs {original_stats['images']})"
            )
    
    # Check if any preprocessing was done
    if preprocessed_stats['images'] == 0:
        results['valid'] = False
        results['errors'].append("No preprocessed images found")
    
    return results


def print_dataset_statistics(
    dataset_path: str,
    pairs_file: str,
    preprocessed_path: Optional[str] = None
):
    """Print detailed statistics about the dataset"""
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    # Load pairs data
    with open(pairs_file, 'r') as f:
        twin_pairs = json.load(f)
    
    print(f"Twin pairs defined: {len(twin_pairs)}")
    all_folders = [pair for twin_pair in twin_pairs for pair in twin_pair]
    print(f"Total individuals: {len(all_folders)}")
    
    # Original dataset statistics
    print(f"\nOriginal Dataset: {dataset_path}")
    original_folders = 0
    original_images = 0
    for folder in all_folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                original_folders += 1
                original_images += len(images)
    
    print(f"  Available folders: {original_folders}/{len(all_folders)}")
    print(f"  Total images: {original_images}")
    if original_folders > 0:
        print(f"  Average images per folder: {original_images/original_folders:.1f}")
    
    # Preprocessed dataset statistics if available
    if preprocessed_path and os.path.exists(preprocessed_path):
        print(f"\nPreprocessed Dataset: {preprocessed_path}")
        preprocessed_folders = 0
        preprocessed_images = 0
        for folder in all_folders:
            folder_path = os.path.join(preprocessed_path, folder)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    preprocessed_folders += 1
                    preprocessed_images += len(images)
        
        print(f"  Available folders: {preprocessed_folders}/{len(all_folders)}")
        print(f"  Total images: {preprocessed_images}")
        if preprocessed_folders > 0:
            print(f"  Average images per folder: {preprocessed_images/preprocessed_folders:.1f}")
        
        if original_images > 0:
            coverage = preprocessed_images / original_images
            print(f"  Coverage: {coverage*100:.1f}% of original images")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test dataset loading
    config = {
        'data': {
            'image_size': 224,
            'batch_size': 4,
            'num_workers': 2
        }
    }
    
    dataset = NDTwinDataset(
        dataset_path="path/to/dataset",
        pairs_file="path/to/pairs.json",
        mode='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image1 shape: {sample['image1'].shape}")
    print(f"Image2 shape: {sample['image2'].shape}")
    print(f"Label: {sample['label']}")