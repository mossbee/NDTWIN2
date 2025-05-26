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
        twin_negative_ratio: float = 0.7  # Ratio of twin pairs in negative samples
    ):
        self.dataset_path = dataset_path
        self.mode = mode
        self.image_size = image_size
        self.use_face_alignment = use_face_alignment
        self.twin_negative_ratio = twin_negative_ratio
        
        # Initialize face aligner
        if use_face_alignment:
            self.face_aligner = FaceAligner((image_size, image_size))
        else:
            self.face_aligner = None
        
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
        for folder in self.all_folders:
            folder_path = os.path.join(dataset_path, folder)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                self.folder_images[folder] = images
        
        # Create training pairs
        self.pairs = self._create_pairs()
        
        # Setup transforms
        self.transforms = self._get_transforms(augment and mode == 'train')
    
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
        image_path = os.path.join(self.dataset_path, folder, image_name)
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Face alignment if enabled
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
    
    # Create full dataset
    full_dataset = NDTwinDataset(
        dataset_path=dataset_path,
        pairs_file=pairs_file,
        mode='train',
        image_size=config['data']['image_size'],
        twin_negative_ratio=0.7
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
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


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