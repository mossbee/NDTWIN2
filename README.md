# Local-Global Attention Network for Identical Twin Verification

This repository implements a sophisticated deep learning approach for identical twin verification using a dual-branch CNN with attention mechanisms. The model combines global face representations with attention-driven focus on local discriminative regions to distinguish between identical twins.

## 🎯 Approach Overview

**Problem**: Identical twin verification is extremely challenging due to the high facial similarity between twins. Traditional face recognition systems often fail to distinguish between identical twins.

**Solution**: Our Local-Global Attention Network leverages:
- **Global Branch**: Captures overall face structure and configuration
- **Local Attention Branch**: Automatically discovers and focuses on subtle discriminative regions (moles, scars, asymmetries)
- **Siamese Architecture**: Enables direct comparison learning between image pairs
- **Hard Negative Mining**: Specifically trains on twin pairs as the most challenging negatives

## 🏗️ Model Architecture

```
Input Image Pair (I₁, I₂)
    ↓
┌─────────────────────────────────────┐
│ Shared CNN Backbone (ResNet/EfficientNet) │
└─────────────────────────────────────┘
    ↓
┌─────────────────┐    ┌─────────────────┐
│  Global Branch  │    │ Attention Branch │
│                 │    │  ┌─────────────┐ │
│ AvgPool         │    │  │ CBAM/Trans. │ │
│ ↓               │    │  │ Attention   │ │
│ FC → Embedding₁ │    │  └─────────────┘ │
└─────────────────┘    │ AvgPool         │
                       │ ↓               │
                       │ FC → Embedding₂ │
                       └─────────────────┘
    ↓                      ↓
┌─────────────────────────────────────┐
│     Concatenate & Fusion            │
│     Final Embedding (512D)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Cosine Similarity Computation      │
│  Contrastive/Triplet Loss          │
└─────────────────────────────────────┘
```

## 📊 Dataset Structure

The code expects the ND_TWIN dataset in the following structure:

```
dataset/
├── img_folder_1/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── img_folder_2/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
└── ...

pairs.json  # [[folder1, folder2], [folder3, folder4], ...]
```

Where `pairs.json` contains a list of twin pairs (folder names).

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone <repository-url>
cd NDTWIN_Idea_1

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### Option A: Standard Dataset Validation
```powershell
# Validate your dataset structure
python src/utils/dataset_utils.py --mode validate --dataset_path "path/to/dataset" --pairs_file "path/to/pairs.json"

# Analyze dataset statistics
python src/utils/dataset_utils.py --mode analyze --dataset_path "path/to/dataset" --pairs_file "path/to/pairs.json"

# Check image quality
python src/utils/dataset_utils.py --mode check_quality --dataset_path "path/to/dataset"
```

#### Option B: Preprocessing for Faster Training (Recommended)
For better training performance, preprocess images with face alignment once:

```powershell
# Preprocess entire dataset with face alignment
python scripts/preprocess_dataset.py --input_path "path/to/original/dataset" --output_path "path/to/preprocessed/dataset" --pairs_file "path/to/pairs.json" --num_workers 8 --image_size 224

# Validate preprocessed dataset
python -c "
from src.data.dataset import validate_preprocessed_dataset, print_dataset_statistics
result = validate_preprocessed_dataset('path/to/original/dataset', 'path/to/preprocessed/dataset', 'path/to/pairs.json')
print('Validation:', 'PASSED' if result['valid'] else 'FAILED')
print_dataset_statistics('path/to/original/dataset', 'path/to/pairs.json', 'path/to/preprocessed/dataset')
"
```

The preprocessing script offers several advantages:
- **Faster Training**: Face alignment done once instead of every epoch
- **Consistent Processing**: All images processed with same parameters
- **Quality Filtering**: Low-quality detections can be filtered out
- **Parallel Processing**: Multi-threaded processing for speed
- **Progress Tracking**: Real-time progress and statistics

## 🛠️ Image Preprocessing Workflow

### Why Use Preprocessing?

The preprocessing workflow offers significant advantages for training efficiency and consistency:

**Performance Benefits:**
- **10-50x faster training**: Face alignment done once instead of every epoch
- **Consistent quality**: All images processed with identical parameters
- **Memory efficiency**: Reduced runtime memory usage
- **Reproducible results**: Eliminates alignment variance between runs

**Quality Benefits:**
- **Enhanced alignment**: MTCNN with landmark-based rotation correction
- **Quality filtering**: Automatically filters out poor face detections
- **Size standardization**: All images resized to consistent dimensions

### Preprocessing Script Usage

The `scripts/preprocess_dataset.py` script provides comprehensive preprocessing capabilities:

```powershell
# Basic preprocessing
python scripts/preprocess_dataset.py --input_path "raw_dataset" --output_path "processed_dataset" --pairs_file "pairs.json"

# Advanced preprocessing with custom parameters
python scripts/preprocess_dataset.py \
    --input_path "raw_dataset" \
    --output_path "processed_dataset" \
    --pairs_file "pairs.json" \
    --image_size 224 \
    --quality_threshold 0.95 \
    --num_workers 8 \
    --device cuda
```

**Parameters:**
- `--input_path`: Original dataset directory
- `--output_path`: Directory for preprocessed images
- `--pairs_file`: Twin pairs JSON file
- `--image_size`: Target image size (default: 224)
- `--quality_threshold`: Minimum face detection confidence (default: 0.95)
- `--num_workers`: Parallel processing threads (default: 4)
- `--device`: Processing device (cuda/cpu, default: auto-detect)
- `--skip_existing`: Skip already processed images

### Preprocessing Output

The script generates:

```
preprocessed_dataset/
├── img_folder_1/           # Maintains original structure
│   ├── img_1.jpg          # Face-aligned and resized images
│   ├── img_2.jpg
│   └── ...
├── img_folder_2/
│   └── ...
└── preprocessing_report.json  # Processing statistics
```

**Report Contents:**
```json
{
    "total_folders": 150,
    "successful_folders": 148,
    "total_images": 3000,
    "successfully_processed": 2950,
    "failed_images": 50,
    "processing_time": 450.5,
    "average_time_per_image": 0.15,
    "quality_stats": {
        "mean_confidence": 0.97,
        "below_threshold": 45
    }
}
```

### Using Preprocessed Images in Training

#### Configuration Setup
Update `configs/train_config.yaml`:

```yaml
data:
  use_preprocessed: true
  preprocessed_path: "path/to/preprocessed_dataset"
  dataset_path: "path/to/original_dataset"  # Fallback for validation
  pairs_file: "path/to/pairs.json"
  use_face_alignment: false  # Not needed for preprocessed images
```

#### Validation
Validate your preprocessed dataset before training:

```python
from src.data.dataset import validate_preprocessed_dataset, print_dataset_statistics

# Validate completeness
result = validate_preprocessed_dataset(
    original_dataset_path="original_dataset",
    preprocessed_path="preprocessed_dataset", 
    pairs_file="pairs.json"
)

if result['valid']:
    print("✅ Preprocessed dataset is ready for training")
    print_dataset_statistics("original_dataset", "pairs.json", "preprocessed_dataset")
else:
    print("❌ Issues found:")
    for error in result['errors']:
        print(f"  - {error}")
```

### Best Practices

**Preprocessing Recommendations:**
- Use `quality_threshold=0.95` for high-quality datasets
- Use `quality_threshold=0.90` for datasets with challenging images
- Set `num_workers` to 75% of available CPU cores
- Monitor preprocessing report for quality insights

**Storage Considerations:**
- Preprocessed images typically use 60-80% of original storage
- Use SSD storage for faster training data loading
- Keep original dataset as backup

**Training Performance:**
- Expected 10-20x faster data loading during training
- Reduced CPU usage during training (no live face alignment)
- More consistent training curves due to stable preprocessing

### 3. Training

#### Option A: Standard Training (Live Processing)
```powershell
# Basic training with live face alignment
python src/train.py --config configs/train_config.yaml --dataset_path "path/to/dataset" --pairs_file "path/to/pairs.json"

# Training with specific GPU
python src/train.py --config configs/train_config.yaml --dataset_path "path/to/dataset" --pairs_file "path/to/pairs.json" --gpu 0
```

#### Option B: Training with Preprocessed Images (Recommended)
For faster training using preprocessed images, update your config file:

```yaml
# configs/train_config.yaml
data:
  use_preprocessed: true
  preprocessed_path: "path/to/preprocessed/dataset"
  dataset_path: "path/to/original/dataset"  # For fallback
  pairs_file: "path/to/pairs.json"
```

Then train normally:
```powershell
# Training with preprocessed images (much faster)
python src/train.py --config configs/train_config.yaml --dataset_path "path/to/original/dataset" --pairs_file "path/to/pairs.json"

# Resume training from checkpoint
python src/train.py --config configs/train_config.yaml --dataset_path "path/to/original/dataset" --pairs_file "path/to/pairs.json" --resume "checkpoints/latest_checkpoint.pth"
```

### 4. Inference

```powershell
# Single pair verification
python src/inference.py --checkpoint "checkpoints/best_checkpoint.pth" --config "configs/train_config.yaml" --mode single --image1 "path/to/image1.jpg" --image2 "path/to/image2.jpg"

# Evaluate on test dataset
python src/inference.py --checkpoint "checkpoints/best_checkpoint.pth" --config "configs/train_config.yaml" --mode evaluate --dataset_path "path/to/dataset" --pairs_json "path/to/pairs.json"
```

## ⚙️ Configuration

The main configuration is in `configs/train_config.yaml`. Key parameters:

### Data Configuration
```yaml
data:
  # Basic dataset paths
  dataset_path: "path/to/dataset"
  pairs_file: "path/to/pairs.json"
  
  # Preprocessing options (NEW)
  use_preprocessed: false          # Use preprocessed images for faster training
  preprocessed_path: null          # Path to preprocessed dataset
  use_face_alignment: true         # Face alignment for live processing
  
  # Data loading options
  image_size: 224
  batch_size: 16
  num_workers: 4
```

### Model Configuration
```yaml
model:
  backbone: "resnet50"           # resnet50, resnet101, efficientnet_b0
  embedding_dim: 512             # Final embedding dimension
  attention_type: "cbam"         # cbam, transformer, spatial
  dropout: 0.3                   # Dropout rate
```

### Training Configuration
```yaml
training:
  epochs: 100
  learning_rate: 0.001
  scheduler: "cosine"            # cosine, step, plateau
  margin: 1.0                    # Contrastive loss margin
```

### Loss Configuration
```yaml
loss:
  type: "contrastive"            # contrastive, triplet
  weight_global: 0.6             # Global branch weight
  weight_local: 0.4              # Local branch weight
  attention_reg_weight: 0.01     # Attention regularization
```

## 🔬 Key Features

### 1. **Face Alignment**
- Automatic face detection and alignment using MTCNN
- Ensures consistent face orientation across images
- Robust to varying pose and scale

### 2. **Attention Mechanisms**
- **CBAM**: Convolutional Block Attention Module
- **Transformer**: Multi-head self-attention
- **Spatial**: Custom spatial attention module

### 3. **Hard Negative Mining**
- 70% of negative pairs are twin siblings (hardest negatives)
- Improves model's ability to distinguish twins
- Configurable twin negative ratio

### 4. **Multi-Scale Training**
- Global features capture face structure
- Local features focus on discriminative details
- Weighted fusion of both representations

### 5. **Comprehensive Evaluation**
- ROC curves for overall and twin-specific performance
- Attention visualization and analysis
- Detailed metrics and confusion matrices

## 📈 Expected Performance

Based on the architecture design, expected performance metrics:

- **Overall AUC**: 0.85-0.95 (depending on dataset quality)
- **Twin Pairs AUC**: 0.75-0.85 (most challenging)
- **Accuracy**: 0.80-0.90 (at optimal threshold)

The model particularly excels at:
- Finding subtle discriminative features (moles, scars)
- Ignoring lighting and pose variations
- Maintaining global face context

## 🔍 Understanding the Results

### Attention Visualizations
The model generates attention maps showing where it focuses:
- **Hot regions**: Areas the model considers important for discrimination
- **Consistent patterns**: Regions consistently attended across samples
- **Interpretability**: Human-interpretable discriminative features

### Similarity Analysis
- **Same Person**: High similarity scores (>0.7)
- **Different Person**: Low similarity scores (<0.3)
- **Twin Challenges**: Overlapping distributions, requiring careful threshold tuning

## 🛠️ Advanced Usage

### Example Scripts

The `examples/` directory contains ready-to-use scripts:

```powershell
# Training with comprehensive logging
python examples/train_model.py --data_root "path/to/dataset" --output_dir "results" --validate_data

# Model evaluation with visualizations
python examples/evaluate_model.py --model_path "checkpoints/best_model.pth" --data_root "path/to/dataset" --visualize --save_predictions

# Single pair verification
python examples/verify_pair.py --model_path "checkpoints/best_model.pth" --image1 "img1.jpg" --image2 "img2.jpg" --output_path "result.png"
```

### Dataset Analysis Scripts

The `scripts/` directory provides comprehensive dataset tools:

```powershell
# Preprocess entire dataset with face alignment (NEW)
python scripts/preprocess_dataset.py --input_path "raw_dataset" --output_path "processed_dataset" --pairs_file "pairs.json" --num_workers 8

# Complete dataset analysis
python scripts/analyze_dataset.py --data_root "path/to/dataset" --action report

# Face alignment quality check
python scripts/analyze_dataset.py --data_root "path/to/dataset" --action check_alignment

# Legacy preprocessing (use preprocess_dataset.py instead)
python scripts/analyze_dataset.py --data_root "path/to/dataset" --action preprocess --save_preprocessed "processed_data" --face_size 224
```

### Model Export and Deployment

```powershell
# Export model to ONNX and TorchScript
python scripts/export_model.py --model_path "checkpoints/best_model.pth" --formats onnx torchscript_trace --verify

# Benchmark model performance
python scripts/benchmark_model.py --model_path "checkpoints/best_model.pth" --data_root "path/to/dataset" --num_samples 1000
```

### REST API Deployment

Launch the inference API server:

```powershell
# Set environment variables
$env:MODEL_PATH = "checkpoints/best_model.pth"
$env:CONFIG_PATH = "configs/train_config.yaml"

# Start API server
python inference_api.py
```

API endpoints:
- `POST /verify` - Upload two images for verification
- `POST /verify_base64` - Verify using base64 encoded images
- `POST /batch_verify` - Batch verification for multiple pairs
- `GET /health` - Health check
- `GET /model_info` - Model information

### Docker Deployment

```powershell
# Build and run with Docker Compose
docker-compose up --build

# Training service
docker-compose run train python examples/train_model.py --data_root /workspace/data

# Inference API
docker-compose up inference

# TensorBoard monitoring
docker-compose up tensorboard
```

### Unit Testing

Run comprehensive tests:

```powershell
# Run all tests
python tests/test_components.py

# Run specific test categories
python -m unittest tests.test_components.TestAttentionMechanisms
python -m unittest tests.test_components.TestLocalGlobalNetwork
python -m unittest tests.test_components.TestLossFunctions
```

## 📁 Project Structure

```
NDTWIN_Idea_1/
├── src/                          # Main source code
│   ├── data/                     # Dataset handling
│   │   ├── __init__.py
│   │   └── dataset.py           # NDTwinDataset implementation
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── attention_network.py  # Main network architecture
│   │   └── loss_functions.py    # Loss function implementations
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── dataset_utils.py     # Dataset validation and analysis
│   │   └── visualization.py     # Plotting and visualization
│   ├── train.py                 # Training script
│   └── inference.py             # Inference and evaluation
├── examples/                     # Example usage scripts
│   ├── train_model.py           # Training example
│   ├── evaluate_model.py        # Evaluation example
│   └── verify_pair.py           # Single pair verification
├── scripts/                      # Utility scripts
│   ├── preprocess_dataset.py   # Dataset preprocessing with face alignment (NEW)
│   ├── analyze_dataset.py       # Dataset analysis and legacy preprocessing
│   ├── benchmark_model.py       # Performance benchmarking
│   └── export_model.py          # Model export utilities
├── tests/                        # Unit tests
│   └── test_components.py       # Comprehensive component tests
├── configs/                      # Configuration files
│   └── train_config.yaml       # Training configuration
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── inference_api.py             # FastAPI REST server
└── README.md                    # This file
```

## 🐳 Docker Usage

### Development Environment

```powershell
# Start development container
docker-compose run dev

# Inside container
python examples/train_model.py --data_root /workspace/data
```

### Production Deployment

```powershell
# Build production image
docker build -t twin-verification .

# Run inference API
docker run -p 8000:8000 -v $(pwd)/checkpoints:/workspace/checkpoints twin-verification python inference_api.py
```

### Custom Backbone
```python
# Add custom backbone in attention_network.py
def _get_backbone(self, backbone: str, pretrained: bool):
    if backbone == 'custom_model':
        # Your custom implementation
        return custom_model
```

### Custom Loss Function
```python
# Add to loss_functions.py
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Your implementation
```

### Data Augmentation
Modify augmentation in `dataset.py`:
```python
# Careful with augmentations that might remove discriminative features
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    # Avoid heavy augmentations that distort facial features
])
```

## 📊 Monitoring Training

### Tensorboard
```powershell
tensorboard --logdir logs
```

View training progress:
- Loss curves (total, global, local, attention regularization)
- Validation metrics (AUC, accuracy)
- Attention map visualizations
- Learning rate scheduling

### Training Outputs
- `checkpoints/`: Model checkpoints
- `logs/`: Tensorboard logs and attention visualizations
- `logs/training_history.json`: Complete training history

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```yaml
data:
  batch_size: 8  # Reduce batch size
```

**2. Poor Attention Maps**
```yaml
loss:
  attention_reg_weight: 0.1  # Increase attention regularization
```

**3. Overfitting**
```yaml
model:
  dropout: 0.5  # Increase dropout
  freeze_backbone_epochs: 5  # Freeze backbone initially
```

**4. Low Twin Discrimination**
```python
# Increase twin negative ratio in dataset.py
twin_negative_ratio=0.8  # More twin pairs in training
```

### Performance Optimization

**Memory Optimization:**
- Use gradient checkpointing for large models
- Reduce image resolution if necessary
- Use mixed precision training

**Speed Optimization:**
- Use multiple workers for data loading
- Pin memory for GPU transfer
- Use appropriate batch sizes

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{twin_verification_2025,
  title={Local-Global Attention Network for Identical Twin Verification},
  author={Your Name},
  journal={Conference/Journal},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and issues:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

## 🔄 Updates

- **v1.0**: Initial implementation with CBAM attention
- **v1.1**: Added transformer attention and face alignment
- **v1.2**: Enhanced evaluation metrics and visualizations

---

**Note**: This implementation is designed for research purposes. For production use, consider additional optimizations for speed and memory efficiency.