# Cardiomegaly Detection Using Deep Learning

This repository contains the implementation of an intelligent cardiomegaly detection system based on deep learning techniques. The project focuses on developing an enhanced convolutional neural network with improved attention mechanisms for accurate identification of cardiac enlargement in chest X-ray images.

## Overview

Cardiomegaly detection is a critical task in cardiovascular diagnosis that traditionally relies on manual interpretation by medical professionals. This project introduces a deep learning-based approach that incorporates several advanced techniques to improve detection accuracy and provide interpretable results for clinical applications.

## Key Features

### Architecture Innovations
- **Enhanced CBAM Attention Mechanism**: Improved channel attention using SiLU activation and learnable Tanh scaling, combined with spatial attention that fuses Scharr edge operators and dilated convolutions
- **Multi-Scale Depthwise Separable Convolutions**: Efficient feature extraction across different spatial scales using 3×3 and 5×5 kernels
- **Advanced Classifier Head**: Incorporates Gated GELU Linear Units (GeGLU) and RMSNorm for improved high-level feature representation

### Training Optimizations
- **MARS-AdamW Optimizer**: Implementation of the variance reduction framework that significantly reduces training time while maintaining performance
- **Focal Loss**: Addresses class imbalance issues common in medical imaging datasets
- **Cosine Annealing with Warm-up**: Sophisticated learning rate scheduling for stable convergence

### Interpretability Tools
- **Grad-CAM Visualization**: Generates heatmaps showing model attention regions
- **UMAP Feature Visualization**: Provides 2D and 3D projections of learned feature representations
- **Comprehensive Evaluation Metrics**: ROC curves, PR curves, confusion matrices, and detailed performance reports

## Requirements

```
torch>=2.0.0
torchvision
numpy
matplotlib
seaborn
scikit-learn
umap-learn
tqdm
Pillow
```

## Dataset Structure

Organize your data in the following structure:
```
train/
├── true/     # Cardiomegaly positive cases
└── false/    # Normal cases
test/
├── true/     # Test positive cases
└── false/    # Test negative cases
```

## Usage

### Basic Training
```bash
python cnnxr.py --data_dir_train ./train --data_dir_test ./test --epochs 50
```

### Advanced Configuration
```bash
python cnnxr.py \
    --batch_size 64 \
    --epochs 50 \
    --lr 5e-4 \
    --wd 1e-3 \
    --focal_alpha 0.35 \
    --focal_gamma 2.0 \
    --mars_gamma 0.01
```

### Using Classic CBAM or PyTorch AdamW
```bash
# Use classic CBAM instead of enhanced version
python cnnxr.py --use_classic_cbam

# Use PyTorch AdamW instead of MARS-AdamW
python cnnxr.py --use_pytorch_adamw
```

## Model Architecture

The enhanced CNN architecture consists of five main stages with progressively increasing channel dimensions (32→64→128→256→512→1024). Each stage incorporates residual connections, multi-scale feature extraction, and attention mechanisms. The final classification head uses advanced components including GeGLU units and RMSNorm for robust decision making.

## Results and Visualization

The system generates comprehensive evaluation outputs including:

- **Performance Metrics**: Precision, recall, F1-score, ROC-AUC, and average precision
- **Visual Analysis**: Grad-CAM heatmaps highlighting model attention regions
- **Feature Analysis**: UMAP projections showing learned feature space organization
- **Training Curves**: Loss, accuracy, and learning rate progression throughout training

All results are automatically saved to the `results/` directory with timestamped filenames.

## Performance Highlights

The enhanced model demonstrates strong performance on cardiomegaly detection tasks:
- High accuracy in distinguishing between normal and enlarged cardiac silhouettes
- Clinically relevant attention patterns as shown by Grad-CAM visualizations
- Clear feature space separation as demonstrated by UMAP analysis
- Significant training time reduction when using MARS-AdamW optimization

## Technical Implementation

The implementation leverages several advanced deep learning techniques:

- **Automatic Mixed Precision**: Reduces memory usage and accelerates training on CUDA-enabled devices
- **Gradient Scaling**: Maintains numerical stability during mixed precision training
- **Hook-based Visualization**: Efficient computation of attention maps without architectural modifications
- **Modular Design**: Easy to extend and modify for different attention mechanisms or optimizers

## File Structure

```
├── cnnxr.py              # Main implementation file
├── results/              # Generated outputs (created automatically)
│   ├── model weights
│   ├── evaluation reports
│   ├── visualization plots
│   └── training logs
└── README.md
```

## Citation

If you use this implementation in your research, please consider citing the associated work that introduces these methodological improvements for medical image analysis.

## License

This project is released under the MIT License for academic and research purposes.

## Contributing

Contributions are welcome, particularly in areas of architecture improvements, additional visualization techniques, or extensions to other medical imaging tasks.
