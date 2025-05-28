# Europalette Conv

A deep learning project for pallet pose estimation using 1D convolutional neural networks on simulated LiDAR data. The system trains a CNN to predict the position and orientation of europallets from 2D LiDAR scans.

## Overview

This project simulates a 2D LiDAR sensor observing the middle of europallets (looks like squares arranged in a 3×3 grid pattern). The neural network learns to estimate:
- **Position**: X,Y coordinates of the pallet center (in mm)
- **Orientation**: Rotation angle in the range [0, π] radians (accounting for pallet symmetry)

The approach uses raycasting to generate synthetic LiDAR data and trains a 1D CNN to map sensor readings to pallet poses.

## Key Features

- **Synthetic Data Generation**: Raycast-based simulation of 2D LiDAR sensing europallets
- **Custom Loss Function**: Handles angular periodicity for pallet orientation estimation
- **Data Augmentation**: Gaussian noise injection during training for robustness
- **Visualization Tools**: Comprehensive plotting for data inspection and model evaluation

## Project Structure

```
europalette-conv/
├── generate_data.py    # Synthetic dataset generation using raycasting
├── train.py           # CNN training script with custom loss functions
├── eval.py            # Model evaluation and visualization
├── raycast.py         # Core raycasting algorithms (optimized)
├── check_data.py      # Dataset inspection utilities
├── dataset.pkl        # Generated training dataset (57MB)
├── pallet_pose_cnn.pth # Trained model weights (1.1MB)
└── README.md          # This file
```

## Configuration

Key parameters (consistent across scripts):
- **LiDAR Range**: 30m maximum detection range
- **Field of View**: 270° sensor coverage
- **Ray Count**: 200 rays per scan
- **Pallet Distance**: 0.5-1.0m from sensor
- **View Randomness**: ±140° viewing angle variation

## Usage

### 1. Generate Training Data
```bash
python generate_data.py
```
Creates `dataset.pkl` with synthetic LiDAR scans and ground truth poses.

### 2. Train the Model
```bash
python train.py
```
Trains the CNN and saves weights to `pallet_pose_cnn.pth`. The script:
- Loads existing model weights if available (continues training)
- Uses Adam optimizer with learning rate 1e-4
- Applies Gaussian noise augmentation
- Balances position and orientation losses

### 3. Evaluate Performance
```bash
python eval.py
```
Loads the trained model and evaluates on test data with visualizations.

### 4. Inspect Dataset
```bash
python check_data.py
```
Visualizes random samples from the dataset for quality assessment.

## Model Architecture

**PalletPoseCNN**: 1D Convolutional Neural Network
- Input: 200-dimensional LiDAR distance measurements
- Feature extraction: 5 conv1d layers with ReLU activation
- Dilated convolutions for larger receptive fields
- Adaptive global average pooling
- Fully connected head outputting [x, y, θ]
- Output activation: Sigmoid on orientation (scaled to [0, π])

## Loss Function

The model uses a composite loss combining:
- **Position Loss**: Standard MSE for X,Y coordinates
- **Orientation Loss**: Custom angular distance metric handling [0, π] periodicity
- **Loss Balancing**: Weighted combination ensuring comparable gradients

## Dependencies

- `numpy` - Numerical computations
- `torch` - PyTorch for deep learning
- `matplotlib` - Visualization and plotting
- `tqdm` - Progress bars
- `pickle` - Data serialization

## Performance

The trained model achieves:
- Position accuracy: ~50-100mm typical error
- Orientation accuracy: ~5-15° typical error

Performance metrics are printed during training and evaluation phases.

## Technical Details

### Coordinate System
- Origin at LiDAR sensor position
- X-axis aligned with primary viewing direction
- Angles measured counterclockwise from +X axis

### Pallet Representation
- Standard europalette dimensions (1200×800mm)
- 3×3 grid arrangement of squares in workspace
- Symmetric orientation handling (θ ≡ θ+π)

### Raycasting Engine
- Ultra-optimized vectorized implementation
- Batch processing for multiple rays/squares
- Axis-aligned square intersection algorithms 