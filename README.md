# Object Detection Training Framework

A PyTorch framework for training object detection models. This repo gives you a single interface to work with six different detection architectures, handling the optimization and training specifics for each one.

## Supported Models

- **RetinaNet** - Single-stage detector that uses focal loss for class imbalance
- **Faster R-CNN** - Two-stage detector with region proposals
- **SSD** (Single Shot Detector) - Fast single-stage with multi-scale features
- **FCOS** (Fully Convolutional One-Stage) - Anchor-free detection
- **MobileFasterRCNN** - Lightweight Faster R-CNN for mobile
- **SSDLite** - Efficient SSD with depthwise separable convolutions

## What You Get

- Single pipeline for all six architectures
- Transfer learning with pre-trained weights
- Separate log files for each component (trust me, this helps)
- Configure everything from the command line
- Auto-saves checkpoints and runs validation
- Handles any image size and custom datasets

## Setup

Install the dependencies:

```bash
pip install torch torchvision numpy
```

## How to Use

### Quick Start

```bash
python main.py \
  --modules retinanet \
  --classes 80 \
  --training-path ./data/train \
  --validation-path ./data/val \
  --output ./models/trained_model.pt
```

### Command Line Arguments

#### Model Setup
- `--modules` - Which architecture to use (required)
- `--classes` - Number of object classes (required)
- `--channels` - Input channels, typically 3 for RGB
- `--weights` - Use pre-trained weights (enabled by default)

#### Data Paths
- `--training-path` - Where your training data lives (required)
- `--validation-path` - Where your validation data lives (required)
- `--testing-path` - Where your test data lives
- `--weights-path` - Path to checkpoint if resuming training

#### Training Parameters
- `--epochs` - How long to train (default: 25)
- `--batch-size` - Samples per batch (default: 64)
- `--learning-rate` - Initial learning rate (default: 0.0001)
- `--dimensions` - Image size as width height (default: 800 800)
- `--workers` - Number of data loading threads (default: 4)
- `--seed` - Random seed for reproducibility

#### Advanced Options
- `--weight-decay` - L2 regularization strength
- `--gamma` - Learning rate decay factor
- `--momentum` - Momentum for SGD
- `--threshold` - IoU threshold for detections (default: 0.5)

## Logging

Logs get written to four separate files:

- `main.log` - High-level program flow
- `loader.log` - Data loading operations
- `modules.log` - Model-specific operations
- `trainer.log` - Training progress and metrics

## Structure

```
.
├── main.py           # Entry point
├── trainer.py        # Training logic
├── modules.py        # All model implementations
└── logs/            # Where logs end up
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- NumPy

## Notes

Pre-trained weights load automatically when you have them. They're especially useful when you're working with smaller datasets.

The defaults work well enough to get started, but you'll want to tune them based on your specific dataset and hardware.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
