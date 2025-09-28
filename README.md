# Object Detection Training Framework

A PyTorch framework for training object detection models with support for 6 popular architectures and automatic optimization configuration based on model-specific best practices.

## Features

- **Six detection architectures** including RetinaNet, Faster R-CNN, SSD, FCOS, and mobile variants
- **Automatic optimization** with architecture-specific learning rates and schedulers
- **Progress tracking** with real-time loss monitoring and training metrics
- **Multi-GPU support** for faster training on multi-card systems
- **Detailed logging** with separate files for different components
- **Overfitting detection** with automatic checkpoint creation
- **Deterministic training** with configurable random seeds for reproducibility

## Available Models

| Model | Architecture Type | Speed | Accuracy | Memory Usage | Input Size |
|-------|-------------------|-------|----------|--------------|------------|
| `retinanet` | Single-stage with focal loss | Medium | High | Medium | Variable |
| `fasterrcnn` | Two-stage with RPN | Slow | Very High | High | Variable |
| `mobilefrcnn` | Mobile two-stage | Fast | High | Low | Variable |
| `ssd` | Single-stage multi-box | Fast | Medium | Medium | 300x300 |
| `ssdlite` | Mobile single-stage | Very Fast | Medium | Low | 320x320 |
| `fcos` | Anchor-free single-stage | Medium | High | Medium | Variable |

## Dataset Structure

Organize your dataset with images and COCO-format annotations:

```
dataset/
├── train/
│   ├── images/
│   │   ├── image001.jpg
│   │   └── ...
│   └── annotations.json
├── val/
│   ├── images/
│   │   ├── image001.jpg
│   │   └── ...
│   └── annotations.json
└── test/
    ├── images/
    │   ├── image001.jpg
    │   └── ...
    └── annotations.json
```

## Installation

```bash
pip install torch torchvision numpy pillow pathlib asyncio
```

## Basic Usage

```bash
python main.py --modules retinanet --classes 10 --training-path ./dataset/train --validation-path ./dataset/val --testing-path ./dataset/test
```

## Command Line Options

### Required Parameters
- `-m, --modules` - Detection model architecture to use
- `-c, --classes` - Number of object classes in your dataset
- `-tp, --training-path` - Training dataset directory
- `-vp, --validation-path` - Validation dataset directory

### Optional Parameters
- `-tep, --testing-path` - Test dataset directory
- `-w, --weights` - Use pre-trained weights (default: True)
- `-ch, --channels` - Input channels: 3 for RGB, 1 for grayscale (default: 3)
- `-wp, --weights-path` - Path to existing model checkpoint
- `-d, --dimensions` - Image size as width height (default: 800 800)
- `-e, --epochs` - Number of training epochs (default: 25)
- `-b, --batch-size` - Training batch size (default: 64)
- `-lr, --learning-rate` - Override default learning rate (default: 0.0001)
- `-wk, --workers` - Data loading workers (default: 4)
- `-s, --seed` - Random seed for reproducible results
- `-wd, --weight-decay` - L2 regularization factor
- `-g, --gamma` - Learning rate scheduler decay
- `-mm, --momentum` - SGD momentum parameter
- `-th, --threshold` - IoU threshold for detection filtering (default: 0.5)
- `-o, --output` - Output path for trained model

## Model-Specific Optimizations

The framework automatically configures optimal training settings for each architecture:

**RetinaNet** uses SGD with milestone scheduling at epochs 8 and 11
**Faster R-CNN** uses conservative SGD with step scheduling every 3 epochs
**SSD variants** use Adam with step scheduling every 5 epochs
**FCOS** uses SGD with cosine annealing for smooth decay

Learning rates, weight decay, and momentum are set based on published research for each model type.

## Output Files

Training generates several log files:
- `main.log` - Overall application status and errors
- `loader.log` - Dataset loading operations and issues
- `modules.log` - Model initialization and configuration
- `trainer.log` - Training progress, loss values, and metrics

Model weights are saved to the path specified with `--output` (defaults to current directory).

## Performance Tips

**Memory issues?**
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Use smaller images: `--dimensions 416 416`
- Switch to mobile models: `ssdlite` or `mobilefrcnn`

**Training too slow?**
- Increase workers: `--workers 8`
- Use faster models: `ssdlite` or `ssd`
- Try smaller image sizes or fewer epochs

**Poor detection accuracy?**
- Use larger models: `fasterrcnn` or `retinanet`
- Increase image dimensions: `--dimensions 1024 1024`
- Train for more epochs: `--epochs 100`
- Check annotation quality and class balance

## Troubleshooting

**"Model type not supported" errors**
- Double-check the model name spelling against the available models list
- Use `--modules` followed by one of the exact names from the table above

**SSD input size errors**
- SSD requires exactly 300x300 pixel images
- SSDLite requires exactly 320x320 pixel images
- Other models can handle variable input sizes

**Out of memory errors**
- Start with `--batch-size 2` and work your way up
- Detection models are memory-intensive due to multi-scale processing
- Consider using `ssdlite` or `mobilefrcnn` for lower memory usage

**Poor training convergence**
- Check that bounding box annotations are accurate
- Verify class labels match your `--classes` parameter
- Ensure sufficient training data (at least 500 annotations per class recommended)
- Monitor `trainer.log` for loss progression

**Multi-class detection issues**
- Make sure annotation format matches COCO structure
- Verify class IDs are consecutive integers starting from 0
- Check for class imbalance in your dataset

## Model Recommendations

- **First experiments**: `ssd` or `retinanet` - good balance of speed and accuracy
- **Production systems**: `fasterrcnn` or `retinanet` - highest accuracy
- **Mobile/edge deployment**: `ssdlite` or `mobilefrcnn`
- **Real-time applications**: `ssd` or `ssdlite`
- **Maximum accuracy**: `fasterrcnn` with large input dimensions
- **Limited memory**: `ssdlite` or `mobilefrcnn`

The framework handles architecture-specific optimization automatically, so you can focus on preparing quality annotations and choosing the right model for your detection requirements.
