import argparse
import asyncio
import logging.config
import warnings
import torch.nn as nn
from pathlib import Path
from typing import Dict, Type
from trainer import Trainer
import modules

# Configure logging system for training pipeline monitoring
configuration = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        # Application-level message handling
        "main": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "main.log",
            "mode": "w",
        },
        # Dataset loading operation logs
        "loader": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "loader.log",
            "mode": "w",
        },
        # Model architecture operation logs
        "modules": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "modules.log",
            "mode": "w",
        },
        # Training process monitoring
        "trainer": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "trainer.log",
            "mode": "w",
        },
    },
    "loggers": {
        # Main application logger
        "main": {"handlers": ["main"], "level": "INFO", "propagate": False},
        # Data loading logger
        "loader": {"handlers": ["loader"], "level": "INFO", "propagate": False},
        # Module operation logger
        "modules": {"handlers": ["modules"], "level": "INFO", "propagate": False},
        # Training process logger
        "trainer": {"handlers": ["trainer"], "level": "INFO", "propagate": False},
    },
}

# Available object detection model architectures
modules: Dict[str, Type[nn.Module]] = {
    "retinanet": modules.RetinaNet,
    "ssd": modules.SSD,
    "fasterrcnn": modules.FasterRCNN,
    "mobilefrcnn": modules.MobileFasterRCNN,
    "ssdlite": modules.SSDLite,
    "fcos": modules.FCOS,
}

if __name__ == "__main__":
    # Initialize logging system
    logging.config.dictConfig(configuration)
    logger = logging.getLogger("main")
    logger.info("Initializing object detection training pipeline")

    # Configure command line argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train object detection models using PyTorch framework"
    )

    # Model architecture selection
    parser.add_argument(
        "-m",
        "--modules",
        type=str,
        required=True,
        metavar="...",
        help=f"Choose detection model architecture: {', '.join(modules.keys())}",
    )

    # Pre-trained weight initialization
    parser.add_argument(
        "-w",
        "--weights",
        type=bool,
        default=True,
        metavar="...",
        help="Initialize with pre-trained weights (recommended)",
    )

    # Dataset configuration
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=True,
        metavar="...",
        help="Number of object classes to detect",
    )

    parser.add_argument(
        "-ch",
        "--channels",
        type=int,
        default=3,
        metavar="...",
        help="Input image channels (3 for RGB, 1 for grayscale)",
    )

    # Dataset path configuration
    parser.add_argument(
        "-tp",
        "--training-path",
        type=Path,
        required=True,
        metavar="...",
        help="Directory containing training dataset",
    )

    parser.add_argument(
        "-vp",
        "--validation-path",
        type=Path,
        required=True,
        metavar="...",
        help="Directory containing validation dataset",
    )

    parser.add_argument(
        "-tep",
        "--testing-path",
        type=Path,
        default=None,
        metavar="...",
        help="Directory containing test dataset (optional)",
    )

    # Model checkpoint loading
    parser.add_argument(
        "-wp",
        "--weights-path",
        type=Path,
        default=None,
        metavar="...",
        help="Path to existing model checkpoint (optional)",
    )

    # Image processing parameters
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        nargs=2,
        default=(800, 800),
        metavar="...",
        help="Input image dimensions as width height",
    )

    # Training hyperparameters
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=25,
        metavar="...",
        help="Number of training epochs",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="...",
        help="Training batch size",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0001,
        metavar="...",
        help="Optimizer learning rate",
    )

    parser.add_argument(
        "-wk",
        "--workers",
        type=int,
        default=4,
        metavar="...",
        help="Number of data loading worker processes",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        metavar="...",
        help="Random seed for reproducible training",
    )

    # Advanced optimization parameters
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=None,
        metavar="...",
        help="L2 regularization weight decay factor",
    )

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=None,
        metavar="...",
        help="Learning rate scheduler decay factor",
    )

    parser.add_argument(
        "-mm",
        "--momentum",
        type=float,
        default=None,
        metavar="...",
        help="SGD optimizer momentum parameter",
    )

    # Object detection specific configuration
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        default=0.5,
        metavar="...",
        help="IoU threshold for detection filtering",
    )

    # Output configuration
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="...",
        help="Output path for saving trained model",
    )

    # Parse command line arguments
    args: argparse.Namespace = parser.parse_args()
    logger.info("Command line arguments processed successfully")

    # Validate model architecture selection
    if args.modules not in modules:
        types: str = ", ".join(modules.keys())
        logger.warning(
            f"Invalid model '{args.modules}' specified. Available options: {types}"
        )
        warnings.warn(
            f"Model '{args.modules}' not available. Valid options: {types}",
            UserWarning,
        )
    else:
        logger.info(f"Selected detection model: {args.modules}")

    # Initialize model instance
    logger.info("Creating model architecture")
    try:
        module: nn.Module = modules[args.modules](
            classes=args.classes,
            channels=args.channels,
            weights=args.weights,
        )
        logger.info(
            f"Model initialized successfully: {args.modules} with {args.classes} detection classes"
        )
    except Exception as error:
        logger.error(f"Model initialization failed: {str(error)}")
        raise Exception(f"Unable to create model: {str(error)}", RuntimeWarning)

    # Configure training pipeline
    logger.info("Setting up training controller")
    try:
        trainer: Trainer = Trainer(
            module=module,
            training_path=args.training_path,
            validation_path=args.validation_path,
            testing_path=args.testing_path,
            weights_path=args.weights_path,
            dimensions=args.dimensions,
            epochs=args.epochs,
            batch=args.batch_size,
            lr=args.learning_rate,
            decay=args.weight_decay,
            gamma=args.gamma,
            momentum=args.momentum,
            workers=args.workers,
            seed=args.seed,
        )
        logger.info("Training pipeline configured and ready")
    except Exception as error:
        logger.error(f"Trainer initialization error: {str(error)}")
        raise Exception(f"Training setup failed: {str(error)}", RuntimeWarning)

    # Execute training phase
    logger.info("Beginning model training process")
    try:
        asyncio.run(trainer.train())
        logger.info("Training phase completed successfully")
    except Exception as error:
        logger.error(f"Training process failed: {str(error)}")
        warnings.warn(f"Training interrupted: {str(error)}", RuntimeWarning)

    # Run evaluation on test set if available
    if args.testing_path is not None:
        logger.info("Evaluating trained model on test dataset")
        try:
            asyncio.run(trainer.test())
            logger.info("Model evaluation completed")
        except Exception as error:
            logger.error(f"Testing phase failed: {str(error)}")
            warnings.warn(f"Evaluation error: {str(error)}", RuntimeWarning)
    else:
        logger.info("Test dataset not provided - skipping evaluation phase")

    # Save trained model weights
    if args.output:
        logger.info(f"Saving trained model to {args.output}")
        try:
            trainer.save(filepath=args.output)
            logger.info("Model weights saved successfully")
        except Exception as error:
            logger.error(f"Model saving failed: {str(error)}")
            warnings.warn(f"Unable to save model: {str(error)}", RuntimeWarning)
    else:
        logger.warning("Output path not specified - trained model will not be saved")

    logger.info("Object detection training pipeline execution completed")