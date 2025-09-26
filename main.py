import argparse
import asyncio
import logging.config
import warnings
import torch.nn as nn
from pathlib import Path
from typing import Dict, Type
from trainer import Trainer
import modules

# Set up logging so we can track what's happening during training
configuration = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        # This handles the main app messages
        "main": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "main.log",
            "mode": "w",
        },
        # This tracks data loading stuff
        "loader": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "loader.log",
            "mode": "w",
        },
        # This logs what the modules are doing
        "modules": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "modules.log",
            "mode": "w",
        },
        # This keeps track of the actual training process
        "trainer": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "trainer.log",
            "mode": "w",
        },
    },
    "loggers": {
        # Main app logger
        "main": {"handlers": ["main"], "level": "INFO", "propagate": False},
        # Data loading logger
        "loader": {"handlers": ["loader"], "level": "INFO", "propagate": False},
        # Module logger
        "modules": {"handlers": ["modules"], "level": "INFO", "propagate": False},
        # Training logger
        "trainer": {"handlers": ["trainer"], "level": "INFO", "propagate": False},
    },
}

# All the different object detection models we can use
modules: Dict[str, Type[nn.Module]] = {
    "retinanet": modules.RetinaNet,
    "ssd": modules.SSD,
    "fasterrcnn": modules.FasterRCNN,
    "mobilefrcnn": modules.MobileFasterRCNN,
    "ssdlite": modules.SSDLite,
    "fcos": modules.FCOS,
}

if __name__ == "__main__":
    # Get logging up and running first
    logging.config.dictConfig(configuration)
    logger = logging.getLogger("main")
    logger.info("Firing up the object detection trainer...")

    # Set up all the command line options
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train different object detection models with PyTorch"
    )

    # Which model do you want to use?
    parser.add_argument(
        "-m",
        "--modules",
        type=str,
        required=True,
        metavar="...",
        help=f"Pick your model architecture. Options: {', '.join(modules.keys())}",
    )

    # Should we use pre-trained weights?
    parser.add_argument(
        "-w",
        "--weights",
        type=bool,
        default=True,
        metavar="...",
        help="Use pre-trained weights? (Default: True)",
    )

    # How many classes are we detecting?
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=True,
        metavar="...",
        help="Number of different object types to detect",
    )

    # Image channels (usually 3 for color images)
    parser.add_argument(
        "-ch",
        "--channels",
        type=int,
        default=3,
        metavar="...",
        help="Input channels (3 for RGB, 1 for grayscale)",
    )

    # Where's our training data?
    parser.add_argument(
        "-tp",
        "--training-path",
        type=Path,
        required=True,
        metavar="...",
        help="Path to your training dataset folder",
    )

    parser.add_argument(
        "-vp",
        "--validation-path",
        type=Path,
        required=True,
        metavar="...",
        help="Path to your validation dataset folder",
    )

    parser.add_argument(
        "-tep",
        "--testing-path",
        type=Path,
        default=None,
        metavar="...",
        help="Path to test dataset (optional)",
    )

    # Got existing weights to load?
    parser.add_argument(
        "-wp",
        "--weights-path",
        type=Path,
        default=None,
        metavar="...",
        help="Path to existing model weights (optional)",
    )

    # What size images are we working with?
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        nargs=2,
        default=(800, 800),
        metavar="...",
        help="Image size as width height",
    )

    # Training settings
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=25,
        metavar="...",
        help="How many times to go through the dataset",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="...",
        help="How many images to process at once",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0001,
        metavar="...",
        help="How fast the model learns",
    )

    parser.add_argument(
        "-wk",
        "--workers",
        type=int,
        default=4,
        metavar="...",
        help="Number of CPU workers for data loading",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        metavar="...",
        help="Random seed for consistent results",
    )

    # Advanced training tweaks
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=None,
        metavar="...",
        help="Weight decay to prevent overfitting",
    )

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=None,
        metavar="...",
        help="Learning rate scheduler gamma",
    )

    parser.add_argument(
        "-mm",
        "--momentum",
        type=float,
        default=None,
        metavar="...",
        help="Optimizer momentum",
    )

    # Object detection specific stuff
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        default=0.5,
        metavar="...",
        help="IoU threshold for filtering detections",
    )

    # Where to save the final model
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="...",
        help="Where to save your trained model",
    )

    # Get all the arguments the user provided
    args: argparse.Namespace = parser.parse_args()
    logger.info("Got all the command line arguments")

    # Make sure they picked a valid model
    if args.modules not in modules:
        types: str = ", ".join(modules.keys())
        logger.warning(
            f"'{args.modules}' isn't a valid model. Try one of these: {types}"
        )
        warnings.warn(
            f"'{args.modules}' isn't available. Valid options: {types}",
            UserWarning,
        )
    else:
        logger.info(f"Using {args.modules} model")

    # Create the model
    logger.info("Setting up the model...")
    try:
        module: nn.Module = modules[args.modules](
            classes=args.classes,
            channels=args.channels,
            weights=args.weights,
        )
        logger.info(
            f"Model ready! {args.modules} configured for {args.classes} classes"
        )
    except Exception as error:
        logger.error(f"Couldn't create the model: {str(error)}")
        raise Exception(f"Model setup failed: {str(error)}", RuntimeWarning)

    # Set up the trainer
    logger.info("Getting the trainer ready...")
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
        logger.info("Trainer is locked and loaded")
    except Exception as error:
        logger.error(f"Trainer setup failed: {str(error)}")
        raise Exception(f"Couldn't initialize trainer: {str(error)}", RuntimeWarning)

    # Let's start training!
    logger.info("Starting the training process...")
    try:
        asyncio.run(trainer.train())
        logger.info("Training finished successfully!")
    except Exception as error:
        logger.error(f"Training crashed: {str(error)}")
        warnings.warn(f"Training didn't work: {str(error)}", RuntimeWarning)

    # Test the model if we have test data
    if args.testing_path is not None:
        logger.info("Running tests on the trained model...")
        try:
            asyncio.run(trainer.test())
            logger.info("Testing complete!")
        except Exception as error:
            logger.error(f"Testing failed: {str(error)}")
            warnings.warn(f"Testing didn't work: {str(error)}", RuntimeWarning)
    else:
        logger.info("No test data provided, skipping tests")

    # Save the model if they want us to
    if args.output:
        logger.info(f"Saving the trained model to {args.output}...")
        try:
            trainer.save(filepath=args.output)
            logger.info("Model saved!")
        except Exception as error:
            logger.error(f"Couldn't save the model: {str(error)}")
            warnings.warn(f"Model saving failed: {str(error)}", RuntimeWarning)
    else:
        logger.warning("No save path provided - your model won't be saved")

    logger.info("All done! Check your logs for details.")