import os
import random
import asyncio
import modules
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim
import torchvision.ops as ops
from torch.utils.data import DataLoader
from torchvision import transforms
from loader import DatasetLoader, collate_fn
from pathlib import Path
import logging
import warnings
from bar import Bar
from typing import Dict, List, Literal, Optional, Tuple

# Get our logger for tracking training progress
logger = logging.getLogger("trainer")


class Trainer:
    """
    The main training class that handles everything from loading data to saving models.

    This is basically your training command center. It loads your data, sets up the model,
    runs the training loops, watches for overfitting, and saves your results. Works with
    all the popular object detection models like RetinaNet, Faster R-CNN, SSD, and FCOS.

    It's pretty smart about picking the right optimizer settings for each model type,
    and it'll even use multiple GPUs if you have them. Plus it has built-in progress
    bars and logging so you can see what's happening.

    Args:
        module: Your PyTorch model (like RetinaNet, SSD, etc.)
        training_path: Where your training images and annotations are
        validation_path: Where your validation data is
        testing_path: Where your test data is
        weights_path: Path to pretrained weights (optional)
        dimensions: Image size as (height, width)
        epochs: How many times to go through your dataset
        batch: How many images to process at once
        lr: Learning rate (how fast the model learns)
        decay: Weight decay to prevent overfitting
        gamma: How much to reduce learning rate over time
        momentum: Momentum for SGD optimizer
        workers: Number of CPU threads for data loading
        seed: Random seed for reproducible results
        parallelism: Use multiple GPUs if available
    """

    def __init__(
            self,
            module: nn.Module,
            training_path: Path,
            validation_path: Path,
            testing_path: Path,
            weights_path: Optional[Path] = None,
            dimensions: Tuple[int, int] = None,
            epochs: int = None,
            batch: int = None,
            lr: Optional[float] = None,
            decay: Optional[float] = None,
            gamma: Optional[float] = None,
            momentum: Optional[float] = None,
            workers: Optional[int] = None,
            seed: Optional[int] = None,
            parallelism: Optional[bool] = False,
    ) -> None:
        """
        Set up everything we need for training.

        This does a lot of validation to make sure your settings make sense,
        then configures the model, data loaders, optimizer, and scheduler.
        Each model type gets its own optimal settings based on what works
        best in practice.
        """

        # Make sure we actually got a PyTorch model
        if not isinstance(module, nn.Module):
            logger.error(f"Expected a PyTorch model, but got {type(module)}")
            raise TypeError(f"Expected a PyTorch model, but got {type(module)}")
        logger.info(f"Model looks good: {type(module).__name__}")

        # Check that all our data paths exist
        if not isinstance(training_path, Path) or not training_path.exists():
            logger.error(f"Can't find training data at: {training_path}")
            raise ValueError(f"Can't find training data at: {training_path}")
        logger.info(f"Training data found: {training_path}")

        if not isinstance(validation_path, Path) or not validation_path.exists():
            logger.error(f"Can't find validation data at: {validation_path}")
            raise ValueError(f"Can't find validation data at: {validation_path}")
        logger.info(f"Validation data found: {validation_path}")

        if not isinstance(testing_path, Path) or not testing_path.exists():
            logger.error(f"Can't find test data at: {testing_path}")
            raise ValueError(f"Can't find test data at: {testing_path}")
        logger.info(f"Test data found: {testing_path}")

        # Make sure dimensions are a proper (height, width) tuple
        if not isinstance(dimensions, tuple) or len(dimensions) != 2:
            logger.error(f"Dimensions should be (height, width), got: {dimensions}")
            raise ValueError(f"Dimensions should be (height, width), got: {dimensions}")
        logger.info(f"Image size set to: {dimensions}")

        # Validate training parameters
        if not isinstance(epochs, int) or epochs <= 0:
            logger.error(f"Epochs must be a positive number, got: {epochs}")
            raise ValueError(f"Epochs must be a positive number, got: {epochs}")
        logger.info(f"Training for {epochs} epochs")

        if not isinstance(batch, int) or batch <= 0:
            logger.error(f"Batch size must be positive, got: {batch}")
            raise ValueError(f"Batch size must be positive, got: {batch}")
        logger.info(f"Batch size: {batch}")

        # Check pretrained weights if provided
        if weights_path and (not isinstance(weights_path, Path) or not weights_path.exists()):
            logger.warning(f"Can't find weights file: {weights_path}")
            warnings.warn(f"Can't find weights file: {weights_path}")
        elif weights_path:
            logger.info(f"Will load weights from: {weights_path}")

        # Validate optional parameters
        if momentum is not None and not isinstance(momentum, (float, int)):
            logger.warning(f"Momentum should be a number, got: {type(momentum)}")
            warnings.warn(f"Momentum should be a number, got: {type(momentum)}")
        elif momentum is not None:
            logger.info(f"Momentum set to: {momentum}")

        if parallelism is not None and not isinstance(parallelism, bool):
            logger.warning(f"Parallelism should be True/False, got: {type(parallelism)}")
            warnings.warn(f"Parallelism should be True/False, got: {type(parallelism)}")
        else:
            logger.info(f"Multi-GPU training: {parallelism}")

        # Set up reproducible training if seed is provided
        if seed is not None:
            self.seed(seed=seed)
            logger.info(f"Random seed: {seed}")

        # Figure out whether to use GPU or CPU
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Move model to GPU/CPU and store settings
        self.module: nn.Module = module.to(self.device)
        self.dimensions: Tuple[int, int] = dimensions
        self.epochs: int = epochs
        self.workers: Optional[int] = workers

        # Initialize weights or load pretrained ones
        if weights_path is None:
            self.initialize_weights(self.module)
            logger.info("Initialized model weights randomly")
        else:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                # Handle different ways models might be saved
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.module.load_state_dict(state_dict)
                logger.info(f"Loaded pretrained weights from {weights_path}")
            except Exception as error:
                logger.warning(f"Couldn't load weights: {error}. Using random weights instead.")
                warnings.warn(f"Couldn't load weights: {error}. Using random weights instead.")
                self.initialize_weights(self.module)

        # Set up loss tracking
        self.cache: Dict[str, List[float]] = {"training": [], "validation": []}
        logger.info("Loss tracking ready")

        # Enable multi-GPU training if we have multiple GPUs
        if parallelism and np.greater(torch.cuda.device_count(), 1):
            self.module = nn.parallel.DistributedDataParallel(self.module)
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        else:
            logger.info("Using single GPU/CPU")

        # Load all our datasets
        try:
            self.training_dataset: DataLoader = self.loader(
                dirpath=training_path, batch=batch, mode="training"
            )
            logger.info("Training data loaded")
        except Exception as error:
            logger.error(f"Failed to load training data: {error}")
            warnings.warn(f"Failed to load training data: {error}")

        try:
            self.validation_dataset: DataLoader = self.loader(
                dirpath=validation_path, batch=batch, mode="validation"
            )
            logger.info("Validation data loaded")
        except Exception as error:
            logger.error(f"Failed to load validation data: {error}")
            warnings.warn(f"Failed to load validation data: {error}")

        try:
            self.testing_dataset: DataLoader = self.loader(
                dirpath=testing_path, batch=batch, mode="testing"
            )
            logger.info("Test data loaded")
        except Exception as error:
            logger.error(f"Failed to load test data: {error}")
            warnings.warn(f"Failed to load test data: {error}")

        # Set up the loss function
        self.criterion: modules.MultiBoxLoss = modules.MultiBoxLoss(module=module)
        logger.info("Loss function ready")

        # Configure optimizer and scheduler based on model type
        # Each model works best with different settings
        if isinstance(module, modules.RetinaNet):
            # RetinaNet likes SGD with specific learning rate scheduling
            decay = decay or 0.0001
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.01

            self.optimizer: torch.optim = torch.optim.SGD(
                params=self.module.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=decay,
            )

            # Reduce learning rate at specific epochs
            self.scheduler: torch.optim.lr_scheduler = (
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=self.optimizer,
                    milestones=[8, 11],  # Drop LR at these epochs
                    gamma=gamma,
                )
            )
            logger.info("RetinaNet optimizer configured")

        elif isinstance(module, modules.FasterRCNN):
            # Faster R-CNN needs lower learning rates
            decay = decay or 0.0005
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.005

            self.optimizer: torch.optim = torch.optim.SGD(
                params=self.module.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=decay,
            )

            # Step down every few epochs
            self.scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=3,
                gamma=gamma,
            )
            logger.info("Faster R-CNN optimizer configured")

        elif isinstance(module, modules.SSD):
            # SSD works well with Adam
            decay = decay or 0.0005
            gamma = gamma or 0.1
            lr = lr or 0.001

            self.optimizer: torch.optim = torch.optim.Adam(
                params=self.module.parameters(), lr=lr, weight_decay=decay
            )

            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=5,
                gamma=gamma,
            )
            logger.info("SSD optimizer configured")

        elif isinstance(module, modules.FCOS):
            # FCOS likes cosine annealing
            decay = decay or 0.0001
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.01

            self.optimizer: torch.optim = torch.optim.SGD(
                params=self.module.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=decay,
            )

            # Smooth learning rate decay
            self.scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=12,
                eta_min=lr * 0.01,
            )
            logger.info("FCOS optimizer configured")

        else:
            logger.error(f"Don't know how to train {type(module)} models")
            raise ValueError(f"Don't know how to train {type(module)} models")

        logger.info(
            f"All set up for {module.__class__.__name__} with decay={decay}, gamma={gamma}, momentum={momentum}")

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        """
        Set up good starting weights for the model.

        Different types of layers need different initialization methods to train well.
        This goes through all the layers and gives them appropriate starting values
        so the model can learn effectively from the beginning.

        Args:
            module: The model to initialize
        """
        for layer in module.modules():
            # Convolutional layers get Kaiming (He) initialization
            # This works great with ReLU activations
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            # Batch norm layers start as identity transforms
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

            # Linear layers get Xavier initialization
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            # LSTM layers get orthogonal initialization to help with gradients
            elif isinstance(layer, nn.LSTM):
                for parameter in layer.parameters():
                    if len(parameter.shape) >= 2:
                        nn.init.orthogonal_(parameter)
                    else:
                        nn.init.normal_(parameter)

            # Same for GRU layers
            elif isinstance(layer, nn.GRU):
                for parameter in layer.parameters():
                    if len(parameter.shape) >= 2:
                        nn.init.orthogonal_(parameter)
                    else:
                        nn.init.normal_(parameter)

    @staticmethod
    def seed(seed: int) -> None:
        """
        Make training reproducible by setting random seeds everywhere.

        This sets the same random seed for Python, NumPy, PyTorch, and CUDA
        so you get the same results every time you run training. Super useful
        for debugging and comparing different approaches.

        Args:
            seed: The random seed number to use
        """
        try:
            # Set environment variable for Python hash randomization
            os.environ["PYTHONHASHSEED"] = str(seed)

            # Set PyTorch seed
            torch.manual_seed(seed=seed)

            # Set Python's random module
            random.seed(a=seed)

            # Set NumPy seed
            np.random.seed(seed=seed)

            # Set CUDA seeds if we're using GPU
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed=seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            logger.info(f"Set random seed to {seed}")
        except Exception as error:
            logger.error(f"Couldn't set random seed: {str(error)}")
            warnings.warn(f"Couldn't set random seed: {str(error)}")

    def loader(
            self,
            dirpath: Path,
            batch: int,
            mode: Literal["training", "validation", "testing"] = "training",
    ) -> Optional[DataLoader]:
        """
        Create a data loader for the specified dataset.

        This sets up the right image transforms (training gets augmentations,
        validation/testing don't), creates the dataset, and wraps it in a
        DataLoader with the right settings for good performance.

        Args:
            dirpath: Where the dataset folder is
            batch: How many images per batch
            mode: Whether this is for training, validation, or testing

        Returns:
            A configured DataLoader ready to use
        """
        # Set up image transforms based on what we're doing
        if mode == "training":
            # Training gets data augmentation to help generalization
            transform = transforms.Compose([
                transforms.Resize(size=self.dimensions),
                transforms.RandomRotation(degrees=10),  # Small rotations
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color variations
                transforms.ToTensor(),
                # Use ImageNet normalization for transfer learning
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            logger.info(f"Training transforms set up with augmentations")
        else:
            # Validation and testing just do basic preprocessing
            transform = transforms.Compose([
                transforms.Resize(size=self.dimensions),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            logger.info(f"Basic transforms set up for {mode}")

        try:
            # Create the dataset
            dataset = DatasetLoader(dirpath=dirpath, transform=transform)

            # Wrap it in a DataLoader
            dataloader = DataLoader(
                dataset=dataset,
                collate_fn=collate_fn,  # Special collation for object detection
                batch_size=batch,
                shuffle=(mode == "training"),  # Only shuffle training data
                num_workers=(4 if self.workers is None else self.workers),
                pin_memory=True,  # Faster GPU transfer
            )
            logger.info(f"{mode.title()} DataLoader created with batch size {batch}")
            return dataloader
        except Exception as error:
            logger.error(f"Failed to create {mode} DataLoader: {str(error)}")
            warnings.warn(f"Failed to create {mode} DataLoader: {str(error)}")
            return None

    async def rehearse(
            self, dataloader: DataLoader, mode: Literal["training", "validation"]
    ) -> np.float64:
        """
        Run one epoch of training or validation.

        This processes all the batches in the dataloader, computes losses,
        and updates the model (if training). Shows a nice progress bar
        so you can see what's happening in real-time.

        Args:
            dataloader: The data to process
            mode: Whether we're training or validating

        Returns:
            The average loss for this epoch
        """
        # Set model to the right mode
        self.module.train() if mode == "training" else self.module.eval()
        logger.info(f"Model in {mode} mode")

        total_loss = np.float64(0.0)

        # Show progress with a nice bar
        async with Bar(iterations=len(dataloader), title=mode, steps=20) as bar:
            time = asyncio.get_event_loop().time()

            for batch, (inputs, targets) in enumerate(dataloader, start=1):
                # Skip bad data
                if not isinstance(inputs, torch.Tensor):
                    logger.warning("Got non-tensor inputs, skipping")
                    warnings.warn("Got non-tensor inputs, skipping")
                    continue

                try:
                    # Move everything to GPU/CPU
                    inputs = [input.to(device=self.device) for input in inputs]
                    targets = [
                        {k: v.to(device=self.device) for k, v in target.items()}
                        for target in targets
                    ]

                    # Clear gradients
                    self.optimizer.zero_grad()

                    # Forward pass (with or without gradients)
                    with torch.set_grad_enabled(mode=(mode == "training")):
                        outputs = self.module(inputs, targets)

                        # Calculate loss
                        if mode == "training":
                            loss = sum(loss for loss in outputs.values())
                        elif mode == "validation":
                            loss = self.criterion(inputs, targets)

                        # Skip bad losses
                        if not isinstance(loss, torch.Tensor):
                            logger.warning("Got non-tensor loss, skipping")
                            warnings.warn("Got non-tensor loss, skipping")
                            continue

                        if torch.isnan(loss):
                            logger.warning("Got NaN loss, skipping this batch")
                            warnings.warn("Got NaN loss, skipping this batch")
                            continue

                        # Backward pass and optimization (training only)
                        if mode == "training":
                            try:
                                loss.backward()
                                # Clip gradients to prevent exploding gradients
                                torch.nn.utils.clip_grad_norm_(
                                    parameters=self.module.parameters(), max_norm=1.0
                                )
                                self.optimizer.step()
                            except Exception as error:
                                logger.error(f"Error in backward pass: {str(error)}")
                                warnings.warn(f"Error in backward pass: {str(error)}")
                                continue

                    # Track the loss
                    total_loss = np.add(
                        total_loss,
                        np.multiply(np.float64(loss.item()), np.float64(len(inputs))),
                    )

                    # Update progress bar
                    await bar.update(batch=batch, time=time)
                    await bar.postfix(loss=np.divide(total_loss, batch))

                except Exception as error:
                    logger.error(f"Error processing batch: {str(error)}")
                    warnings.warn(f"Error processing batch: {str(error)}")
                    continue

            # Calculate average loss for the epoch
            average_loss = np.divide(total_loss, np.float64(len(dataloader)))
            logger.info(f"{mode.title()} epoch done, average loss: {average_loss:.4f}")

            return average_loss

    async def train(self) -> None:
        """
        Run the complete training process.

        This handles the main training loop, validation, overfitting detection,
        learning rate scheduling, and checkpointing. Basically everything you
        need for a complete training run.
        """
        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            try:
                print(f"Epoch {epoch + 1}/{self.epochs}")
                logger.info(f"Starting epoch {epoch + 1}/{self.epochs}")

                # Run training and validation
                for mode, dataloader in [
                    ("training", self.training_dataset),
                    ("validation", self.validation_dataset),
                ]:
                    loss = await self.rehearse(dataloader=dataloader, mode=mode)
                    logger.info(f"Epoch {epoch + 1}/{self.epochs}, {mode.title()} Loss: {loss:.4f}")
                    self.cache[mode].append(loss)

                # Check for overfitting
                if np.greater(epoch, 0):
                    # If validation loss goes up while training loss goes down = overfitting
                    if (np.greater(self.cache["validation"][-1], self.cache["validation"][-2]) and
                            np.less(self.cache["training"][-1], self.cache["training"][-2])):
                        logger.warning(f"Overfitting detected at epoch {epoch + 1}! Saving checkpoint.")
                        warnings.warn(f"Overfitting detected at epoch {epoch + 1}! Saving checkpoint.")
                        self.save(filepath=Path(f"checkpoints/epoch-{epoch + 1}.pth"))

                # Update learning rate
                scheduler = type(self.scheduler)

                if scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(epoch=epoch)
                    logger.info("Updated learning rate with ReduceLROnPlateau")
                elif scheduler in [
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    torch.optim.lr_scheduler.StepLR,
                    torch.optim.lr_scheduler.MultiStepLR,
                    torch.optim.lr_scheduler.ExponentialLR,
                ]:
                    self.optimizer.step()
                    self.scheduler.step()
                    logger.info(f"Updated learning rate with {scheduler.__name__}")

                # Log current learning rate
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Current learning rate: {lr:.6f}")

            except Exception as error:
                logger.error(f"Error in epoch {epoch + 1}: {str(error)}")
                warnings.warn(f"Error in epoch {epoch + 1}: {str(error)}")
                continue

        logger.info("Training finished!")

    async def test(self) -> None:
        """
        Test the final model and calculate accuracy.

        This runs the model on the test dataset and calculates both loss
        and accuracy using IoU matching between predictions and ground truth.
        Shows you how well your trained model actually performs.
        """
        self.module.eval()
        total_loss = np.float64(0.0)

        # Track predictions vs ground truth for accuracy
        total_targets = np.array([], dtype=np.int64)
        total_predictions = np.array([], dtype=np.int64)

        async with Bar(iterations=len(self.testing_dataset), title="Testing", steps=20) as bar:
            time = asyncio.get_event_loop().time()

            for batch, (inputs, targets) in enumerate(self.testing_dataset, start=1):
                try:
                    inputs = [input.to(self.device) for input in inputs]
                    targets = [
                        {k: v.to(self.device) for k, v in target.items()}
                        for target in targets
                    ]

                    # No gradients needed for testing
                    with torch.no_grad():
                        outputs = self.module(inputs)
                        loss = self.criterion(inputs, targets)

                        total_loss = np.add(
                            total_loss,
                            np.multiply(np.float64(loss.item()), np.float64(len(inputs))),
                        )

                        # Calculate accuracy for each image
                        for output, target in zip(outputs, targets):
                            total_targets = np.append(total_targets, len(target["boxes"]))

                            try:
                                # Match predictions to ground truth using IoU
                                iou = np.multiply(
                                    np.multiply(
                                        ops.box_iou(target["boxes"], output["boxes"]).numpy(),
                                        (target["labels"].unsqueeze(1) == output["labels"].unsqueeze(0)).numpy(),
                                    ),
                                    output["scores"].numpy()[np.newaxis, :],
                                )
                            except (RuntimeError, ValueError, IndexError):
                                warnings.warn("Empty boxes detected")
                                logger.warning("Empty boxes found during IoU calculation")

                            # Count correct predictions (IoU > 0)
                            total_predictions = np.append(
                                total_predictions,
                                np.sum(np.greater(np.max(iou, axis=1), 0)),
                            )

                    await bar.update(batch, time)
                    await bar.postfix(loss=np.divide(total_loss, np.float64(batch)))

                except Exception as error:
                    warnings.warn(f"Error in test batch {batch}: {str(error)}")
                    logger.error(f"Error in test batch {batch}: {str(error)}")
                    continue

        # Calculate final metrics
        accuracy = np.multiply(
            np.divide(
                np.sum(total_predictions),
                np.sum(total_targets),
                where=np.greater(np.sum(total_targets), 0),
            ),
            np.float64(100),
        )
        average_loss = np.divide(total_loss, np.float64(len(self.testing_dataset)))

        print(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        logger.info(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print("Testing done!")

    def save(self, filepath: Optional[Path] = None) -> None:
        """
        Save the trained model to disk.

        You can save just the model weights (.pth files) or the entire model.
        The weights-only format is usually better since it's more portable
        and takes up less space.

        Args:
            filepath: Where to save the model (defaults to 'model.pt')
        """
        # Default save location if none specified
        if not filepath:
            parent = Path(__file__).parent
            filepath = Path(parent, "model.pt")
        else:
            # Make sure the directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            if filepath.suffix == ".pth":
                # Save just the weights (recommended)
                torch.save(obj=self.module.state_dict(), f=filepath)
                print(f"Model weights saved to: {filepath}")
                logger.info(f"Model weights saved to: {filepath}")
            else:
                # Save the entire model
                torch.save(obj=self.module, f=filepath)
                print(f"Complete model saved to: {filepath}")
                logger.info(f"Complete model saved to: {filepath}")

        except (IOError, OSError) as error:
            warnings.warn(f"Couldn't save model to {filepath}: {str(error)}")
            logger.error(f"Couldn't save model to {filepath}: {str(error)}")
            raise
        except Exception as error:
            warnings.warn(f"Unexpected error saving model: {str(error)}")
            logger.error(f"Unexpected error saving model: {str(error)}")
            raise