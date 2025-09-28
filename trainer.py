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

# Use logger configured in main application
logger = logging.getLogger("trainer")


class Trainer:
    """
    Object detection training controller with architecture-specific optimization.

    Manages the complete training pipeline for object detection models including
    data loading, model training, validation, overfitting detection, and model
    persistence. Automatically configures optimal hyperparameters and schedules
    based on the specific detection architecture being trained.

    Supports RetinaNet, Faster R-CNN, SSD, FCOS, and mobile variants with
    multi-GPU training capabilities and comprehensive progress monitoring.

    Args:
        module: Object detection model instance
        training_path: Directory containing training dataset
        validation_path: Directory containing validation dataset
        testing_path: Directory containing test dataset
        weights_path: Path to pre-trained model weights (optional)
        dimensions: Target image dimensions as (height, width) tuple
        epochs: Number of training epochs to execute
        batch: Training batch size
        lr: Learning rate override (uses architecture defaults if None)
        decay: Weight decay regularization factor
        gamma: Learning rate scheduler decay factor
        momentum: SGD momentum parameter
        workers: Number of data loading worker processes
        seed: Random seed for reproducible training
        parallelism: Enable multi-GPU distributed training
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
        Initialize training configuration with parameter validation.

        Validates all input parameters, configures the training environment,
        and sets up architecture-specific optimization strategies based on
        established best practices for each detection model type.
        """

        # Validate model instance
        if not isinstance(module, nn.Module):
            logger.error(f"Expected nn.Module, received {type(module)}")
            raise TypeError(f"Expected PyTorch model, got {type(module)}")
        logger.info(f"Model validated: {type(module).__name__}")

        # Validate dataset directory paths
        if not isinstance(training_path, Path) or not training_path.exists():
            logger.error(f"Training path invalid: {training_path}")
            raise ValueError(f"Training directory not found: {training_path}")
        logger.info(f"Training dataset located: {training_path}")

        if not isinstance(validation_path, Path) or not validation_path.exists():
            logger.error(f"Validation path invalid: {validation_path}")
            raise ValueError(f"Validation directory not found: {validation_path}")
        logger.info(f"Validation dataset located: {validation_path}")

        if not isinstance(testing_path, Path) or not testing_path.exists():
            logger.error(f"Test path invalid: {testing_path}")
            raise ValueError(f"Test directory not found: {testing_path}")
        logger.info(f"Test dataset located: {testing_path}")

        # Validate image dimensions configuration
        if not isinstance(dimensions, tuple) or len(dimensions) != 2:
            logger.error(f"Invalid dimensions format: {dimensions}")
            raise ValueError(f"Dimensions must be (height, width) tuple, got: {dimensions}")
        logger.info(f"Image target dimensions: {dimensions}")

        # Validate training parameters
        if not isinstance(epochs, int) or epochs <= 0:
            logger.error(f"Invalid epochs value: {epochs}")
            raise ValueError(f"Epochs must be positive integer, got: {epochs}")
        logger.info(f"Training duration: {epochs} epochs")

        if not isinstance(batch, int) or batch <= 0:
            logger.error(f"Invalid batch size: {batch}")
            raise ValueError(f"Batch size must be positive integer, got: {batch}")
        logger.info(f"Batch size configured: {batch}")

        # Validate pre-trained weights path
        if weights_path and (not isinstance(weights_path, Path) or not weights_path.exists()):
            logger.warning(f"Weights path not accessible: {weights_path}")
            warnings.warn(f"Pre-trained weights not found: {weights_path}")
        elif weights_path:
            logger.info(f"Pre-trained weights specified: {weights_path}")

        # Validate optional hyperparameters
        if momentum is not None and not isinstance(momentum, (float, int)):
            logger.warning(f"Momentum type invalid: {type(momentum)}")
            warnings.warn(f"Momentum should be numeric, got: {type(momentum)}")
        elif momentum is not None:
            logger.info(f"Momentum override: {momentum}")

        if parallelism is not None and not isinstance(parallelism, bool):
            logger.warning(f"Parallelism type invalid: {type(parallelism)}")
            warnings.warn(f"Parallelism should be boolean, got: {type(parallelism)}")
        else:
            logger.info(f"Multi-GPU training enabled: {parallelism}")

        # Configure deterministic training environment
        if seed is not None:
            self.seed(seed=seed)
            logger.info(f"Deterministic training configured: seed={seed}")

        # Determine compute device
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Compute device selected: {self.device}")

        # Configure model and store training parameters
        self.module: nn.Module = module.to(self.device)
        self.dimensions: Tuple[int, int] = dimensions
        self.epochs: int = epochs
        self.workers: Optional[int] = workers

        # Initialize model weights or load from checkpoint
        if weights_path is None:
            self.initialize_weights(self.module)
            logger.info("Model weights initialized randomly")
        else:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.module.load_state_dict(state_dict)
                logger.info(f"Pre-trained weights loaded: {weights_path}")
            except Exception as error:
                logger.warning(f"Weight loading failed: {error}. Using random initialization.")
                warnings.warn(f"Weight loading error: {error}. Using random initialization.")
                self.initialize_weights(self.module)

        # Initialize loss tracking storage
        self.cache: Dict[str, List[float]] = {"training": [], "validation": []}
        logger.info("Loss tracking initialized")

        # Configure multi-GPU training if available
        if parallelism and np.greater(torch.cuda.device_count(), 1):
            self.module = nn.parallel.DistributedDataParallel(self.module)
            logger.info(f"Multi-GPU training configured: {torch.cuda.device_count()} devices")
        else:
            logger.info("Single-device training mode")

        # Initialize dataset loaders
        try:
            self.training_dataset: DataLoader = self.loader(
                dirpath=training_path, batch=batch, mode="training"
            )
            logger.info("Training dataset loader created")
        except Exception as error:
            logger.error(f"Training dataset loading failed: {error}")
            warnings.warn(f"Training data loading error: {error}")

        try:
            self.validation_dataset: DataLoader = self.loader(
                dirpath=validation_path, batch=batch, mode="validation"
            )
            logger.info("Validation dataset loader created")
        except Exception as error:
            logger.error(f"Validation dataset loading failed: {error}")
            warnings.warn(f"Validation data loading error: {error}")

        try:
            self.testing_dataset: DataLoader = self.loader(
                dirpath=testing_path, batch=batch, mode="testing"
            )
            logger.info("Test dataset loader created")
        except Exception as error:
            logger.error(f"Test dataset loading failed: {error}")
            warnings.warn(f"Test data loading error: {error}")

        # Configure detection-specific loss function
        self.criterion: modules.MultiBoxLoss = modules.MultiBoxLoss(module=module)
        logger.info("Detection loss function configured")

        # Configure architecture-specific optimization strategies
        if isinstance(module, modules.RetinaNet):
            # RetinaNet optimization based on focal loss paper recommendations
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

            # Learning rate reduction at specific milestones
            self.scheduler: torch.optim.lr_scheduler = (
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=self.optimizer,
                    milestones=[8, 11],  # Standard RetinaNet schedule
                    gamma=gamma,
                )
            )
            logger.info("RetinaNet optimization strategy configured")

        elif isinstance(module, modules.FasterRCNN):
            # Faster R-CNN requires conservative learning rates
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

            # Step-based learning rate decay
            self.scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=3,
                gamma=gamma,
            )
            logger.info("Faster R-CNN optimization strategy configured")

        elif isinstance(module, modules.SSD):
            # SSD performs well with Adam optimization
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
            logger.info("SSD optimization strategy configured")

        elif isinstance(module, modules.FCOS):
            # FCOS benefits from cosine annealing schedule
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

            # Smooth cosine learning rate decay
            self.scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=12,
                eta_min=lr * 0.01,
            )
            logger.info("FCOS optimization strategy configured")

        else:
            logger.error(f"Unsupported model architecture: {type(module)}")
            raise ValueError(f"Training configuration not available for: {type(module)}")

        logger.info(
            f"Training configuration completed for {module.__class__.__name__}: "
            f"decay={decay}, gamma={gamma}, momentum={momentum}")

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        """
        Initialize model weights using layer-appropriate strategies.

        Applies optimal initialization schemes for different layer types
        to ensure stable training convergence and gradient flow throughout
        the network architecture.

        Args:
            module: Model instance to initialize
        """
        for layer in module.modules():
            # Convolutional layers use Kaiming initialization for ReLU networks
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            # Batch normalization layers initialized as identity transforms
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

            # Linear layers use Xavier initialization
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            # Recurrent layers use orthogonal initialization for gradient stability
            elif isinstance(layer, nn.LSTM):
                for parameter in layer.parameters():
                    if len(parameter.shape) >= 2:
                        nn.init.orthogonal_(parameter)
                    else:
                        nn.init.normal_(parameter)

            elif isinstance(layer, nn.GRU):
                for parameter in layer.parameters():
                    if len(parameter.shape) >= 2:
                        nn.init.orthogonal_(parameter)
                    else:
                        nn.init.normal_(parameter)

    @staticmethod
    def seed(seed: int) -> None:
        """
        Configure deterministic random state for reproducible training.

        Sets random seeds across Python, NumPy, PyTorch, and CUDA to ensure
        consistent results across training runs for debugging and comparison.

        Args:
            seed: Integer seed value for random state initialization
        """
        try:
            # Configure Python hash randomization
            os.environ["PYTHONHASHSEED"] = str(seed)

            # Set PyTorch random state
            torch.manual_seed(seed=seed)

            # Set Python standard library random state
            random.seed(a=seed)

            # Set NumPy random state
            np.random.seed(seed=seed)

            # Configure CUDA determinism if available
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed=seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            logger.info(f"Deterministic random state configured: seed={seed}")
        except Exception as error:
            logger.error(f"Random seed configuration failed: {str(error)}")
            warnings.warn(f"Unable to set random seed: {str(error)}")

    def loader(
            self,
            dirpath: Path,
            batch: int,
            mode: Literal["training", "validation", "testing"] = "training",
    ) -> Optional[DataLoader]:
        """
        Create DataLoader with mode-specific preprocessing pipeline.

        Configures appropriate image transformations based on dataset usage:
        training includes data augmentation while validation/testing uses
        basic preprocessing only.

        Args:
            dirpath: Dataset directory path
            batch: Batch size for data loading
            mode: Dataset usage mode (training, validation, testing)

        Returns:
            Configured DataLoader instance or None if creation fails
        """
        # Configure mode-specific image transformations
        if mode == "training":
            # Training augmentations for improved generalization
            transform = transforms.Compose([
                transforms.Resize(size=self.dimensions),
                transforms.RandomRotation(degrees=10),  # Geometric augmentation
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color augmentation
                transforms.ToTensor(),
                # ImageNet normalization for transfer learning compatibility
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            logger.info(f"Training preprocessing with augmentation configured")
        else:
            # Basic preprocessing for evaluation modes
            transform = transforms.Compose([
                transforms.Resize(size=self.dimensions),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            logger.info(f"Evaluation preprocessing configured for {mode}")

        try:
            # Initialize dataset with transformations
            dataset = DatasetLoader(dirpath=dirpath, transform=transform)

            # Configure DataLoader with detection-specific settings
            dataloader = DataLoader(
                dataset=dataset,
                collate_fn=collate_fn,  # Object detection batch collation
                batch_size=batch,
                shuffle=(mode == "training"),  # Randomize training data only
                num_workers=(4 if self.workers is None else self.workers),
                pin_memory=True,  # Optimize GPU memory transfers
            )
            logger.info(f"{mode.title()} DataLoader configured: batch_size={batch}")
            return dataloader
        except Exception as error:
            logger.error(f"{mode.title()} DataLoader creation failed: {str(error)}")
            warnings.warn(f"{mode.title()} data loading error: {str(error)}")
            return None

    async def rehearse(
            self, dataloader: DataLoader, mode: Literal["training", "validation"]
    ) -> np.float64:
        """
        Execute single epoch of training or validation with progress monitoring.

        Processes all batches in the dataloader, computing losses and updating
        model parameters (training only). Includes gradient clipping and
        comprehensive error handling for robust training.

        Args:
            dataloader: DataLoader containing batched dataset
            mode: Execution mode (training or validation)

        Returns:
            Average loss value for the completed epoch
        """
        # Configure model state for execution mode
        self.module.train() if mode == "training" else self.module.eval()
        logger.info(f"Model configured for {mode} mode")

        total_loss = np.float64(0.0)

        # Execute with progress monitoring
        async with Bar(iterations=len(dataloader), title=mode, steps=20) as bar:
            time = asyncio.get_event_loop().time()

            for batch, (inputs, targets) in enumerate(dataloader, start=1):
                # Validate input tensor format
                if not isinstance(inputs, torch.Tensor):
                    logger.warning("Non-tensor inputs detected - skipping batch")
                    warnings.warn("Non-tensor inputs detected - skipping batch")
                    continue

                try:
                    # Transfer data to compute device
                    inputs = [input.to(device=self.device) for input in inputs]
                    targets = [
                        {k: v.to(device=self.device) for k, v in target.items()}
                        for target in targets
                    ]

                    # Clear accumulated gradients
                    self.optimizer.zero_grad()

                    # Execute forward pass with conditional gradient computation
                    with torch.set_grad_enabled(mode=(mode == "training")):
                        outputs = self.module(inputs, targets)

                        # Compute appropriate loss based on mode
                        if mode == "training":
                            loss = sum(loss for loss in outputs.values())
                        elif mode == "validation":
                            loss = self.criterion(inputs, targets)

                        # Validate loss tensor
                        if not isinstance(loss, torch.Tensor):
                            logger.warning("Non-tensor loss detected - skipping batch")
                            warnings.warn("Non-tensor loss detected - skipping batch")
                            continue

                        if torch.isnan(loss):
                            logger.warning("NaN loss detected - skipping batch")
                            warnings.warn("NaN loss detected - skipping batch")
                            continue

                        # Execute backward pass and parameter updates (training only)
                        if mode == "training":
                            try:
                                loss.backward()
                                # Apply gradient clipping for training stability
                                torch.nn.utils.clip_grad_norm_(
                                    parameters=self.module.parameters(), max_norm=1.0
                                )
                                self.optimizer.step()
                            except Exception as error:
                                logger.error(f"Backward pass failed: {str(error)}")
                                warnings.warn(f"Backward pass error: {str(error)}")
                                continue

                    # Accumulate batch loss weighted by batch size
                    total_loss = np.add(
                        total_loss,
                        np.multiply(np.float64(loss.item()), np.float64(len(inputs))),
                    )

                    # Update progress display
                    await bar.update(batch=batch, time=time)
                    await bar.postfix(loss=np.divide(total_loss, batch))

                except Exception as error:
                    logger.error(f"Batch processing error: {str(error)}")
                    warnings.warn(f"Batch processing failed: {str(error)}")
                    continue

            # Calculate epoch average loss
            average_loss = np.divide(total_loss, np.float64(len(dataloader)))
            logger.info(f"{mode.title()} epoch completed - average loss: {average_loss:.4f}")

            return average_loss

    async def train(self) -> None:
        """
        Execute complete training process with overfitting detection and scheduling.

        Manages the full training loop including validation, learning rate
        scheduling, overfitting detection with automatic checkpointing,
        and comprehensive progress monitoring.
        """
        logger.info(f"Training process initiated: {self.epochs} epochs scheduled")

        for epoch in range(self.epochs):
            try:
                print(f"Epoch {epoch + 1}/{self.epochs}")
                logger.info(f"Epoch {epoch + 1}/{self.epochs} started")

                # Execute training and validation phases
                for mode, dataloader in [
                    ("training", self.training_dataset),
                    ("validation", self.validation_dataset),
                ]:
                    loss = await self.rehearse(dataloader=dataloader, mode=mode)
                    logger.info(f"Epoch {epoch + 1}/{self.epochs} - {mode.title()} Loss: {loss:.4f}")
                    self.cache[mode].append(loss)

                # Analyze for overfitting patterns
                if np.greater(epoch, 0):
                    # Detect validation loss increase with training loss decrease
                    if (np.greater(self.cache["validation"][-1], self.cache["validation"][-2]) and
                            np.less(self.cache["training"][-1], self.cache["training"][-2])):
                        logger.warning(f"Overfitting detected at epoch {epoch + 1} - creating checkpoint")
                        warnings.warn(f"Overfitting detected at epoch {epoch + 1} - creating checkpoint")
                        self.save(filepath=Path(f"checkpoints/epoch-{epoch + 1}.pth"))

                # Update learning rate based on scheduler configuration
                scheduler = type(self.scheduler)

                if scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(epoch=epoch)
                    logger.info("Learning rate updated via ReduceLROnPlateau")
                elif scheduler in [
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    torch.optim.lr_scheduler.StepLR,
                    torch.optim.lr_scheduler.MultiStepLR,
                    torch.optim.lr_scheduler.ExponentialLR,
                ]:
                    self.optimizer.step()
                    self.scheduler.step()
                    logger.info(f"Learning rate updated via {scheduler.__name__}")

                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Current learning rate: {current_lr:.6f}")

            except Exception as error:
                logger.error(f"Epoch {epoch + 1} execution failed: {str(error)}")
                warnings.warn(f"Epoch {epoch + 1} error: {str(error)}")
                continue

        logger.info("Training process completed successfully")

    async def test(self) -> None:
        """
        Evaluate trained model performance on test dataset with IoU-based accuracy.

        Executes model inference on test data and computes both loss and
        accuracy metrics using IoU matching between predictions and ground
        truth annotations.
        """
        self.module.eval()
        total_loss = np.float64(0.0)

        # Initialize prediction tracking for accuracy computation
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

                    # Execute inference without gradient computation
                    with torch.no_grad():
                        outputs = self.module(inputs)
                        loss = self.criterion(inputs, targets)

                        total_loss = np.add(
                            total_loss,
                            np.multiply(np.float64(loss.item()), np.float64(len(inputs))),
                        )

                        # Compute accuracy using IoU matching
                        for output, target in zip(outputs, targets):
                            total_targets = np.append(total_targets, len(target["boxes"]))

                            try:
                                # Calculate IoU-based prediction matching
                                iou = np.multiply(
                                    np.multiply(
                                        ops.box_iou(target["boxes"], output["boxes"]).numpy(),
                                        (target["labels"].unsqueeze(1) == output["labels"].unsqueeze(0)).numpy(),
                                    ),
                                    output["scores"].numpy()[np.newaxis, :],
                                )
                            except (RuntimeError, ValueError, IndexError):
                                warnings.warn("Empty detection boxes encountered")
                                logger.warning("Empty boxes during IoU calculation")

                            # Count correct predictions based on IoU threshold
                            total_predictions = np.append(
                                total_predictions,
                                np.sum(np.greater(np.max(iou, axis=1), 0)),
                            )

                    await bar.update(batch, time)
                    await bar.postfix(loss=np.divide(total_loss, np.float64(batch)))

                except Exception as error:
                    warnings.warn(f"Test batch {batch} processing failed: {str(error)}")
                    logger.error(f"Test batch {batch} error: {str(error)}")
                    continue

        # Compute final evaluation metrics
        accuracy = np.multiply(
            np.divide(
                np.sum(total_predictions),
                np.sum(total_targets),
                where=np.greater(np.sum(total_targets), 0),
            ),
            np.float64(100),
        )
        average_loss = np.divide(total_loss, np.float64(len(self.testing_dataset)))

        print(f"Test Results - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        logger.info(f"Test evaluation completed - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print("Model evaluation completed")

    def save(self, filepath: Optional[Path] = None) -> None:
        """
        Persist trained model to storage with format-specific handling.

        Saves model weights (.pth format) or complete model architecture
        based on file extension. Creates necessary directory structure
        if it doesn't exist.

        Args:
            filepath: Target save location (defaults to 'model.pt')
        """
        # Use default save location if none specified
        if not filepath:
            parent = Path(__file__).parent
            filepath = Path(parent, "model.pt")
        else:
            # Ensure target directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            if filepath.suffix == ".pth":
                # Save state dictionary (recommended approach)
                torch.save(obj=self.module.state_dict(), f=filepath)
                print(f"Model weights saved: {filepath}")
                logger.info(f"Model state dictionary saved: {filepath}")
            else:
                # Save complete model architecture
                torch.save(obj=self.module, f=filepath)
                print(f"Complete model saved: {filepath}")
                logger.info(f"Full model architecture saved: {filepath}")

        except (IOError, OSError) as error:
            warnings.warn(f"Model save failed to {filepath}: {str(error)}")
            logger.error(f"Model save error to {filepath}: {str(error)}")
            raise
        except Exception as error:
            warnings.warn(f"Unexpected model save error: {str(error)}")
            logger.error(f"Unexpected model save error: {str(error)}")
            raise