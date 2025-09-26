import torch
import torch.nn as nn
import torchvision.ops as ops
import torchvision.models.detection as modules
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.image_list import ImageList
from collections import OrderedDict
import logging
import warnings

# Grab our logger from the main app
logger = logging.getLogger("modules")


class FasterRCNN(nn.Module):
    """The classic two-stage object detector that's super accurate but kinda slow.

    This is the gold standard for object detection when you care more about getting
    every single object right than running in real-time. It works in two steps:
    first it finds potential objects, then it classifies what they are and cleans
    up the bounding boxes.

    Args:
        classes: How many different types of objects you want to detect
        channels: Number of color channels (3 for normal RGB images)
        weights: Should we start with weights trained on COCO? (Usually yes)

    What goes in:
        A list of image tensors (can be different sizes, it'll handle it)

    What comes out:
        During training: A dictionary with all the different loss values
        During testing: Boxes, labels, and confidence scores for each detection
    """

    def __init__(self, classes, channels=3, weights=True):
        super().__init__()

        logger.info(
            f"Setting up Faster R-CNN with {classes} classes and {channels} channels"
        )

        # Get the base model, optionally with pretrained weights
        self.module = modules.fasterrcnn_resnet50_fpn(
            weights=modules.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if weights else None
        )

        # If we're not using regular RGB images, we need to change the first layer
        if channels != 3:
            self.module.backbone.body.conv1 = nn.Conv2d(
                channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Swap out the final classification layer for our number of classes
        in_features = self.module.roi_heads.box_predictor.cls_score.in_features
        self.module.roi_heads.box_predictor = modules.faster_rcnn.FastRCNNPredictor(
            in_features, classes
        )

    def forward(self, inputs, targets=None):
        """Run the model on some images.

        Args:
            inputs: Your images as tensors
            targets: The correct answers (only needed during training)

        Returns:
            Training mode: How wrong we were (losses for backprop)
            Testing mode: What we think is in each image
        """
        return self.module(inputs, targets)


class MobileFasterRCNN(nn.Module):
    """Faster R-CNN's little brother that runs on phones.

    Same idea as regular Faster R-CNN but uses a mobile-friendly backbone.
    It's smaller, faster, and uses less memory, but might miss a few more
    objects. Perfect when you need to run detection on mobile devices.

    Args:
        classes: How many different object types to detect
        channels: Number of image channels (usually 3)
        weights: Use pretrained weights? (recommended)
    """

    def __init__(self, classes, channels=3, weights=True):
        super().__init__()

        logger.info(
            f"Setting up Mobile Faster R-CNN with {classes} classes and {channels} channels"
        )

        # Load the mobile version
        self.module = modules.fasterrcnn_mobilenet_v3_large_fpn(
            weights=(
                modules.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                if weights
                else None
            )
        )

        # Mobile nets have different first layer structure
        if channels != 3:
            self.module.backbone.body.features[0][0] = nn.Conv2d(
                channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )

        # Update the classifier head
        in_features = self.module.roi_heads.box_predictor.cls_score.in_features
        self.module.roi_heads.box_predictor = modules.faster_rcnn.FastRCNNPredictor(
            in_features, classes
        )

    def forward(self, inputs, targets=None):
        """Run the mobile model."""
        return self.module(inputs, targets)


class RetinaNet(nn.Module):
    """Single-stage detector that's really good at finding small objects.

    Instead of the two-stage approach, this model predicts everything in one go.
    It uses something called "focal loss" to handle the fact that most of the
    image is background. Great for crowded scenes with lots of small objects.

    Args:
        classes: Number of object types to detect
        channels: Image channels (3 for RGB)
        threshold: How confident does it need to be to keep a detection?
        weights: Use pretrained weights?
    """

    def __init__(self, classes, channels=3, threshold=0.01, weights=True):
        super().__init__()

        logger.info(f"Setting up RetinaNet with {classes} classes and {channels} channels")

        # Load the model with our confidence threshold
        self.module = modules.retinanet_resnet50_fpn(
            weights=modules.RetinaNet_ResNet50_FPN_Weights.DEFAULT if weights else None,
            score_thresh=threshold,
        )

        # Handle non-RGB images
        if channels != 3:
            self.module.backbone.body.conv1 = nn.Conv2d(
                channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace the detection head for our classes
        self.module.head = modules.retinanet.RetinaNetHead(
            self.module.backbone.out_channels,
            self.module.anchor_generator.num_anchors_per_location()[0],
            classes,
        )

    def forward(self, inputs, targets=None):
        """Run RetinaNet detection."""
        return self.module(inputs, targets)


class FCOS(nn.Module):
    """The anchor-free detector that doesn't use anchor boxes at all.

    This is pretty cool - instead of predicting whether predefined boxes contain
    objects, it predicts for each pixel whether it's the center of an object and
    how far the edges are. No anchor boxes needed! It also predicts "centerness"
    to filter out low-quality detections.

    Args:
        classes: Number of different object types
        channels: Image channels (3 for RGB)
        threshold: Confidence threshold for keeping detections
        weights: Use pretrained weights?
    """

    def __init__(self, classes, channels=3, threshold=0.01, weights=True):
        super().__init__()

        logger.info(f"Setting up FCOS with {classes} classes and {channels} channels")

        # Load the anchor-free model
        self.module = modules.fcos_resnet50_fpn(
            weights=modules.FCOS_ResNet50_FPN_Weights.DEFAULT if weights else None,
            score_thresh=threshold,
        )

        # Handle different input channels
        if channels != 3:
            self.module.backbone.body.conv1 = nn.Conv2d(
                channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace the detection head
        self.module.head = modules.fcos.FCOSHead(
            self.module.backbone.out_channels,
            1,
            classes,
        )

    def forward(self, inputs, targets=None):
        """Run FCOS detection."""
        return self.module(inputs, targets)


class SSD(nn.Module):
    """The original single-stage detector that started it all.

    SSD (Single Shot MultiBox Detector) predicts objects at multiple scales
    using a VGG backbone. It's pretty fast and works well, but needs exactly
    300x300 pixel images. Old but reliable!

    Args:
        classes: Number of object types to detect
        channels: Image channels (3 for RGB)
        weights: Use pretrained weights?

    Important: Your images MUST be 300x300 pixels!
    """

    def __init__(self, classes, channels=3, weights=True):
        super().__init__()

        logger.info(f"Setting up SSD300 with {classes} classes and {channels} channels")

        # Load SSD with VGG backbone
        self.module = modules.ssd300_vgg16(
            weights=modules.SSD300_VGG16_Weights.DEFAULT if weights else None
        )

        # VGG has a different structure for the first layer
        if channels != 3:
            self.module.backbone[0] = nn.Conv2d(
                channels, 64, kernel_size=3, stride=1, padding=1
            )

        # Replace both the classification and box regression heads
        num_anchors = self.module.anchor_generator.num_anchors_per_location()
        in_channels = [
            layer.in_channels
            for layer in self.module.head.classification_head.module_list
        ]

        self.module.head.classification_head = modules.ssd.SSDClassificationHead(
            in_channels, num_anchors, classes
        )
        self.module.head.regression_head = modules.ssd.SSDRegressionHead(
            in_channels, num_anchors
        )

    def forward(self, inputs, targets=None):
        """Run SSD detection (remember: needs 300x300 images!)."""
        return self.module(inputs, targets)


class SSDLite(nn.Module):
    """SSD's mobile-friendly cousin that runs great on phones.

    Uses MobileNet instead of VGG, which makes it much faster and smaller.
    Needs 320x320 images instead of 300x300. Perfect for mobile apps where
    you need real-time detection without killing the battery.

    Args:
        classes: Number of object types to detect
        channels: Image channels (3 for RGB)
        weights: Use pretrained weights?

    Important: Your images need to be 320x320 pixels!
    """

    def __init__(self, classes, channels=3, weights=True):
        super().__init__()

        logger.info(f"Setting up SSDLite320 with {classes} classes and {channels} channels")

        # Load the mobile version
        self.module = modules.ssdlite320_mobilenet_v3_large(
            weights=(
                modules.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                if weights
                else None
            )
        )

        # MobileNet first layer is different
        if channels != 3:
            self.module.backbone.features[0][0] = nn.Conv2d(
                channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )

        # Replace the heads
        num_anchors = self.module.anchor_generator.num_anchors_per_location()
        in_channels = [
            layer.in_channels
            for layer in self.module.head.classification_head.module_list
        ]

        self.module.head.classification_head = modules.ssd.SSDClassificationHead(
            in_channels, num_anchors, classes
        )
        self.module.head.regression_head = modules.ssd.SSDRegressionHead(
            in_channels, num_anchors
        )

    def forward(self, inputs, targets=None):
        """Run SSDLite detection (needs 320x320 images!)."""
        return self.module(inputs, targets)


class MultiBoxLoss(nn.Module):
    """Universal loss calculator that works with all our detection models.

    This is pretty handy - it figures out what type of model you're using
    and calculates the loss the right way for that architecture. Each model
    has its own quirks for how losses are computed, so this handles all that
    complexity for you.

    Args:
        module: The detection model you want to calculate losses for

    Note: This is designed for validation where we don't need gradients,
    so it's more memory efficient than training mode.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        logger.info(f"Setting up loss calculator for {type(module).__name__}")

    def forward(self, inputs, targets):
        """Calculate how wrong our predictions are.

        This gets pretty complex because each model type needs different
        loss calculations. We basically recreate what happens during training
        but without storing gradients to save memory.

        Args:
            inputs: List of image tensors
            targets: List of ground truth boxes and labels

        Returns:
            A single loss value that combines all the different loss components
        """
        # Temporarily enable training mode for loss calculations
        self.module.module.training = True

        # Figure out what device we're on
        device = (
            inputs[0].device
            if inputs and isinstance(inputs[0], torch.Tensor)
            else torch.device("cpu")
        )

        # Start with zero loss
        loss = torch.tensor(0.0, device=device)

        # Can't calculate loss without targets
        if not targets:
            warnings.warn("No ground truth provided - can't calculate loss")
            return loss

        # Package up the images with size info
        sizes = [(input.shape[-2], input.shape[-1]) for input in inputs]
        batch = ImageList(torch.stack(inputs), sizes)

        # Switch to eval mode and disable gradients for memory efficiency
        self.module.eval()
        with torch.no_grad():
            try:
                # Get features from the backbone (common to all models)
                backbone = self.module.module.backbone(batch.tensors)

                # Make sure we have a consistent format
                if isinstance(backbone, torch.Tensor):
                    backbone = OrderedDict([("0", backbone)])

                features = list(backbone.values())

                # SSD and SSDLite need special handling
                if isinstance(self.module, (SSD, SSDLite)):
                    # Check if images are big enough to avoid kernel errors
                    minimum = 300 if isinstance(self.module, SSD) else 320
                    for size in sizes:
                        if size[0] < minimum or size[1] < minimum:
                            raise ValueError(
                                f"Images too small! Need at least {minimum}x{minimum} pixels"
                            )

                    try:
                        # Run through the detection head
                        predictions = self.module.module.head(features)
                        anchors = self.module.module.anchor_generator(batch, features)

                        # Match anchors to ground truth boxes
                        matches = [
                            torch.full(
                                (anchor.size(0),),
                                -1,
                                dtype=torch.int64,
                                device=anchor.device,
                            )
                            if target["boxes"].numel() == 0
                            else self.module.module.proposal_matcher(ops.box_iou(target["boxes"], anchor))
                            for anchor, target in zip(anchors, targets)
                        ]

                        # Calculate the actual loss
                        results = self.module.module.compute_loss(
                            targets, predictions, anchors, matches
                        )
                        loss = sum(results.values())

                    except Exception as error:
                        warnings.warn(f"SSD loss calculation failed: {str(error)}")
                        return torch.tensor(0.0, device=device)

                # Faster R-CNN and Mobile Faster R-CNN (the two-stage models)
                elif isinstance(self.module, (FasterRCNN, MobileFasterRCNN)):
                    try:
                        # Enable training mode for the RPN and ROI heads
                        self.module.module.rpn.training = True
                        self.module.module.roi_heads.training = True

                        # Run through the Region Proposal Network
                        predictions, deltas = self.module.module.rpn.head(features)
                        anchors = self.module.module.rpn.anchor_generator(batch, features)

                        # Format predictions properly
                        layers = [torch.prod(torch.tensor(output[0].shape)) for output in predictions]
                        predictions, deltas = concat_box_prediction_layers(predictions, deltas)

                        # Generate object proposals
                        proposals = self.module.module.rpn.box_coder.decode(deltas, anchors)
                        proposals = proposals.view(len(anchors), -1, 4)
                        proposals, scores = self.module.module.rpn.filter_proposals(
                            proposals, predictions, batch.image_sizes, layers
                        )

                        # Get RPN training targets
                        labels, boxes = self.module.module.rpn.assign_targets_to_anchors(anchors, targets)
                        offsets = self.module.module.rpn.box_coder.encode(boxes, anchors)

                        # Calculate RPN losses
                        localization, regression = self.module.module.rpn.compute_loss(
                            predictions, deltas, labels, offsets
                        )

                        # Process ROI head
                        proposals, indices, labels, offsets = (
                            self.module.module.roi_heads.select_training_samples(proposals, targets)
                        )

                        # Extract features and make final predictions
                        pooled = self.module.module.roi_heads.box_roi_pool(backbone, proposals, batch.image_sizes)
                        extracted = self.module.module.roi_heads.box_head(pooled)
                        logits, refinement = self.module.module.roi_heads.box_predictor(extracted)

                        # Calculate final classification and box refinement losses
                        classification, positioning = fastrcnn_loss(logits, refinement, labels, offsets)

                        # Reset training flags
                        self.module.module.rpn.training = False
                        self.module.module.roi_heads.training = False

                        # Add up all the loss components
                        loss = regression + localization + classification + positioning

                    except Exception as error:
                        warnings.warn(f"Faster R-CNN loss calculation failed: {str(error)}")
                        return torch.tensor(0.0, device=device)

                # RetinaNet (single-stage with focal loss)
                elif isinstance(self.module, RetinaNet):
                    try:
                        predictions = self.module.module.head(features)
                        anchors = self.module.module.anchor_generator(batch, features)
                        results = self.module.module.compute_loss(targets, predictions, anchors)
                        loss = sum(results.values())

                    except Exception as error:
                        warnings.warn(f"RetinaNet loss calculation failed: {str(error)}")
                        return torch.tensor(0.0, device=device)

                # FCOS (anchor-free detection)
                elif isinstance(self.module, FCOS):
                    try:
                        # FCOS is simpler - just call the model directly
                        results = self.module.module(inputs, targets)
                        loss = sum(results.values())

                    except Exception as error:
                        warnings.warn(f"FCOS loss calculation failed: {str(error)}")
                        return torch.tensor(0.0, device=device)

            except Exception as error:
                # Handle the common SSD kernel size error with a helpful message
                if isinstance(self.module, (SSD, SSDLite)) and "Kernel size can't be greater than actual input size" in str(error):
                    raise ValueError("Images are too small! Make sure they're at least 330x330 pixels.")
                else:
                    warnings.warn(f"Loss calculation failed: {error}")
                    return torch.tensor(0.0, device=device)

        # Make sure we're back in eval mode
        self.module.module.training = False
        return loss