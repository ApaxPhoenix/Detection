import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from pathlib import Path
import warnings
import logging

# Get the logger from our main app
logger = logging.getLogger("loader")


def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """Combines individual data samples into a single batch for training.

    Images sometimes fail to load and return None. This function removes
    failed samples and stacks successful ones into proper batch tensors
    that PyTorch can work with.

    Args:
        batch: Collection of (image, mask) pairs. Failed samples will be None.

    Returns:
        Batch tensors ready for model training, or None if all samples failed.
    """
    # Remove any samples that failed during loading
    batch: List[Tuple[torch.Tensor, torch.Tensor]] = [item for item in batch if item is not None]

    # Check if we lost everything
    if not batch:
        warnings.warn(message="Complete batch failure - no samples loaded successfully")
        logger.warning("All samples in batch returned None")
        return None

    # Split images and masks for separate stacking
    images: Tuple[torch.Tensor, ...]
    masks: Tuple[torch.Tensor, ...]
    images, masks = zip(*batch)

    # Combine into batch tensors
    return torch.stack(tensors=images), torch.stack(tensors=masks)


class DatasetLoader(Dataset):
    """
    Loads images and their bounding box annotations for object detection.

    This class handles the boring stuff of reading image files and matching them
    up with their XML annotation files. It figures out what classes you have,
    loads everything into the right format, and handles errors when files are
    corrupted or missing.

    The annotations need to be in Pascal VOC XML format (the standard format
    that most annotation tools export to).

    Expected directory structure:
        dataset/
        ├── images/
        │   ├── image001.jpg
        │   ├── image002.png
        │   └── ...
        └── annotations/
            ├── image001.xml
            ├── image002.xml
            └── ...
    """

    def __init__(
            self, dirpath: Path, transform: Optional[transforms.Compose] = None
    ) -> None:
        """
        Set up the dataset by finding all matching image and annotation files.

        Args:
            dirpath: Path to your dataset folder (should contain 'images' and 'annotations')
            transform: Any image transformations you want to apply (resize, normalize, etc.)
        """
        # Remember where everything is
        self.images: Path = Path(dirpath, "images")
        self.annotations: Path = Path(dirpath, "annotations")
        self.transform: Optional[transforms.Compose] = transform

        # Make sure the folders actually exist
        if not self.images.exists():
            warnings.warn(message=f"Can't find images folder at {self.images}")
            logger.warning(f"Images folder missing: {self.images}")
            return

        if not self.annotations.exists():
            warnings.warn(
                message=f"Can't find annotations folder at {self.annotations}"
            )
            logger.warning(f"Annotations folder missing: {self.annotations}")
            return

        # Find pairs of images and annotations that have matching names
        # This looks for .jpg, .jpeg, .png, and .tiff files
        self.files: List[Tuple[Path, Path]] = [
            (image, annotation)
            for annotation in self.annotations.glob(pattern="*.xml")
            for pattern in [
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.tiff",
            ]
            for image in self.images.glob(pattern=pattern)
            if image.stem == annotation.stem  # Same filename, different extensions
        ]

        # Build a dictionary that maps class names to numbers
        # This goes through all the XML files and finds every unique class name
        self.classes: Dict[str, int] = {
            name: label
            for label, name in enumerate(
                dict.fromkeys(  # Removes duplicates while keeping order
                    obj.find(path="name").text
                    for _, annotation in self.files
                    for obj in ET.parse(source=annotation)
                    .getroot()
                    .findall(path="object")
                )
            )
        }

        # Warn if we didn't find anything useful
        if not self.files:
            warnings.warn(
                message="Couldn't find any matching image-annotation pairs. Check your file names!"
            )
            logger.warning("No valid pairs found - check that image and annotation filenames match")
            return

        logger.info(f"Found {len(self.files)} image-annotation pairs")
        logger.info(f"Detected {len(self.classes)} different classes: {list(self.classes.keys())}")

    def __len__(self) -> int:
        """
        How many items are in this dataset?

        Returns:
            Number of image-annotation pairs we found
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Get a single item from the dataset.

        This loads the image, applies any transforms, reads the XML file,
        and packages everything up in the format the model expects.

        Args:
            index: Which item you want (0 to len(dataset)-1)

        Returns:
            A tuple with:
            - The image as a tensor
            - A dictionary with 'boxes', 'labels', and 'scores'
            Returns None if something went wrong loading this item
        """
        # Get the file paths for this index
        image: Path
        annotation: Path
        image, annotation = self.files[index]

        try:
            # Load and transform the image
            image: Image.Image = Image.open(fp=image)
            if self.transform:
                image: Union[Image.Image, torch.Tensor] = self.transform(img=image)

            # Parse the XML annotation
            tree: ET.ElementTree = ET.parse(source=annotation)
            root: ET.Element = tree.getroot()

            # Get all the objects in this image
            objects: List[ET.Element] = root.findall(path="object")

            # Extract bounding boxes (x1, y1, x2, y2 format)
            boxes: List[List[int]] = [
                [
                    int(cast(ET.Element, object.find(path="bndbox/xmin")).text),
                    int(cast(ET.Element, object.find(path="bndbox/ymin")).text),
                    int(cast(ET.Element, object.find(path="bndbox/xmax")).text),
                    int(cast(ET.Element, object.find(path="bndbox/ymax")).text),
                ]
                for object in objects
            ]

            # Convert class names to numbers using our class dictionary
            labels: List[int] = [
                self.classes[cast(ET.Element, object.find(path="name")).text]
                for object in objects
            ]

            # Package everything up for the model
            target: Dict[str, torch.Tensor] = {
                "boxes": torch.tensor(data=boxes),
                "labels": torch.tensor(data=labels),
            }

            return image, target

        except Exception as error:
            # Something went wrong - log it and move on
            logger.error(f"Failed to load item {index} ({image}): {str(error)}")
            warnings.warn(f"Couldn't load item {index}: {str(error)}")
            return None


