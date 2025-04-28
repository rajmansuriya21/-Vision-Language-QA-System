import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Union, Tuple

def preprocess_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess an image for model input.
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size for resizing
        normalize: Whether to normalize the image
    
    Returns:
        Preprocessed image tensor
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Basic preprocessing
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    if normalize:
        transform = transforms.Compose([
            transform,
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Apply transformations
    image_tensor = transform(image)
    
    # Add batch dimension if needed
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def validate_image(image: Image.Image) -> bool:
    """
    Validate if the image is suitable for processing.
    
    Args:
        image: PIL Image to validate
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        # Check if image is not empty
        if image.size[0] == 0 or image.size[1] == 0:
            return False
        
        # Check if image has valid mode
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False
        
        # Check if image has reasonable dimensions
        if image.size[0] > 10000 or image.size[1] > 10000:
            return False
        
        return True
    
    except Exception:
        return False 