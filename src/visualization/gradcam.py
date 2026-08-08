"""
Grad-CAM explainability visualizations.

Produces heatmaps showing which image regions drove the model's defect
prediction — used for qualitative error analysis and demo purposes (very
effective in a viva to show the model is actually attending to defects,
not spurious background features).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam_overlay(
    model: nn.Module,
    image_path: str | Path,
    target_layer: nn.Module,
    target_class: int | None = None,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap overlay for a single image.

    Args:
        model: Trained model in eval mode.
        image_path: Path to the input image.
        target_layer: The convolutional layer to compute Grad-CAM against
            (typically the last conv block, e.g. model.layer4[-1] for ResNet18).
        target_class: Class index to explain; if None, uses the model's predicted class.
        save_path: If provided, saves the overlay image.

    Returns:
        RGB numpy array of the image with the Grad-CAM heatmap overlaid.
    """
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    rgb_image = np.array(image.resize((224, 224))) / 255.0

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None if target_class is None else [target_class])[0]

    overlay = show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(save_path)

    return overlay
