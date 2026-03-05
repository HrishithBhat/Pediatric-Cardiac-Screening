"""
Grad-CAM Explainability Module
================================
Produces class-discriminative visual explanations for each modality.

Supported targets:
  - EfficientNetV2XRay  → heatmap on chest X-ray
  - NTSNet              → heatmap on ultrasound image
  - CRNN2D              → saliency overlay on spectrogram

Algorithm (standard Grad-CAM):
  1. Forward pass – record activations at the target conv layer
  2. Backward pass – record gradients at the target conv layer
  3. Weight each channel by its global average gradient
  4. ReLU-activate the weighted sum → CAM  (H', W')
  5. Bilinear-upsample to input resolution
  6. Blend with original image (jet colormap)

References
-----------
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hook-based Grad-CAM Engine
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Generic Grad-CAM that can be attached to any conv layer.

    Parameters
    ----------
    model      : nn.Module
    target_layer: the specific nn.Module (conv/block) to hook
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        input_tensor : (1, C, H, W) or (1, 1, H, W)  — single sample only
        target_class : which output neuron to back-propagate from (default 0)
        input_size   : (H, W) of the original input for upsampling

        Returns
        -------
        cam : np.ndarray  shape (H, W)  float32 in [0, 1]
        """
        assert input_tensor.shape[0] == 1, "Grad-CAM only supports batch_size=1"
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]  # MultimodalModel returns (logit, gates)

        # Backward on the target class score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Grad-CAM
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (H', W')
        cam = F.relu(cam)

        # Normalise
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        cam = cam.cpu().numpy()

        # Upsample to input resolution
        if input_size is not None:
            cam = cv2.resize(cam, (input_size[1], input_size[0]))

        return cam.astype(np.float32)  # (H, W)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def overlay_heatmap(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap with the original image.

    Parameters
    ----------
    original_image : (H, W, 3) uint8  BGR
    cam            : (H, W) float32 in [0, 1]
    alpha          : overlay transparency

    Returns
    -------
    blended : (H, W, 3) uint8
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    blended = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    return blended


def spectrogram_gradcam_overlay(
    spectrogram: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Overlay Grad-CAM on a spectrogram (grayscale → RGB).

    Parameters
    ----------
    spectrogram : (H, W) float32 in [0, 1]
    cam         : (H, W) float32 in [0, 1]

    Returns
    -------
    (H, W, 3) uint8
    """
    spec_rgb = cv2.cvtColor(np.uint8(255 * spectrogram), cv2.COLOR_GRAY2BGR)
    return overlay_heatmap(spec_rgb, cam, alpha=alpha)


# ---------------------------------------------------------------------------
# Modality-specific Grad-CAM wrappers
# ---------------------------------------------------------------------------

class XRayGradCAM:
    """
    Grad-CAM for EfficientNetV2XRay.
    Hooks onto the last feature block.
    """

    def __init__(self, model):
        target_layer = model.features[-1]
        self.gc = GradCAM(model, target_layer)

    def explain(self, image_tensor: torch.Tensor) -> np.ndarray:
        """image_tensor: (1, 3, 224, 224) → cam (224, 224)"""
        return self.gc.generate(image_tensor, target_class=0, input_size=(224, 224))

    def explain_and_overlay(
        self, image_tensor: torch.Tensor, original_bgr: np.ndarray
    ) -> np.ndarray:
        cam = self.explain(image_tensor)
        return overlay_heatmap(original_bgr, cam)

    def remove_hooks(self):
        self.gc.remove_hooks()


class UltrasoundGradCAM:
    """
    Grad-CAM for NTSNet.
    Hooks onto the last block of the ResNet-50 backbone.
    """

    def __init__(self, model):
        # model.backbone is Sequential(*list(resnet50.children())[:-2])
        # The last child is layer4
        target_layer = model.backbone[-1]
        self.gc = GradCAM(model, target_layer)

    def explain(self, image_tensor: torch.Tensor) -> np.ndarray:
        return self.gc.generate(image_tensor, target_class=0, input_size=(224, 224))

    def explain_and_overlay(
        self, image_tensor: torch.Tensor, original_bgr: np.ndarray
    ) -> np.ndarray:
        cam = self.explain(image_tensor)
        return overlay_heatmap(original_bgr, cam)

    def remove_hooks(self):
        self.gc.remove_hooks()


class AudioGradCAM:
    """
    Grad-CAM for CRNN2D.
    Hooks onto the last ResNet-18 stage (layer4) to produce a
    frequency×time saliency map over the spectrogram.
    """

    def __init__(self, model):
        # model.cnn is a Sequential; index 7 is layer4
        target_layer = model.cnn[7]   # layer4
        self.gc = GradCAM(model, target_layer)

    def explain(self, spec_tensor: torch.Tensor) -> np.ndarray:
        """spec_tensor: (1, 1, 64, 256) → cam (64, 256)"""
        return self.gc.generate(spec_tensor, target_class=0, input_size=(64, 256))

    def explain_and_overlay(
        self, spec_tensor: torch.Tensor
    ) -> np.ndarray:
        """Returns a coloured spectrogram with Grad-CAM overlay."""
        cam = self.explain(spec_tensor)
        spec_np = spec_tensor[0, 0].cpu().numpy()  # (64, 256) float
        return spectrogram_gradcam_overlay(spec_np, cam)

    def remove_hooks(self):
        self.gc.remove_hooks()


# ---------------------------------------------------------------------------
# Full Multimodal Explainability Report
# ---------------------------------------------------------------------------

def generate_explainability_report(
    multimodal_model,
    audio_spec: Optional[torch.Tensor] = None,
    us_image: Optional[torch.Tensor] = None,
    xray_image: Optional[torch.Tensor] = None,
    us_bgr: Optional[np.ndarray] = None,
    xray_bgr: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> dict:
    """
    Generate Grad-CAM visualisations for each available modality.

    Returns a dict:
    {
        "audio_cam_overlay"   : np.ndarray (64, 256, 3) uint8  or None,
        "us_cam_overlay"      : np.ndarray (224, 224, 3) uint8 or None,
        "xray_cam_overlay"    : np.ndarray (224, 224, 3) uint8 or None,
        "gate_weights"        : dict  {audio, ultrasound, xray} mean gate values,
        "prediction"          : float  (probability of referral),
    }
    """
    results = {
        "audio_cam_overlay": None,
        "us_cam_overlay": None,
        "xray_cam_overlay": None,
        "gate_weights": None,
        "prediction": None,
    }

    # Overall prediction + gate weights (no grad needed)
    with torch.no_grad():
        logit, gate_weights = multimodal_model(
            audio_spec.to(device) if audio_spec is not None else None,
            us_image.to(device) if us_image is not None else None,
            xray_image.to(device) if xray_image is not None else None,
        )
    results["prediction"] = torch.sigmoid(logit).item()
    results["gate_weights"] = {
        k: v.mean().item() for k, v in gate_weights.items()
    }

    # Audio Grad-CAM
    if audio_spec is not None:
        audio_gc = AudioGradCAM(multimodal_model.crnn)
        results["audio_cam_overlay"] = audio_gc.explain_and_overlay(
            audio_spec[:1].to(device)
        )
        audio_gc.remove_hooks()

    # Ultrasound Grad-CAM
    if us_image is not None:
        us_gc = UltrasoundGradCAM(multimodal_model.nts_net)
        bgr = us_bgr if us_bgr is not None else np.zeros((224, 224, 3), np.uint8)
        results["us_cam_overlay"] = us_gc.explain_and_overlay(
            us_image[:1].to(device), bgr
        )
        us_gc.remove_hooks()

    # X-Ray Grad-CAM
    if xray_image is not None:
        xray_gc = XRayGradCAM(multimodal_model.effnet)
        bgr = xray_bgr if xray_bgr is not None else np.zeros((224, 224, 3), np.uint8)
        results["xray_cam_overlay"] = xray_gc.explain_and_overlay(
            xray_image[:1].to(device), bgr
        )
        xray_gc.remove_hooks()

    return results
