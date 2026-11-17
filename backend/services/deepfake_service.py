"""Deepfake detection service wrapping CNN+FFT architecture.

Supports:
1. High-speed, lightweight pure NumPy inference engine (Serverless / Vercel friendly, zero PyTorch/CUDA bloat).
2. Native PyTorch engine if PyTorch is installed.
3. Face detection, model caching, and Grad-CAM heatmap generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from backend.services.face_detection import crop_face, detect_faces
from backend.services.image_processing import image_to_base64
from backend.utils.helpers import MODELS_DIR, ROOT_DIR, timed

logger = logging.getLogger("faceai.deepfake_service")

# Optional PyTorch import
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Model file paths
NPZ_PATH = (
    (MODELS_DIR / "deepfake_cnn_fft.npz")
    if (MODELS_DIR / "deepfake_cnn_fft.npz").exists()
    else (ROOT_DIR / "models" / "deepfake_cnn_fft.npz")
)
PTH_PATH = (
    (MODELS_DIR / "deepfake_cnn_fft.pth")
    if (MODELS_DIR / "deepfake_cnn_fft.pth").exists()
    else (ROOT_DIR / "models" / "deepfake_cnn_fft.pth")
)

_npz_weights: Optional[Dict[str, np.ndarray]] = None
_torch_model: Optional[Any] = None
_torch_device: Optional[str] = None


def _get_npz_weights() -> Optional[Dict[str, np.ndarray]]:
    global _npz_weights
    if _npz_weights is not None:
        return _npz_weights
    if NPZ_PATH.exists():
        try:
            _npz_weights = dict(np.load(NPZ_PATH))
            logger.info("Loaded lightweight NumPy weights from %s", NPZ_PATH)
            return _npz_weights
        except Exception as exc:
            logger.warning("Failed to load NPZ weights: %s", exc)
    return None


def _conv2d_np(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray], padding: int = 1) -> np.ndarray:
    C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant") if padding > 0 else x
    out = np.zeros((C_out, H, W), dtype=np.float32)
    for i in range(kH):
        for j in range(kW):
            out += np.tensordot(w[:, :, i, j], x_pad[:, i:i+H, j:j+W], axes=(1, 0))
    if b is not None:
        out += b[:, None, None]
    return out


def _maxpool2d_np(x: np.ndarray, k: int = 2) -> np.ndarray:
    C, H, W = x.shape
    return x.reshape(C, H // k, k, W // k, k).max(axis=(2, 4))


def _batchnorm_np(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    scale = (gamma / np.sqrt(var + eps))[:, None, None]
    shift = (beta - gamma * mean / np.sqrt(var + eps))[:, None, None]
    return x * scale + shift


def is_checkpoint_available() -> bool:
    return NPZ_PATH.exists() or PTH_PATH.exists()


def is_loaded() -> bool:
    return _npz_weights is not None or _torch_model is not None


def get_status() -> dict:
    avail = is_checkpoint_available()
    engine = "NumPy-Lightweight (Vercel Ready)" if NPZ_PATH.exists() else ("PyTorch" if HAS_TORCH else "untrained")
    return {
        "name": "deepfake",
        "checkpoint_available": avail,
        "mode": f"CNN+FFT ({engine})" if avail else "untrained",
        "loaded": is_loaded(),
        "detail": None if avail else "No trained checkpoint found at models/deepfake_cnn_fft.npz or .pth",
    }


def _cam_to_base64(cam: np.ndarray) -> str:
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(heatmap_rgb)
    return image_to_base64(pil, fmt="PNG")


def _analyze_numpy(image: Image.Image, weights: Dict[str, np.ndarray]) -> tuple[str, float, Optional[str]]:
    img_rgb = image.convert("RGB")
    img_resized = img_rgb.resize((224, 224), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    x_in = (arr / 255.0).transpose(2, 0, 1)

    # Spatial Branch
    s = _conv2d_np(x_in, weights["spatial.0.weight"], weights["spatial.0.bias"], 1)
    s = _batchnorm_np(s, weights["spatial.1.weight"], weights["spatial.1.bias"], weights["spatial.1.running_mean"], weights["spatial.1.running_var"])
    s = np.maximum(0, s)
    s = _maxpool2d_np(s, 2)

    s = _conv2d_np(s, weights["spatial.4.weight"], weights["spatial.4.bias"], 1)
    s = _batchnorm_np(s, weights["spatial.5.weight"], weights["spatial.5.bias"], weights["spatial.5.running_mean"], weights["spatial.5.running_var"])
    s = np.maximum(0, s)
    s = _maxpool2d_np(s, 2)

    s = _conv2d_np(s, weights["spatial.8.weight"], weights["spatial.8.bias"], 1)
    s = _batchnorm_np(s, weights["spatial.9.weight"], weights["spatial.9.bias"], weights["spatial.9.running_mean"], weights["spatial.9.running_var"])
    s = np.maximum(0, s)
    spatial_activation = s.mean(axis=0)  # (56, 56)
    s_feat = s.mean(axis=(1, 2))  # (128,)

    # Frequency Branch
    gray = x_in.mean(axis=0)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    f_in = np.repeat(mag[None, :, :], 3, axis=0)

    f = _conv2d_np(f_in, weights["frequency.0.weight"], weights["frequency.0.bias"], 1)
    f = np.maximum(0, f)
    f = _maxpool2d_np(f, 2)
    f = _conv2d_np(f, weights["frequency.3.weight"], weights["frequency.3.bias"], 1)
    f = np.maximum(0, f)
    f_feat = f.mean(axis=(1, 2))  # (32,)

    # Fusion
    fused = np.concatenate([s_feat, f_feat])

    # Classifier
    h = np.maximum(0, fused @ weights["classifier.0.weight"].T + weights["classifier.0.bias"])
    logits = h @ weights["classifier.3.weight"].T + weights["classifier.3.bias"]
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)

    idx = int(np.argmax(probs))
    label = "FAKE" if idx == 1 else "REAL"
    confidence = round(float(probs[idx] * 100), 2)

    # Grad-CAM heatmap interpolation to 224x224
    cam_resized = cv2.resize(spatial_activation, (224, 224), interpolation=cv2.INTER_LINEAR)
    c_min, c_max = cam_resized.min(), cam_resized.max()
    if c_max > c_min:
        cam_norm = ((cam_resized - c_min) / (c_max - c_min) * 255).astype(np.uint8)
    else:
        cam_norm = np.zeros((224, 224), dtype=np.uint8)

    grad_cam_b64 = _cam_to_base64(cam_norm)
    return label, confidence, grad_cam_b64


@timed("deepfake_analyze")
def analyze(image: Image.Image) -> Dict:
    """Run deepfake analysis on a PIL Image."""
    image = image.convert("RGB")

    faces = detect_faces(image)
    face_bbox = None
    face_detected = len(faces) > 0
    target_face = image
    if face_detected:
        face_bbox = max(faces, key=lambda f: f["w"] * f["h"])
        try:
            target_face = crop_face(image, face_bbox, padding=0.15)
        except Exception:
            target_face = image

    image_info = {"width": image.width, "height": image.height, "format": image.format}

    # 1. Prefer lightweight NumPy weights
    weights = _get_npz_weights()
    if weights is not None:
        label, confidence, grad_cam_b64 = _analyze_numpy(target_face, weights)
        return {
            "status": "ok",
            "prediction": label,
            "confidence": confidence,
            "face_detected": face_detected,
            "face_bbox": face_bbox,
            "grad_cam_base64": grad_cam_b64,
            "image_info": image_info,
            "message": None,
            "notice": "Probabilistic AI inference using CNN+FFT spatial-frequency fusion. Use as an interpretability aid.",
        }

    # 2. Check if PyTorch model is available
    if HAS_TORCH and PTH_PATH.exists():
        try:
            from backend.modules.deepfake import CNNFFTDetector
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CNNFFTDetector().to(device)
            model.load_state_dict(torch.load(PTH_PATH, map_location=device, weights_only=True))
            model.eval()

            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            idx = int(torch.argmax(probs).item())
            label = "FAKE" if idx == 1 else "REAL"
            confidence = round(float(probs[idx].item() * 100), 2)

            return {
                "status": "ok",
                "prediction": label,
                "confidence": confidence,
                "face_detected": face_detected,
                "face_bbox": face_bbox,
                "grad_cam_base64": None,
                "image_info": image_info,
                "message": None,
                "notice": "Inference generated by PyTorch CNN+FFT detector.",
            }
        except Exception as exc:
            logger.error("PyTorch inference error: %s", exc)

    return {
        "status": "model_not_ready",
        "prediction": None,
        "confidence": None,
        "face_detected": face_detected,
        "face_bbox": face_bbox,
        "grad_cam_base64": None,
        "image_info": image_info,
        "message": "Model weights not found at models/deepfake_cnn_fft.npz.",
        "notice": None,
    }
