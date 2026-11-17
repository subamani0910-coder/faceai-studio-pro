"""Shared image-processing utilities for the FaceAI Studio backend."""

from __future__ import annotations

import base64
import io
import logging
from typing import Dict, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger("faceai.image_processing")

MAX_BYTES = 25 * 1024 * 1024  # 25 MB

ALLOWED_FORMATS = {
    "JPEG", "JPG", "PNG", "WEBP", "BMP", "GIF", "TIFF", "TIF", "MPO", "PPM", "DIB", "ICO"
}

ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp",
    "image/gif", "image/tiff", "image/pjpeg", "image/x-png", "image/jfif",
    "application/octet-stream"
}


def load_image_safely(file_bytes: bytes) -> Image.Image:
    """Load and normalize any image format cleanly into an RGB Image with EXIF orientation corrected."""
    img = Image.open(io.BytesIO(file_bytes))
    
    # Correct EXIF rotation for smartphone photos
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    
    # Clean conversion to RGB
    if img.mode in ("RGBA", "LA"):
        # Alpha blend over clean white/neutral background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    elif img.mode != "RGB":
        return img.convert("RGB")
    

    return img


def validate_image(
    file_bytes: bytes,
    content_type: Optional[str] = None,
) -> Dict:
    """Validate raw image bytes and return a dict with metadata + errors list.

    Accepts all mainstream image formats (JPEG, PNG, WEBP, BMP, etc.).
    """
    errors: list[str] = []
    result: Dict = {
        "valid": True,
        "width": None,
        "height": None,
        "format": None,
        "file_size_bytes": len(file_bytes) if file_bytes else 0,
        "errors": errors,
    }

    if not file_bytes:
        errors.append("File is empty.")
        result["valid"] = False
        return result

    if len(file_bytes) > MAX_BYTES:
        errors.append(f"File exceeds maximum allowed size of {MAX_BYTES // (1024 * 1024)} MB.")
        result["valid"] = False
        return result

    try:
        raw_img = Image.open(io.BytesIO(file_bytes))
        fmt = (raw_img.format or "JPEG").upper()
        result["format"] = fmt
        result["width"] = raw_img.width
        result["height"] = raw_img.height

        if raw_img.width < 16 or raw_img.height < 16:
            errors.append("Image dimensions are too small (minimum 16×16 pixels).")
            result["valid"] = False
            return result

        if fmt not in ALLOWED_FORMATS:
            errors.append(f"Image format '{fmt}' is not supported. Please upload JPEG, PNG, WEBP, or BMP.")
            result["valid"] = False
            return result

    except (UnidentifiedImageError, Exception) as exc:
        errors.append(f"Could not decode image file: {type(exc).__name__}")
        result["valid"] = False
        return result

    return result


def resize_for_model(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize an image to target dimensions using high-quality Lanczos resampling."""
    return image.convert("RGB").resize(target_size, Image.LANCZOS)


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image to a base64 string (no data-URI prefix)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def base64_to_image(b64_str: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")
