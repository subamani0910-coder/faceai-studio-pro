"""Pydantic response models for all FaceAI Studio API endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["FACEAI STUDIO API"])
    version: str = Field(..., examples=["2.0.0"])


# ── Model Status ───────────────────────────────────────────────────────
class SingleModelStatus(BaseModel):
    name: str
    checkpoint_available: bool
    mode: str = Field(..., description="Current operating mode")
    loaded: bool = Field(False, description="Whether the model is loaded in memory")
    detail: Optional[str] = None


class ModelStatusResponse(BaseModel):
    deepfake: SingleModelStatus
    sketch: SingleModelStatus
    text_to_face: SingleModelStatus


# ── Deepfake ───────────────────────────────────────────────────────────
class FaceBBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    confidence: Optional[float] = None
    landmarks: Optional[Dict[str, Any]] = None


class ImageInfo(BaseModel):
    width: int
    height: int
    format: Optional[str] = None


class DeepfakeResult(BaseModel):
    status: str = Field(..., description="'ok' or 'model_not_ready'")
    prediction: Optional[str] = Field(None, description="REAL or FAKE")
    confidence: Optional[float] = Field(None, ge=0, le=100)
    face_detected: bool = False
    face_bbox: Optional[FaceBBox] = None
    grad_cam_base64: Optional[str] = Field(None, description="Base64 Grad-CAM heatmap overlay")
    image_info: Optional[ImageInfo] = None
    message: Optional[str] = None
    notice: Optional[str] = None


# ── Sketch ─────────────────────────────────────────────────────────────
class SketchResult(BaseModel):
    status: str
    mode: str = Field(..., description="Generation mode used")
    face_detected: bool = False
    message: Optional[str] = None
    # The actual image is returned as a StreamingResponse,
    # so this model is used only for JSON error / info responses.


# ── Text-to-Face ──────────────────────────────────────────────────────
class TextToFaceResult(BaseModel):
    status: str
    device: Optional[str] = None
    label: str = "AI-GENERATED - FICTIONAL FACE"
    message: Optional[str] = None


# ── Validation ─────────────────────────────────────────────────────────
class ValidationResult(BaseModel):
    valid: bool
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    file_size_bytes: Optional[int] = None
    face_detected: bool = False
    faces_count: int = 0
    faces: List[FaceBBox] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ── Error ──────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
    request_id: Optional[str] = None
