// API Client for FACEAI STUDIO Backend
// Uses relative /api path (proxied by Vite or rewritten by Vercel) with intelligent local port discovery
let RAW_URL = import.meta.env.VITE_API_URL || '';
let API_BASE = RAW_URL.replace(/\/+$/, '');

export function getApiBase(): string {
  return API_BASE;
}

export function setApiBase(url: string) {
  API_BASE = url.replace(/\/+$/, '');
}

export interface FaceBBox {
  x: number;
  y: number;
  w: number;
  h: number;
  confidence?: number;
  landmarks?: Record<string, [number, number]>;
}

export interface ModelStatusItem {
  name: string;
  checkpoint_available: boolean;
  mode: string;
  loaded: boolean;
  detail?: string | null;
}

export interface ModelStatusResponse {
  deepfake: ModelStatusItem;
  sketch: ModelStatusItem;
  text_to_face: ModelStatusItem;
}

export interface ImageValidationResult {
  valid: boolean;
  width?: number;
  height?: number;
  format?: string;
  file_size_bytes?: number;
  face_detected: boolean;
  faces_count: number;
  faces: FaceBBox[];
  errors: string[];
}

export interface DeepfakeAnalysisResult {
  status: 'ok' | 'model_not_ready';
  prediction?: 'REAL' | 'FAKE' | null;
  confidence?: number | null;
  face_detected: boolean;
  face_bbox?: FaceBBox | null;
  grad_cam_base64?: string | null;
  image_info?: { width: number; height: number; format?: string } | null;
  message?: string | null;
  notice?: string | null;
}

export interface SketchGenerateResult {
  blob: Blob;
  imageUrl: string;
  mode: string;
}

export interface TextToFaceResult {
  blob: Blob;
  imageUrl: string;
  label: string;
  device: string;
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
    if (res.ok) return true;
  } catch {
    // Relative request failed; check fallback below
  }

  // Local auto-discovery: if running on localhost and relative fetch failed, try direct FastAPI port 8000
  if (!API_BASE && typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    try {
      const fallbackRes = await fetch(`http://127.0.0.1:8000/api/health`, { method: 'GET' });
      if (fallbackRes.ok) {
        API_BASE = 'http://127.0.0.1:8000';
        return true;
      }
    } catch {
      // Direct connection also offline
    }
  }

  return false;
}

export async function getModelStatus(): Promise<ModelStatusResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/api/models/status`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function validateImageFile(file: File): Promise<ImageValidationResult> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/image/validate`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to validate image format.');
  }
  return await res.json();
}

export async function analyzeDeepfake(file: File): Promise<DeepfakeAnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/deepfake/analyze`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Deepfake analysis failed on backend.');
  }
  return await res.json();
}

/**
 * Universal predict function matching the standard API endpoint:
 * fetch(`${API_URL}/predict`, { method: "POST", body: formData })
 */
export async function predict(formData: FormData): Promise<DeepfakeAnalysisResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Predict endpoint request failed.');
  }
  return await res.json();
}

export async function generateFaceSketch(file: File): Promise<SketchGenerateResult> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/sketch/generate`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Face sketch generation failed.');
  }
  const mode = res.headers.get('X-FaceAI-Mode') || 'Pencil sketch';
  const blob = await res.blob();
  const imageUrl = URL.createObjectURL(blob);
  return { blob, imageUrl, mode };
}

export async function generateTextToFace(params: {
  prompt: string;
  modelId?: string;
  negativePrompt?: string;
  steps?: number;
  guidance?: number;
  seed?: number | null;
  width?: number;
  height?: number;
}): Promise<TextToFaceResult> {
  const formData = new FormData();
  formData.append('prompt', params.prompt);
  formData.append('model_id', params.modelId || 'runwayml/stable-diffusion-v1-5');
  if (params.negativePrompt) {
    formData.append('negative_prompt', params.negativePrompt);
  }
  formData.append('steps', String(params.steps ?? 25));
  formData.append('guidance', String(params.guidance ?? 7.0));
  if (params.seed !== undefined && params.seed !== null) {
    formData.append('seed', String(params.seed));
  }
  formData.append('width', String(params.width ?? 512));
  formData.append('height', String(params.height ?? 512));

  const res = await fetch(`${API_BASE}/api/text-to-face/generate`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Text-to-face synthesis failed.');
  }
  const label = res.headers.get('X-FaceAI-Label') || 'AI-GENERATED - FICTIONAL FACE';
  const device = res.headers.get('X-FaceAI-Device') || 'cpu';
  const blob = await res.blob();
  const imageUrl = URL.createObjectURL(blob);
  return { blob, imageUrl, label, device };
}
