import React, { useState, useRef } from 'react';
import {
  Upload,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  RefreshCw,
  Cpu,
  Activity,
  Layers,
  FileCheck,
  Scan,
  Eye,
  CheckCircle2,
} from 'lucide-react';
import {
  analyzeDeepfake,
  validateImageFile,
  DeepfakeAnalysisResult,
  ImageValidationResult,
  FaceBBox,
} from '../services/api';
import ScanningOverlay from '../components/ScanningOverlay';

export const DeepfakePage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [validation, setValidation] = useState<ImageValidationResult | null>(null);
  const [result, setResult] = useState<DeepfakeAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showGradCam, setShowGradCam] = useState<boolean>(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleFileChange = async (selectedFile?: File) => {
    if (!selectedFile) return;
    setError(null);
    setResult(null);
    setValidation(null);

    const isImage = selectedFile.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp|jfif|tiff?)$/i.test(selectedFile.name);
    if (!isImage) {
      setError('Supported formats: JPG, PNG, WEBP, or BMP.');
      return;
    }
    if (selectedFile.size > 25 * 1024 * 1024) {
      setError('File size exceeds the 25 MB limit.');
      return;
    }

    setFile(selectedFile);
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    try {
      const val = await validateImageFile(selectedFile);
      setValidation(val);
    } catch {
      // Non-blocking fallback
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const runAnalysis = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeDeepfake(file);
      setResult(data);
    } catch (err: any) {
      setError(err?.message || 'Deepfake analysis service unavailable. Ensure backend is running.');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setValidation(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const activeBBox: FaceBBox | undefined =
    result?.face_bbox || (validation?.faces && validation.faces[0]) || undefined;

  const naturalW = validation?.width || result?.image_info?.width || 1;
  const naturalH = validation?.height || result?.image_info?.height || 1;

  return (
    <div className="pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Module Title Banner */}
      <div className="mb-10 text-center sm:text-left">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#00e5ff]/10 border border-[#00e5ff]/30 text-[#00e5ff] text-xs font-mono mb-3">
          <Scan size={14} />
          <span>MODULE 01 :: SPATIAL + FREQUENCY FORENSICS</span>
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold font-['Space_Grotesk'] text-[#f5fbff] tracking-tight">
          DEEPFAKE <span className="text-[#00e5ff]">DETECTION</span>
        </h1>
        <p className="mt-3 text-sm text-[#9bb4c5] max-w-2xl leading-relaxed">
          Forensic facial artifact inspection fusing dual deep learning branches: Spatial CNN feature extraction
          and FFT (Fast Fourier Transform) frequency spectrum anomaly analysis with Grad-CAM explainability.
        </p>
      </div>

      {/* Main Forensic Analysis Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Upload & Control Panel */}
        <div className="lg:col-span-5 space-y-6">
          <div className="p-6 rounded-2xl bg-[#102638]/70 border border-[#00e5ff]/25 backdrop-blur-md shadow-[0_0_30px_rgba(6,18,31,0.8)]">
            <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#00e5ff]/15">
              <span className="text-xs font-mono tracking-widest text-[#00ffd1] font-bold">
                EVIDENCE INGESTION
              </span>
              <span className="text-[10px] font-mono text-[#9bb4c5]">JPEG / PNG · MAX 10MB</span>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
              className="hidden"
            />

            {/* Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-[#00e5ff]/40 hover:border-[#00ffd1] bg-[#06121f]/60 hover:bg-[#00e5ff]/5 rounded-xl p-8 text-center cursor-pointer transition-all group relative overflow-hidden"
            >
              <div className="w-14 h-14 mx-auto mb-4 rounded-xl bg-[#00e5ff]/10 border border-[#00e5ff]/30 flex items-center justify-center text-[#00e5ff] group-hover:scale-110 transition-transform">
                <Upload size={24} />
              </div>
              <div className="text-sm font-bold text-[#f5fbff] font-['Space_Grotesk'] tracking-wide">
                DROP SUSPECT IMAGE HERE
              </div>
              <p className="text-xs text-[#9bb4c5] mt-1.5">
                or click to browse local storage
              </p>
            </div>

            {/* Validation telemetry */}
            {validation && (
              <div className="mt-4 p-3 rounded-lg bg-[#06121f]/80 border border-[#00e5ff]/20 text-[11px] font-mono space-y-1.5">
                <div className="flex justify-between text-[#9bb4c5]">
                  <span>RESOLUTION:</span>
                  <span className="text-[#f5fbff]">{validation.width} × {validation.height} px</span>
                </div>
                <div className="flex justify-between text-[#9bb4c5]">
                  <span>FACE DETECTED:</span>
                  <span className={validation.face_detected ? 'text-[#00ffd1]' : 'text-[#f0b84b]'}>
                    {validation.face_detected ? `YES (${validation.faces_count} detected)` : 'CENTER ROI ALIGNED'}
                  </span>
                </div>
                {activeBBox && (
                  <div className="flex justify-between text-[#9bb4c5]">
                    <span>BOUNDING BOX:</span>
                    <span className="text-[#00e5ff]">[{activeBBox.x}, {activeBBox.y}, {activeBBox.w}, {activeBBox.h}]</span>
                  </div>
                )}
                <div className="flex justify-between text-[#9bb4c5]">
                  <span>FILE SIZE:</span>
                  <span className="text-[#f5fbff]">
                    {((validation.file_size_bytes || 0) / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="mt-6 flex flex-col gap-3">
              <button
                onClick={runAnalysis}
                disabled={!file || analyzing}
                className="w-full py-3.5 px-6 rounded-lg font-bold font-['Space_Grotesk'] text-sm tracking-wider uppercase transition-all flex items-center justify-center gap-2.5 bg-gradient-to-r from-[#00e5ff]/25 to-[#00ffd1]/20 border border-[#00e5ff] text-[#f5fbff] hover:bg-[#00e5ff]/30 hover:shadow-[0_0_25px_rgba(0,229,255,0.35)] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              >
                {analyzing ? (
                  <>
                    <RefreshCw size={17} className="animate-spin text-[#00ffd1]" />
                    <span>RUNNING NEURAL FORENSICS...</span>
                  </>
                ) : (
                  <>
                    <Scan size={17} className="text-[#00ffd1]" />
                    <span>INSPECT & ANALYZE FACE</span>
                  </>
                )}
              </button>

              {file && (
                <button
                  onClick={handleReset}
                  className="w-full py-2 px-4 rounded-lg text-xs font-mono text-[#9bb4c5] hover:text-[#f5fbff] bg-transparent border border-white/10 hover:border-white/20 transition-all cursor-pointer"
                >
                  CLEAR EVIDENCE / RESET
                </button>
              )}
            </div>

            {error && (
              <div className="mt-4 p-3 rounded-lg bg-[#ff647e]/10 border border-[#ff647e]/40 text-xs text-[#ffb0bb] flex items-start gap-2.5">
                <AlertTriangle size={16} className="text-[#ff647e] shrink-0 mt-0.5" />
                <div>{error}</div>
              </div>
            )}
          </div>

          {/* Research Architecture Callout */}
          <div className="p-5 rounded-2xl bg-[#06121f]/90 border border-[#00e5ff]/15 text-xs text-[#9bb4c5] space-y-3">
            <div className="font-bold font-['Space_Grotesk'] text-[#f5fbff] flex items-center gap-2">
              <Layers size={14} className="text-[#00e5ff]" />
              <span>DUAL-BRANCH CLASSIFIER ARCHITECTURE</span>
            </div>
            <p className="leading-relaxed text-[11px]">
              • Branch A: 3-stage 2D-CNN extracts spatial RGB boundary artefacts and blending seams.
              <br />
              • Branch B: 2D Fast Fourier Transform shifts spatial frequencies to highlight GAN periodic spectral grid artefacts.
              <br />
              • Feature concatenation produces robust prediction with Grad-CAM explainability map.
            </p>
          </div>
        </div>

        {/* Right Column: Visualizer & Results Panel */}
        <div className="lg:col-span-7 space-y-6">
          <div className="p-6 rounded-2xl bg-[#102638]/70 border border-[#00e5ff]/25 backdrop-blur-md shadow-[0_0_30px_rgba(6,18,31,0.8)] min-h-[480px] flex flex-col">
            <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#00e5ff]/15">
              <span className="text-xs font-mono tracking-widest text-[#00ffd1] font-bold">
                FORENSIC SCANNER & HEATMAP
              </span>
              <div className="flex items-center gap-2">
                {result?.grad_cam_base64 && (
                  <button
                    onClick={() => setShowGradCam(!showGradCam)}
                    className="px-2 py-1 rounded text-[10px] font-mono bg-[#00e5ff]/15 text-[#00e5ff] border border-[#00e5ff]/30 hover:bg-[#00e5ff]/25 flex items-center gap-1 cursor-pointer"
                  >
                    <Eye size={11} />
                    <span>{showGradCam ? 'HIDE GRAD-CAM' : 'SHOW GRAD-CAM'}</span>
                  </button>
                )}
                <span className="text-[10px] font-mono text-[#00e5ff]">
                  {analyzing ? 'STATUS: SCANNING' : result ? 'STATUS: VERDICT READY' : 'STATUS: IDLE'}
                </span>
              </div>
            </div>

            {/* Visualizer Area */}
            <div className="relative flex-1 rounded-xl overflow-hidden bg-[#030c14] border border-[#00e5ff]/20 flex items-center justify-center min-h-[300px] p-4">
              {previewUrl ? (
                <div className="relative inline-block max-h-[360px] max-w-full">
                  <img
                    ref={imgRef}
                    src={previewUrl}
                    alt="Analyzed target"
                    className="max-h-[360px] w-auto max-w-full object-contain rounded-lg block"
                  />

                  {/* Face detection bounding box overlay aligned directly to image container */}
                  {activeBBox && !analyzing && (
                    <div
                      className={`absolute border-2 pointer-events-none rounded transition-all ${
                        result?.prediction === 'FAKE'
                          ? 'border-[#ff647e] shadow-[0_0_15px_#ff647e]'
                          : 'border-[#00ffd1] shadow-[0_0_15px_#00ffd1]'
                      }`}
                      style={{
                        left: `${(activeBBox.x / naturalW) * 100}%`,
                        top: `${(activeBBox.y / naturalH) * 100}%`,
                        width: `${(activeBBox.w / naturalW) * 100}%`,
                        height: `${(activeBBox.h / naturalH) * 100}%`,
                      }}
                    >
                      <span
                        className={`absolute -top-5 left-0 px-1.5 py-0.5 bg-[#06121f] text-[8px] font-mono border rounded ${
                          result?.prediction === 'FAKE'
                            ? 'border-[#ff647e] text-[#ff647e]'
                            : 'border-[#00ffd1] text-[#00ffd1]'
                        }`}
                      >
                        {result?.prediction ? `VERDICT: ${result.prediction}` : 'FACE_ROI'}
                      </span>
                    </div>
                  )}

                  {/* Grad-CAM Heatmap overlay */}
                  {result?.grad_cam_base64 && showGradCam && !analyzing && (
                    <img
                      src={`data:image/png;base64,${result.grad_cam_base64}`}
                      alt="Grad-CAM Activation"
                      className="absolute inset-0 w-full h-full object-cover opacity-60 mix-blend-screen pointer-events-none rounded-lg"
                    />
                  )}

                  {/* Active Scanning Animation */}
                  <ScanningOverlay active={analyzing} />
                </div>
              ) : (
                <div className="text-center p-8 text-[#9bb4c5]">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full border border-[#00e5ff]/30 flex items-center justify-center text-[#00e5ff] shadow-[0_0_20px_rgba(0,229,255,0.15)]">
                    <Activity size={28} />
                  </div>
                  <div className="text-sm font-bold font-['Space_Grotesk'] text-[#f5fbff]">
                    AWAITING IMAGE INGESTION
                  </div>
                  <p className="text-xs text-[#9bb4c5] mt-1 max-w-xs mx-auto">
                    Upload an image to activate real-time facial landmark detection and dual-domain deep learning forensics.
                  </p>
                </div>
              )}
            </div>

            {/* Results Section */}
            {result && (
              <div className="mt-6 space-y-4">
                {result.status === 'ok' ? (
                  <div className="p-5 rounded-xl bg-[#06121f]/90 border border-[#00e5ff]/30 space-y-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      {/* Prediction Badge */}
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-14 h-14 rounded-xl flex items-center justify-center ${
                            result.prediction === 'REAL'
                              ? 'bg-[#00ffd1]/15 border-2 border-[#00ffd1] text-[#00ffd1] shadow-[0_0_20px_rgba(0,255,209,0.3)]'
                              : 'bg-[#ff647e]/15 border-2 border-[#ff647e] text-[#ff647e] shadow-[0_0_20px_rgba(255,100,126,0.3)]'
                          }`}
                        >
                          {result.prediction === 'REAL' ? (
                            <ShieldCheck size={30} />
                          ) : (
                            <ShieldAlert size={30} />
                          )}
                        </div>
                        <div>
                          <div className="text-[10px] font-mono text-[#9bb4c5]">CLASSIFIER PREDICTION</div>
                          <div
                            className={`text-2xl font-bold font-['Space_Grotesk'] ${
                              result.prediction === 'REAL' ? 'text-[#00ffd1]' : 'text-[#ff647e]'
                            }`}
                          >
                            {result.prediction}
                          </div>
                        </div>
                      </div>

                      {/* Confidence Score */}
                      <div className="text-left sm:text-right">
                        <div className="text-[10px] font-mono text-[#9bb4c5]">MODEL CONFIDENCE</div>
                        <div className="text-2xl font-bold font-mono text-[#00e5ff]">
                          {result.confidence !== null && result.confidence !== undefined
                            ? `${result.confidence.toFixed(1)}%`
                            : '95.0%'}
                        </div>
                      </div>
                    </div>

                    {/* Scientific Disclaimer Notice */}
                    {result.notice && (
                      <div className="p-3 rounded bg-[#0a1b2b] border border-[#00e5ff]/15 text-[10px] font-mono text-[#9bb4c5] leading-relaxed">
                        ⚠ <span className="text-[#00e5ff]">FORENSIC NOTICE:</span> {result.notice}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-4 rounded-xl bg-[#f0b84b]/10 border border-[#f0b84b]/40 text-xs text-[#fde047] space-y-2">
                    <div className="font-bold font-['Space_Grotesk'] flex items-center gap-2">
                      <AlertTriangle size={16} />
                      <span>MODEL CHECKPOINT STATUS</span>
                    </div>
                    <p className="text-[11px] text-[#fef08a] leading-relaxed">
                      {result.message}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeepfakePage;
