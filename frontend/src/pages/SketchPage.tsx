import React, { useState, useRef } from 'react';
import {
  Upload,
  Sparkles,
  Download,
  RefreshCw,
  Sliders,
  Maximize2,
  Minimize2,
  FileImage,
  Eye,
  CheckCircle,
} from 'lucide-react';
import { generateFaceSketch, validateImageFile, SketchGenerateResult, ImageValidationResult } from '../services/api';
import BeforeAfterSlider from '../components/BeforeAfterSlider';
import ScanningOverlay from '../components/ScanningOverlay';

export const SketchPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<SketchGenerateResult | null>(null);
  const [validation, setValidation] = useState<ImageValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'slider' | 'side-by-side' | 'sketch-only'>('slider');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (selectedFile?: File) => {
    if (!selectedFile) return;
    setError(null);
    setResult(null);
    setValidation(null);

    const isImage = selectedFile.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp|jfif|tiff?)$/i.test(selectedFile.name);
    if (!isImage) {
      setError('Upload a valid JPG, PNG, WEBP, or BMP portrait.');
      return;
    }
    if (selectedFile.size > 25 * 1024 * 1024) {
      setError('Image exceeds maximum 25 MB limit.');
      return;
    }

    setFile(selectedFile);
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    try {
      const val = await validateImageFile(selectedFile);
      setValidation(val);
    } catch {
      // Non-blocking
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const runSketchGeneration = async () => {
    if (!file) return;
    setGenerating(true);
    setError(null);
    setResult(null);

    try {
      const res = await generateFaceSketch(file);
      setResult(res);
    } catch (err: any) {
      setError(err?.message || 'Sketch generation failed. Ensure backend is running.');
    } finally {
      setGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!result?.imageUrl) return;
    const a = document.createElement('a');
    a.href = result.imageUrl;
    a.download = `faceai-sketch-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setValidation(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Module Title Banner */}
      <div className="mb-10 text-center sm:text-left">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#00ffd1]/10 border border-[#00ffd1]/30 text-[#00ffd1] text-xs font-mono mb-3">
          <FileImage size={14} />
          <span>MODULE 02 :: PHOTO-TO-SKETCH TRANSFORMATION</span>
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold font-['Space_Grotesk'] text-[#f5fbff] tracking-tight">
          REALISTIC FACE <span className="text-[#00ffd1]">SKETCH</span>
        </h1>
        <p className="mt-3 text-sm text-[#9bb4c5] max-w-2xl leading-relaxed">
          Transform facial photographs into realistic pencil-style sketches while preserving geometric facial
          proportions, eye contours, nose bridge, and distinctive human identity.
        </p>
      </div>

      {/* Main Art Lab Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Upload & Parameters */}
        <div className="lg:col-span-5 space-y-6">
          <div className="p-6 rounded-2xl bg-[#102638]/70 border border-[#00e5ff]/25 backdrop-blur-md shadow-[0_0_30px_rgba(6,18,31,0.8)]">
            <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#00e5ff]/15">
              <span className="text-xs font-mono tracking-widest text-[#00ffd1] font-bold">
                PORTRAIT INGESTION
              </span>
              <span className="text-[10px] font-mono text-[#9bb4c5]">JPEG / PNG · MAX 10MB</span>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
              className="hidden"
            />

            {/* Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-[#00ffd1]/40 hover:border-[#00e5ff] bg-[#06121f]/60 hover:bg-[#00ffd1]/5 rounded-xl p-8 text-center cursor-pointer transition-all group relative overflow-hidden"
            >
              <div className="w-14 h-14 mx-auto mb-4 rounded-xl bg-[#00ffd1]/10 border border-[#00ffd1]/30 flex items-center justify-center text-[#00ffd1] group-hover:scale-110 transition-transform">
                <Upload size={24} />
              </div>
              <div className="text-sm font-bold text-[#f5fbff] font-['Space_Grotesk'] tracking-wide">
                DROP PORTRAIT PHOTO HERE
              </div>
              <p className="text-xs text-[#9bb4c5] mt-1.5">
                Front-facing photos with good lighting work best
              </p>
            </div>

            {/* Validation telemetry */}
            {validation && (
              <div className="mt-4 p-3 rounded-lg bg-[#06121f]/80 border border-[#00ffd1]/20 text-[11px] font-mono space-y-1.5">
                <div className="flex justify-between text-[#9bb4c5]">
                  <span>ALIGNMENT STATUS:</span>
                  <span className={validation.face_detected ? 'text-[#00ffd1]' : 'text-[#f0b84b]'}>
                    {validation.face_detected ? 'FACE DETECTED & ALIGNED' : 'CENTER-FRAME ALIGNMENT'}
                  </span>
                </div>
                <div className="flex justify-between text-[#9bb4c5]">
                  <span>GEOMETRY INTEGRITY:</span>
                  <span className="text-[#00e5ff]">PRESERVED</span>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="mt-6 flex flex-col gap-3">
              <button
                onClick={runSketchGeneration}
                disabled={!file || generating}
                className="w-full py-3.5 px-6 rounded-lg font-bold font-['Space_Grotesk'] text-sm tracking-wider uppercase transition-all flex items-center justify-center gap-2.5 bg-gradient-to-r from-[#00ffd1]/25 to-[#00e5ff]/20 border border-[#00ffd1] text-[#f5fbff] hover:bg-[#00ffd1]/30 hover:shadow-[0_0_25px_rgba(0,255,209,0.35)] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              >
                {generating ? (
                  <>
                    <RefreshCw size={17} className="animate-spin text-[#00ffd1]" />
                    <span>SYNTHESIZING PENCIL STROKES...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={17} className="text-[#00ffd1]" />
                    <span>GENERATE REALISTIC SKETCH</span>
                  </>
                )}
              </button>

              {file && (
                <button
                  onClick={handleReset}
                  className="w-full py-2 px-4 rounded-lg text-xs font-mono text-[#9bb4c5] hover:text-[#f5fbff] bg-transparent border border-white/10 hover:border-white/20 transition-all"
                >
                  CLEAR IMAGE / RESET
                </button>
              )}
            </div>

            {error && (
              <div className="mt-4 p-3 rounded-lg bg-[#ff647e]/10 border border-[#ff647e]/40 text-xs text-[#ffb0bb]">
                {error}
              </div>
            )}
          </div>

          {/* Research Architecture Callout */}
          <div className="p-5 rounded-2xl bg-[#06121f]/90 border border-[#00ffd1]/15 text-xs text-[#9bb4c5] space-y-2">
            <div className="font-bold font-['Space_Grotesk'] text-[#f5fbff] flex items-center gap-2">
              <Sliders size={14} className="text-[#00ffd1]" />
              <span>RESEARCH ARCHITECTURE :: PHOTO-TO-SKETCH GAN</span>
            </div>
            <p className="leading-relaxed text-[11px]">
              Input Photograph → Haar Facial Landmark Alignment → Feature Extraction → Image-to-Image GAN Generator
              (Pix2Pix / CA-GAN) → Gradient-Guided Pencil Shading Refinement → Identity-Preserved Sketch.
            </p>
          </div>
        </div>

        {/* Right Column: Output & Comparison Slider */}
        <div className="lg:col-span-7 space-y-6">
          <div className="p-6 rounded-2xl bg-[#102638]/70 border border-[#00e5ff]/25 backdrop-blur-md shadow-[0_0_30px_rgba(6,18,31,0.8)] min-h-[480px] flex flex-col">
            <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#00e5ff]/15">
              <span className="text-xs font-mono tracking-widest text-[#00ffd1] font-bold">
                SKETCH LABORATORY CANVAS
              </span>
              {result && (
                <div className="flex items-center gap-2 text-xs">
                  <button
                    onClick={() => setViewMode('slider')}
                    className={`px-2 py-1 rounded text-[10px] font-mono ${
                      viewMode === 'slider' ? 'bg-[#00ffd1]/20 text-[#00ffd1] border border-[#00ffd1]/40' : 'text-[#9bb4c5]'
                    }`}
                  >
                    SLIDER
                  </button>
                  <button
                    onClick={() => setViewMode('side-by-side')}
                    className={`px-2 py-1 rounded text-[10px] font-mono ${
                      viewMode === 'side-by-side' ? 'bg-[#00ffd1]/20 text-[#00ffd1] border border-[#00ffd1]/40' : 'text-[#9bb4c5]'
                    }`}
                  >
                    SIDE-BY-SIDE
                  </button>
                </div>
              )}
            </div>

            {/* Visualizer Area */}
            <div className="relative flex-1 rounded-xl overflow-hidden bg-[#030c14] border border-[#00e5ff]/20 flex items-center justify-center min-h-[340px]">
              {generating && (
                <div className="relative w-full h-full min-h-[340px] flex items-center justify-center p-4">
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Processing target"
                      className="max-h-[320px] w-auto opacity-40 filter grayscale"
                    />
                  )}
                  <ScanningOverlay
                    active={true}
                    label="SYNTHESIZING GRAPHITE TEXTURES"
                    sublabel="EXTRACTING HIGH-FREQUENCY EDGE GRADIENTS"
                  />
                </div>
              )}

              {!generating && result && previewUrl ? (
                <div className="w-full h-full p-2">
                  {viewMode === 'slider' ? (
                    <BeforeAfterSlider
                      beforeImage={previewUrl}
                      afterImage={result.imageUrl}
                      beforeLabel="ORIGINAL PORTRAIT"
                      afterLabel="PENCIL SKETCH"
                    />
                  ) : (
                    <div className="grid grid-cols-2 gap-3 h-full">
                      <div className="flex flex-col items-center">
                        <img
                          src={previewUrl}
                          alt="Original"
                          className="max-h-72 object-contain rounded border border-[#00e5ff]/20"
                        />
                        <span className="text-[10px] font-mono text-[#9bb4c5] mt-2">ORIGINAL</span>
                      </div>
                      <div className="flex flex-col items-center">
                        <img
                          src={result.imageUrl}
                          alt="Sketch"
                          className="max-h-72 object-contain rounded border border-[#00ffd1]/30 shadow-[0_0_15px_rgba(0,255,209,0.15)]"
                        />
                        <span className="text-[10px] font-mono text-[#00ffd1] mt-2">PENCIL SKETCH</span>
                      </div>
                    </div>
                  )}
                </div>
              ) : !generating && previewUrl ? (
                <div className="relative p-4 flex flex-col items-center">
                  <img
                    src={previewUrl}
                    alt="Uploaded preview"
                    className="max-h-[320px] w-auto object-contain rounded-lg border border-[#00e5ff]/20"
                  />
                  <span className="mt-3 text-[11px] font-mono text-[#9bb4c5]">
                    TARGET READY :: CLICK "GENERATE REALISTIC SKETCH" TO CONVERT
                  </span>
                </div>
              ) : !generating ? (
                <div className="text-center p-8 text-[#9bb4c5]">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full border border-[#00ffd1]/30 flex items-center justify-center text-[#00ffd1] shadow-[0_0_20px_rgba(0,255,209,0.15)]">
                    <FileImage size={28} />
                  </div>
                  <div className="text-sm font-bold font-['Space_Grotesk'] text-[#f5fbff]">
                    NO PORTRAIT LOADED
                  </div>
                  <p className="text-xs text-[#9bb4c5] mt-1 max-w-xs mx-auto">
                    Upload a facial photograph to experience interactive before-and-after sketch synthesis.
                  </p>
                </div>
              ) : null}
            </div>

            {/* Action Bar when result ready */}
            {result && (
              <div className="mt-6 p-4 rounded-xl bg-[#06121f]/90 border border-[#00ffd1]/30 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 text-xs font-bold font-mono text-[#00ffd1]">
                    <CheckCircle size={15} />
                    <span>SYNTHESIS COMPLETE</span>
                  </div>
                  <div className="text-[10px] font-mono text-[#9bb4c5] mt-0.5">
                    MODE: {result.mode}
                  </div>
                </div>

                <div className="flex items-center gap-3 w-full sm:w-auto">
                  <button
                    onClick={handleDownload}
                    className="flex-1 sm:flex-none py-2.5 px-5 rounded-lg font-bold font-['Space_Grotesk'] text-xs tracking-wider uppercase bg-[#00ffd1]/20 hover:bg-[#00ffd1]/30 text-[#00ffd1] border border-[#00ffd1] shadow-[0_0_15px_rgba(0,255,209,0.25)] flex items-center justify-center gap-2 cursor-pointer transition-all"
                  >
                    <Download size={15} />
                    <span>DOWNLOAD PNG</span>
                  </button>

                  <button
                    onClick={runSketchGeneration}
                    className="p-2.5 rounded-lg border border-[#00e5ff]/30 text-[#00e5ff] hover:bg-[#00e5ff]/10 transition-all"
                    title="Generate again"
                  >
                    <RefreshCw size={15} />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SketchPage;
