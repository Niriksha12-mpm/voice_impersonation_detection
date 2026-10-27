import React, { useState, useRef } from 'react';

export default function AudioUploader({ onAnalyze, loading, selectedModel, setSelectedModel, error }) {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [presetActive, setPresetActive] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);

  const presets = [
    { id: 'real_lj', name: 'Real Human (LJSpeech)', type: 'REAL', label: 'Human Bonafide' },
    { id: 'real_yt', name: 'Real Human (YouTube)', type: 'REAL', label: 'Human Speech' },
    { id: 'fake_elevenlabs', name: 'ElevenLabs AI Clone', type: 'FAKE', label: 'Neural Clone' },
    { id: 'fake_wavefake', name: 'WaveFake Vocoder', type: 'FAKE', label: 'Vocoded Impersonation' },
  ];

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setAudioUrl(URL.createObjectURL(selected));
      setPresetActive(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      setFile(dropped);
      setAudioUrl(URL.createObjectURL(dropped));
      setPresetActive(null);
    }
  };

  const loadPreset = async (preset) => {
    setPresetActive(preset.id);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/sample-audio/${preset.id}`);
      if (!res.ok) throw new Error('Preset audio download failed');
      const blob = await res.blob();
      const ext = preset.id.includes('yt') || preset.id.includes('elevenlabs') ? 'flac' : 'wav';
      const presetFile = new File([blob], `${preset.id}_sample.${ext}`, { type: blob.type });
      setFile(presetFile);
      setAudioUrl(URL.createObjectURL(blob));
    } catch (err) {
      console.error('Error loading preset:', err);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    onAnalyze(file, selectedModel);
  };

  return (
    <div className="glass-panel rounded-2xl p-6 sm:p-8 shadow-2xl border border-slate-800">
      <div className="mb-6">
        <h2 className="text-xl sm:text-2xl font-bold text-white flex items-center space-x-2">
          <span>Upload Audio File</span>
          <span className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded-full border border-slate-700 font-normal">
            WAV, MP3, FLAC
          </span>
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Upload any speech recording to detect whether it is authentic human speech or an AI-generated/vocoded impersonation.
        </p>
      </div>

      {/* Preset Demo Quick-Selector */}
      <div className="mb-6 bg-slate-900/60 p-4 rounded-xl border border-slate-800/80">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center justify-between">
          <span>⚡ Or select a verified test recording:</span>
          <span className="text-[10px] text-slate-500 font-mono">1-Click Test</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {presets.map((p) => {
            const isSelected = presetActive === p.id;
            const isFake = p.type === 'FAKE';
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => loadPreset(p)}
                className={`p-2.5 rounded-lg text-left text-xs transition-all border ${
                  isSelected
                    ? isFake
                      ? 'bg-rose-500/20 border-rose-500/50 text-rose-300'
                      : 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                    : 'bg-slate-800/60 border-slate-700/60 hover:bg-slate-800 text-slate-300'
                }`}
              >
                <div className="font-semibold truncate">{p.name}</div>
                <div className="flex items-center space-x-1 mt-1 text-[10px] opacity-80">
                  <span className={`w-1.5 h-1.5 rounded-full ${isFake ? 'bg-rose-400' : 'bg-emerald-400'}`}></span>
                  <span>{p.label}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Drag & Drop File Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            dragOver
              ? 'border-emerald-400 bg-emerald-500/10 scale-[1.01]'
              : file
              ? 'border-emerald-500/40 bg-slate-900/80'
              : 'border-slate-700 hover:border-slate-500 bg-slate-900/40 hover:bg-slate-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-slate-800/80 flex items-center justify-center text-2xl shadow-inner">
            {file ? '🎵' : '📁'}
          </div>

          {file ? (
            <div>
              <p className="text-sm font-semibold text-emerald-400 truncate max-w-md mx-auto">
                {file.name}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                {(file.size / 1024).toFixed(1)} KB • Click or drop a different file to replace
              </p>
            </div>
          ) : (
            <div>
              <p className="text-sm font-medium text-slate-200">
                Click to browse or drag and drop audio file here
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Supports WAV, MP3, FLAC (16kHz-48kHz, mono or stereo)
              </p>
            </div>
          )}
        </div>

        {/* Audio Player if file loaded */}
        {audioUrl && (
          <div className="mt-4 p-3 bg-slate-900/90 rounded-xl border border-slate-800 flex items-center space-x-3">
            <span className="text-xs text-slate-400 font-mono">Audio Preview:</span>
            <audio
              ref={audioRef}
              src={audioUrl}
              controls
              className="w-full h-8 outline-none rounded"
            />
          </div>
        )}

        {/* Model Selector Bar */}
        <div className="mt-6 p-4 bg-slate-900/70 rounded-xl border border-slate-800">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block mb-2">
            Target Model Architecture:
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setSelectedModel('fine_tuned')}
              className={`p-3 rounded-lg text-left transition-all border ${
                selectedModel === 'fine_tuned'
                  ? 'bg-emerald-500/15 border-emerald-500 text-emerald-300 ring-1 ring-emerald-500/50'
                  : 'bg-slate-800/40 border-slate-700/60 text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-sm">Model B (Fine-Tuned)</span>
                <span className="text-[10px] uppercase font-bold bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded">
                  Recommended
                </span>
              </div>
              <p className="text-xs mt-1 opacity-80">
                84.6% Accuracy, 0% FNR. Fine-tuned on training split with vocoder artifact sensitivity.
              </p>
            </button>

            <button
              type="button"
              onClick={() => setSelectedModel('pretrained')}
              className={`p-3 rounded-lg text-left transition-all border ${
                selectedModel === 'pretrained'
                  ? 'bg-indigo-500/15 border-indigo-500 text-indigo-300 ring-1 ring-indigo-500/50'
                  : 'bg-slate-800/40 border-slate-700/60 text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-sm">Model A (Pretrained)</span>
                <span className="text-[10px] uppercase font-semibold bg-slate-700 text-slate-300 px-1.5 py-0.5 rounded">
                  Baseline
                </span>
              </div>
              <p className="text-xs mt-1 opacity-80">
                46.2% Accuracy. Un-tuned out-of-the-box baseline for fair comparative benchmarking.
              </p>
            </button>
          </div>
        </div>

        {/* Error message if any */}
        {error && (
          <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-2">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Analyze Button */}
        <div className="mt-6 flex justify-end">
          <button
            type="submit"
            disabled={!file || loading}
            className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-bold text-sm transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg ${
              !file || loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:scale-[1.01]'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-slate-950" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Analyzing Acoustic Features...</span>
              </>
            ) : (
              <>
                <span>Run Impersonation Detection</span>
                <span>→</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
