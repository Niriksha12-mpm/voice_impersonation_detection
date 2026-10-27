import React from 'react';

export default function PredictionResult({ result }) {
  if (!result) return null;

  const { prediction, audio_metadata, acoustic_diagnostics, waveform_preview } = result;
  const isReal = prediction.label === 'REAL';
  const confidencePct = (prediction.confidence_score * 100).toFixed(1);

  return (
    <div className={`mt-8 glass-panel rounded-2xl p-6 sm:p-8 shadow-2xl transition-all duration-300 border ${
      isReal ? 'border-emerald-500/40 glow-emerald' : 'border-rose-500/40 glow-rose'
    }`}>
      {/* Primary Verdict Banner */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 border-b border-slate-800 gap-4">
        <div>
          <div className="text-xs uppercase font-mono tracking-wider text-slate-400 mb-1">
            Classification Verdict
          </div>
          <div className="flex items-center space-x-3">
            <span className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${
              isReal ? 'text-emerald-400' : 'text-rose-500'
            }`}>
              {prediction.label}
            </span>
            <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
              isReal ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
            }`}>
              {isReal ? 'Human Voice' : 'AI Synthetic / Deepfake'}
            </span>
          </div>
        </div>

        {/* Confidence Meter Badge */}
        <div className="bg-slate-900/90 px-5 py-3 rounded-xl border border-slate-800 text-right sm:text-right w-full sm:w-auto">
          <div className="text-xs text-slate-400">Confidence Score</div>
          <div className="text-2xl font-black font-mono text-white flex items-center justify-end space-x-2">
            <span>{confidencePct}%</span>
            <span className="text-xs font-normal text-slate-400">({prediction.confidence_score.toFixed(4)})</span>
          </div>
        </div>
      </div>

      {/* Probability Gauge Bar */}
      <div className="my-6">
        <div className="flex justify-between text-xs font-mono mb-1.5 text-slate-300">
          <span className="flex items-center space-x-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span>P(REAL): {(prediction.probability_real * 100).toFixed(1)}%</span>
          </span>
          <span className="flex items-center space-x-1">
            <span className="w-2 h-2 rounded-full bg-rose-500"></span>
            <span>P(FAKE): {(prediction.probability_fake * 100).toFixed(1)}%</span>
          </span>
        </div>
        <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden flex border border-slate-800">
          <div
            style={{ width: `${prediction.probability_real * 100}%` }}
            className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-full transition-all duration-700"
          ></div>
          <div
            style={{ width: `${prediction.probability_fake * 100}%` }}
            className="bg-gradient-to-r from-rose-500 to-rose-600 h-full transition-all duration-700"
          ></div>
        </div>
      </div>

      {/* Model Used & Mandatory Disclaimer */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400 font-mono uppercase mb-1">Model Evaluator</div>
          <div className="text-sm font-bold text-white flex items-center space-x-2">
            <span>⚙️</span>
            <span>{prediction.model_used}</span>
          </div>
          <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
            {prediction.model_description}
          </p>
        </div>

        <div className="bg-amber-500/10 p-4 rounded-xl border border-amber-500/25">
          <div className="text-xs text-amber-400 font-mono uppercase mb-1 flex items-center space-x-1">
            <span>ℹ️</span>
            <span>Confidence Disclaimer</span>
          </div>
          <p className="text-xs text-amber-200/90 leading-relaxed">
            {prediction.confidence_disclaimer} High or low confidence indicates statistical affinity to learned training distributions and should be cross-verified for critical biometric applications.
          </p>
        </div>
      </div>

      {/* Waveform Visualization SVG */}
      {waveform_preview && waveform_preview.length > 0 && (
        <div className="mb-6 bg-slate-900/80 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono uppercase text-slate-400">Standardized Normalized Waveform (16kHz)</span>
            <span className="text-[10px] text-slate-500 font-mono">120 Downsampled Points</span>
          </div>
          <div className="h-20 w-full flex items-center justify-between space-x-0.5 overflow-hidden">
            {waveform_preview.map((val, idx) => {
              const heightPct = Math.max(8, Math.min(100, Math.abs(val) * 100));
              return (
                <div
                  key={idx}
                  style={{ height: `${heightPct}%` }}
                  className={`w-1 rounded-full transition-all duration-300 ${
                    isReal ? 'bg-emerald-500/70 hover:bg-emerald-400' : 'bg-rose-500/70 hover:bg-rose-400'
                  }`}
                  title={`Point ${idx}: ${val}`}
                ></div>
              );
            })}
          </div>
        </div>
      )}

      {/* Audio Metadata and Forensic Diagnostics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-slate-800">
        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[11px] text-slate-400">Duration</div>
          <div className="text-sm font-bold font-mono text-white mt-0.5">
            {audio_metadata.duration_seconds}s
          </div>
          <div className="text-[10px] text-slate-500">{audio_metadata.num_segments_analyzed} segment(s) analyzed</div>
        </div>

        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[11px] text-slate-400">Spectral Centroid</div>
          <div className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
            {acoustic_diagnostics.spectral_centroid_hz} Hz
          </div>
          <div className="text-[10px] text-slate-500">Spectral center of mass</div>
        </div>

        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[11px] text-slate-400">Spectral Rolloff</div>
          <div className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
            {acoustic_diagnostics.spectral_rolloff_hz} Hz
          </div>
          <div className="text-[10px] text-slate-500">85% Energy Boundary</div>
        </div>

        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[11px] text-slate-400">Spectral Flatness</div>
          <div className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
            {acoustic_diagnostics.spectral_flatness}
          </div>
          <div className="text-[10px] text-slate-500">Tonality vs Noise measure</div>
        </div>
      </div>
    </div>
  );
}
