import React, { useState, useEffect } from 'react';

export default function MetricsDashboard({ comparisonData }) {
  const [history, setHistory] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/training-history')
      .then((res) => res.json())
      .then((data) => setHistory(data))
      .catch((err) => console.error('Error fetching training history:', err));
  }, []);

  if (!comparisonData) return null;

  const modelB = comparisonData.model_b_fine_tuned;
  const classWise = modelB.class_wise_metrics;
  const datasetBreakdown = modelB.dataset_accuracy_breakdown;

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800">
        <div className="text-xs uppercase font-mono tracking-wider text-cyan-400 mb-1">
          Quantitative Analysis
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white">
          Evaluation Metrics & Convergence Dashboard
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl leading-relaxed">
          Comprehensive breakdown of classification metrics, per-class sensitivity, training progression curves, and dataset-wise accuracy.
        </p>
      </div>

      {/* Class-wise Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* REAL Voice Class */}
        <div className="glass-card rounded-2xl p-6 border border-emerald-500/30">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-emerald-400"></span>
              <h3 className="text-lg font-bold text-white">Class: REAL (Human Speech)</h3>
            </div>
            <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2.5 py-1 rounded">
              Support: {classWise.real.support} samples
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 font-mono text-center">
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div className="text-[10px] text-slate-400">PRECISION</div>
              <div className="text-xl font-bold text-emerald-400 mt-1">{classWise.real.precision}%</div>
            </div>
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div className="text-[10px] text-slate-400">RECALL</div>
              <div className="text-xl font-bold text-emerald-400 mt-1">{classWise.real.recall}%</div>
            </div>
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div className="text-[10px] text-slate-400">F1-SCORE</div>
              <div className="text-xl font-bold text-emerald-400 mt-1">{classWise.real.f1_score}%</div>
            </div>
          </div>
        </div>

        {/* FAKE Voice Class */}
        <div className="glass-card rounded-2xl p-6 border border-rose-500/30">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-rose-500"></span>
              <h3 className="text-lg font-bold text-white">Class: FAKE (AI / Vocoded)</h3>
            </div>
            <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2.5 py-1 rounded">
              Support: {classWise.fake.support} samples
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 font-mono text-center">
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div className="text-[10px] text-slate-400">PRECISION</div>
              <div className="text-xl font-bold text-rose-400 mt-1">{classWise.fake.precision}%</div>
            </div>
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div className="text-[10px] text-slate-400">RECALL</div>
              <div className="text-xl font-bold text-rose-400 mt-1">{classWise.fake.recall}%</div>
            </div>
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div className="text-[10px] text-slate-400">F1-SCORE</div>
              <div className="text-xl font-bold text-rose-400 mt-1">{classWise.fake.f1_score}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset-wise Accuracy Breakdown */}
      {datasetBreakdown && Object.keys(datasetBreakdown).length > 0 && (
        <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800">
          <h3 className="text-lg font-bold text-white mb-4">
            Dataset-Wise Evaluation Breakdown (Test Set)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-xs font-mono text-slate-400 uppercase">
                  <th className="py-3 px-4">Dataset Source</th>
                  <th className="py-3 px-4">Total Test Samples</th>
                  <th className="py-3 px-4">Correct Predictions</th>
                  <th className="py-3 px-4 text-right">Accuracy Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-xs sm:text-sm">
                {Object.entries(datasetBreakdown).map(([dsName, stats]) => (
                  <tr key={dsName}>
                    <td className="py-3.5 px-4 font-sans font-medium text-slate-200">
                      {dsName}
                    </td>
                    <td className="py-3.5 px-4 text-slate-400">{stats.total_samples}</td>
                    <td className="py-3.5 px-4 text-slate-300">{stats.correct_samples}</td>
                    <td className="py-3.5 px-4 text-right">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                        stats.accuracy_percent >= 80 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-300'
                      }`}>
                        {stats.accuracy_percent}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model B Training Loss & Accuracy Progression Curve */}
      {history && (
        <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-2">
            <div>
              <div className="text-xs font-mono uppercase text-emerald-400">Optimization Dynamics</div>
              <h3 className="text-lg font-bold text-white">
                Model B Fine-Tuning Convergence ({history.total_epochs} Epochs)
              </h3>
            </div>
            <div className="text-xs font-mono text-slate-400 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800">
              Duration: {history.training_time_seconds}s • Final Loss: {history.loss[history.loss.length - 1]}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Loss History Visualizer */}
            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
              <div className="text-xs font-mono text-slate-400 mb-3 flex justify-between">
                <span>Loss Curve (BCEWithLogitsLoss)</span>
                <span>Initial: {history.loss[0]} → Final: {history.loss[history.loss.length - 1]}</span>
              </div>
              <div className="h-36 flex items-end space-x-1.5 pt-4">
                {history.loss.map((l, idx) => {
                  const maxL = Math.max(...history.loss, 1.0);
                  const heightPct = Math.max(10, Math.min(100, (l / maxL) * 100));
                  return (
                    <div
                      key={idx}
                      style={{ height: `${heightPct}%` }}
                      className="flex-1 bg-cyan-500/70 hover:bg-cyan-400 rounded-t transition-all group relative cursor-pointer"
                    >
                      <div className="opacity-0 group-hover:opacity-100 absolute -top-7 left-1/2 -translate-x-1/2 bg-slate-950 text-[10px] font-mono text-white px-1.5 py-0.5 rounded border border-slate-700 pointer-events-none z-10">
                        E{idx + 1}: {l}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-2">
                <span>Epoch 1</span>
                <span>Epoch {history.total_epochs}</span>
              </div>
            </div>

            {/* Accuracy History Visualizer */}
            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
              <div className="text-xs font-mono text-slate-400 mb-3 flex justify-between">
                <span>Training Accuracy Curve</span>
                <span>Initial: {history.train_accuracy[0]}% → Peak: {Math.max(...history.train_accuracy)}%</span>
              </div>
              <div className="h-36 flex items-end space-x-1.5 pt-4">
                {history.train_accuracy.map((acc, idx) => {
                  return (
                    <div
                      key={idx}
                      style={{ height: `${Math.max(10, acc)}%` }}
                      className="flex-1 bg-emerald-500/70 hover:bg-emerald-400 rounded-t transition-all group relative cursor-pointer"
                    >
                      <div className="opacity-0 group-hover:opacity-100 absolute -top-7 left-1/2 -translate-x-1/2 bg-slate-950 text-[10px] font-mono text-white px-1.5 py-0.5 rounded border border-slate-700 pointer-events-none z-10">
                        E{idx + 1}: {acc}%
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-2">
                <span>Epoch 1</span>
                <span>Epoch {history.total_epochs}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
