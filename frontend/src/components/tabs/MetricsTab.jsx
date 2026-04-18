import useAppStore from '../../store/appStore'

function toDisplayValue(value) {
  if (value == null) return '—'
  if (typeof value === 'number' || typeof value === 'string') return value
  return '—'
}

function StatCard({ label, value, unit = '', color = '#f0f0f0' }) {
  return (
    <div className="amoled-panel p-4 flex flex-col gap-1 min-w-[120px]">
      <span className="text-[10px] uppercase tracking-widest text-[#555]">{label}</span>
      <span className="text-2xl font-bold tabular-nums" style={{ color }}>
        {toDisplayValue(value)}<span className="text-xs font-normal text-[#555] ml-1">{unit}</span>
      </span>
    </div>
  )
}

function PipelineBar({ stats }) {
  if (!stats) return null
  const items = [
    { label: 'YOLO',     value: stats.yolo_n    ?? 0, color: '#ffab00' },
    { label: 'PP raw',   value: stats.pp_raw_n  ?? 0, color: '#2979ff' },
    { label: 'PP gated', value: stats.pp_gated_n ?? 0, color: '#2979ff' },
    { label: 'OBB',      value: stats.obb_n     ?? 0, color: '#00e676' },
    { label: 'Final',    value: stats.final_n   ?? 0, color: '#00e676' },
  ]
  return (
    <div className="amoled-panel p-4">
      <div className="text-[10px] uppercase tracking-widest text-[#555] mb-3">Pipeline stages</div>
      <div className="flex items-end gap-4">
        {items.map((item, i) => (
          <div key={i} className="flex flex-col items-center gap-1">
            <span className="text-xl font-bold tabular-nums" style={{ color: item.color }}>{item.value}</span>
            <span className="text-[9px] text-[#555] uppercase tracking-wider">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function MetricsSection({ metrics }) {
  if (!metrics) return null

  const summary = metrics.summary ?? metrics
  const precision = summary.precision
  const recall = summary.recall
  const f1 = summary.f1 ?? (
    precision != null && recall != null && (precision + recall) > 0
      ? (2 * precision * recall) / (precision + recall)
      : null
  )
  const iouThreshold = summary.iou_threshold ?? 0.3

  const tp = summary.true_positives
    ?? summary.matched
    ?? (Array.isArray(metrics.matched) ? metrics.matched.length : null)
  const fp = summary.false_positives
    ?? (Array.isArray(metrics.false_positives) ? metrics.false_positives.length : null)
  const fn = summary.false_negatives
    ?? (Array.isArray(metrics.false_negatives) ? metrics.false_negatives.length : null)

  return (
    <div className="amoled-panel p-4">
      <div className="text-[10px] uppercase tracking-widest text-[#555] mb-3">
        GT Evaluation — IoU@{iouThreshold}
      </div>
      <div className="flex gap-4 flex-wrap">
        <StatCard label="Precision" value={precision != null ? (precision * 100).toFixed(1) : null} unit="%" color="#00e676" />
        <StatCard label="Recall"    value={recall != null ? (recall * 100).toFixed(1) : null}       unit="%" color="#2979ff" />
        <StatCard label="F1"        value={f1 != null ? (f1 * 100).toFixed(1) : null}               unit="%" color="#ffab00" />
        <StatCard label="TP" value={tp} color="#00e676" />
        <StatCard label="FP" value={fp} color="#ff3d71" />
        <StatCard label="FN" value={fn} color="#ffab00" />
      </div>
    </div>
  )
}

export default function MetricsTab() {
  const { result } = useAppStore()

  if (!result) {
    return (
      <div className="flex-1 flex items-center justify-center text-[#555]">
        <span className="text-sm">No data yet — run inference first</span>
      </div>
    )
  }

  const { inference_time_ms, num_points, pipeline_stats, metrics, detections = [] } = result

  const classes = {}
  for (const d of detections) {
    const cls = d?.label ?? d?.class ?? 'unknown'
    classes[cls] = (classes[cls] || 0) + 1
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
      {/* Quick stats */}
      <div className="flex gap-3 flex-wrap">
        <StatCard label="Inference"   value={typeof inference_time_ms === 'number' ? inference_time_ms : null} unit="ms"  color="#00e676" />
        <StatCard label="LiDAR pts"   value={(num_points ?? 0).toLocaleString()} color="#2979ff" />
        <StatCard label="Detections"  value={Array.isArray(detections) ? detections.length : 0} color="#ffab00" />
        {Object.entries(classes).map(([cls, n]) => (
          <StatCard key={cls} label={cls} value={n} color="#f0f0f0" />
        ))}
      </div>

      <PipelineBar stats={pipeline_stats} />

      {metrics && <MetricsSection metrics={metrics} />}

      {!metrics && (
        <div className="amoled-panel p-4 text-[#555] text-xs">
          Upload a ground-truth label file (.txt) to see precision / recall metrics.
        </div>
      )}
    </div>
  )
}
