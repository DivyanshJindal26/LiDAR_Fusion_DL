import { useState } from 'react'
import useAppStore from '../store/appStore'
import { ragQuery } from '../api/inferApi'

const CLASS_COLORS = {
  car:        '#2979ff',
  pedestrian: '#00e676',
  cyclist:    '#ffab00',
  van:        '#e040fb',
  truck:      '#ff3d71',
}

function tierBadge(tier) {
  if (tier === 'HIGH') return 'text-[#00e676] border-[#00e676]/30 bg-[#00e676]/10'
  if (tier === 'MED')  return 'text-[#ffab00] border-[#ffab00]/30 bg-[#ffab00]/10'
  return 'text-[#555] border-white/10 bg-white/5'
}

function objectDistance(a, b) {
  const [x1, y1, z1] = a.center
  const [x2, y2, z2] = b.center
  return Math.hypot(x2 - x1, y2 - y1, z2 - z1).toFixed(2)
}

export default function DetectionSidebar() {
  const {
    result, confidenceThreshold, setConfidenceThreshold,
    selectedObjects, toggleSelectedObject, clearSelectedObjects,
    ragQuery: ragQueryText, setRagQuery, ragLoading, setRagLoading, ragResult, setRagResult,
  } = useAppStore()

  const [collapsed, setCollapsed] = useState(false)

  const dets = (result?.detections || []).filter(
    (d) => (d.score ?? 1) >= confidenceThreshold
  )

  const handleRag = async () => {
    if (!ragQueryText.trim() || ragLoading) return
    setRagLoading(true)
    try {
      const res = await ragQuery(ragQueryText)
      setRagResult(res)
    } catch (e) {
      setRagResult({ answer: `Error: ${e.message}`, matches: [] })
    } finally {
      setRagLoading(false)
    }
  }

  if (collapsed) {
    return (
      <div className="w-8 border-l border-white/[0.06] bg-[#0a0a0a] flex flex-col items-center pt-3">
        <button
          onClick={() => setCollapsed(false)}
          className="text-[#555] hover:text-[#f0f0f0] text-xs"
          title="Expand sidebar"
        >
          ‹
        </button>
      </div>
    )
  }

  return (
    <div className="w-64 border-l border-white/[0.06] bg-[#0a0a0a] flex flex-col flex-shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
        <span className="text-[10px] uppercase tracking-widest text-[#555]">
          Detections {dets.length > 0 && `· ${dets.length}`}
        </span>
        <button
          onClick={() => setCollapsed(true)}
          className="text-[#555] hover:text-[#f0f0f0] text-xs"
        >›</button>
      </div>

      {/* Confidence slider */}
      <div className="px-3 py-2 border-b border-white/[0.06]">
        <div className="flex justify-between text-[9px] text-[#555] mb-1">
          <span>Confidence</span>
          <span>{(confidenceThreshold * 100).toFixed(0)}%</span>
        </div>
        <input
          type="range" min={0} max={1} step={0.05}
          value={confidenceThreshold}
          onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
          className="w-full h-0.5 accent-[#00e676] cursor-pointer"
        />
      </div>

      {/* Detection list */}
      <div className="flex-1 overflow-y-auto">
        {dets.length === 0 && (
          <div className="p-3 text-[#555] text-xs text-center">No detections</div>
        )}
        {dets.map((det, i) => {
          const color = CLASS_COLORS[det.label?.toLowerCase()] || '#f0f0f0'
          const isSelected = selectedObjects.some(
            (s) => JSON.stringify(s.center) === JSON.stringify(det.center)
          )
          return (
            <button
              key={i}
              onClick={() => toggleSelectedObject(det)}
              className={`w-full flex items-center gap-2 px-3 py-2 text-left transition-colors border-b border-white/[0.04] ${
                isSelected ? 'bg-[#00e676]/10' : 'hover:bg-white/[0.03]'
              }`}
            >
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: color }}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-medium" style={{ color }}>
                    {det.label}
                  </span>
                  {det.confidence_tier && (
                    <span className={`text-[8px] border rounded px-1 ${tierBadge(det.confidence_tier)}`}>
                      {det.confidence_tier}
                    </span>
                  )}
                </div>
                <div className="text-[10px] text-[#555]">
                  {(det.distance_m ?? 0).toFixed(1)} m
                  {det.score != null && ` · ${(det.score * 100).toFixed(0)}%`}
                </div>
              </div>
              {isSelected && (
                <div
                  className="w-3.5 h-3.5 rounded-full border-2 flex-shrink-0"
                  style={{ borderColor: '#00e676' }}
                />
              )}
            </button>
          )
        })}
      </div>

      {/* Distance readout */}
      {selectedObjects.length === 2 && (
        <div className="border-t border-white/[0.06] px-3 py-2 bg-[#00e676]/5">
          <div className="text-[9px] uppercase tracking-widest text-[#00e676]/70 mb-1">Distance</div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-[#555]">
              {selectedObjects[0].label} ↔ {selectedObjects[1].label}
            </span>
            <span className="text-sm font-bold text-[#00e676]">
              {objectDistance(selectedObjects[0], selectedObjects[1])} m
            </span>
          </div>
          <button
            onClick={clearSelectedObjects}
            className="text-[9px] text-[#555] hover:text-[#f0f0f0] mt-1"
          >
            clear selection
          </button>
        </div>
      )}

      {/* RAG query */}
      <div className="border-t border-white/[0.06] p-3 flex flex-col gap-2">
        <div className="text-[9px] uppercase tracking-widest text-[#555]">Scene search (RAG)</div>
        <div className="flex gap-1.5">
          <input
            type="text"
            value={ragQueryText}
            onChange={(e) => setRagQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleRag()}
            placeholder="find pedestrians near..."
            className="flex-1 min-w-0 text-xs bg-[#111] border border-white/[0.06] rounded px-2 py-1.5 text-[#f0f0f0] placeholder-[#333] outline-none focus:border-[#00e676]/40"
          />
          <button
            onClick={handleRag}
            disabled={ragLoading}
            className="text-xs px-2 py-1.5 rounded border border-[#00e676]/30 text-[#00e676] bg-[#00e676]/10 hover:bg-[#00e676]/20 disabled:opacity-40 transition-colors"
          >
            {ragLoading ? '…' : 'Go'}
          </button>
        </div>
        {ragResult && (
          <div className="text-[10px] text-[#f0f0f0] bg-[#111] border border-white/[0.06] rounded p-2 max-h-32 overflow-y-auto leading-relaxed">
            {ragResult.answer}
          </div>
        )}
      </div>
    </div>
  )
}
