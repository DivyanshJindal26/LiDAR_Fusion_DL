import { useEffect, useRef } from 'react'
import useAppStore from '../../store/appStore'

export default function FrameScrubber() {
  const { bulkFrames, bulkSelectedIdx, setBulkSelectedIdx, setResult } = useAppStore()
  const scrollRef = useRef(null)

  const idx = bulkSelectedIdx ?? 0

  const select = (i) => {
    setBulkSelectedIdx(i)
    setResult(bulkFrames[i])
  }

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowRight' && idx < bulkFrames.length - 1) select(idx + 1)
      if (e.key === 'ArrowLeft'  && idx > 0)                      select(idx - 1)
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [idx, bulkFrames])

  // Scroll thumbnail into view when idx changes
  useEffect(() => {
    const el = scrollRef.current?.children[idx]
    el?.scrollIntoView({ inline: 'center', behavior: 'smooth', block: 'nearest' })
  }, [idx])

  if (!bulkFrames.length) return null

  return (
    <div className="h-20 border-t border-white/[0.06] bg-[#0a0a0a] flex flex-col px-3 py-1.5 gap-1 flex-shrink-0">
      {/* Thumbnail row */}
      <div className="flex-1 flex items-center gap-1.5 overflow-x-auto" ref={scrollRef}>
        {bulkFrames.map((frame, i) => (
          <button
            key={i}
            onClick={() => select(i)}
            className={`flex-shrink-0 w-16 h-10 rounded overflow-hidden border transition-colors ${
              i === idx
                ? 'border-[#00e676]'
                : 'border-white/[0.06] opacity-60 hover:opacity-100'
            }`}
          >
            {frame.camera_image ? (
              <img
                src={`data:image/png;base64,${frame.camera_image}`}
                className="w-full h-full object-cover"
                alt={`frame ${i}`}
              />
            ) : (
              <div className="w-full h-full bg-[#111] flex items-center justify-center">
                <span className="text-[8px] text-[#555]">{i}</span>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Scrubber range */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => idx > 0 && select(idx - 1)}
          className="text-[#555] hover:text-[#f0f0f0] text-xs px-1"
        >◄</button>
        <input
          type="range"
          min={0}
          max={bulkFrames.length - 1}
          value={idx}
          onChange={(e) => select(Number(e.target.value))}
          className="flex-1 h-1 accent-[#00e676] cursor-pointer"
        />
        <button
          onClick={() => idx < bulkFrames.length - 1 && select(idx + 1)}
          className="text-[#555] hover:text-[#f0f0f0] text-xs px-1"
        >►</button>
        <span className="text-[10px] text-[#555] tabular-nums w-12 text-right">
          {idx + 1}/{bulkFrames.length}
        </span>
      </div>
    </div>
  )
}
