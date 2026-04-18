import { useEffect } from 'react'
import useAppStore from '../../store/appStore'
import { getScenes } from '../../api/scenesApi'

export default function Header() {
  const {
    result, scenes, selectedScene, setScenes, setSelectedScene,
    chatOpen, setChatOpen,
  } = useAppStore()

  useEffect(() => {
    getScenes().then(setScenes).catch(() => {})
  }, [setScenes])

  const dets  = result?.detections?.length ?? null
  const ms    = result?.inference_time_ms ?? null
  const fid   = result?.frame_id ?? null

  return (
    <header className="flex items-center justify-between px-4 py-2.5 border-b border-white/[0.06] bg-[#000] sticky top-0 z-40 flex-shrink-0">
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold text-[#f0f0f0] tracking-tight">
          LiDAR Fusion <span className="text-[#00e676]">v2</span>
        </span>

        {fid && (
          <span className="text-[10px] text-[#555] font-mono border border-white/[0.06] rounded px-2 py-0.5">
            {fid}
          </span>
        )}

        {dets !== null && (
          <span className="text-[10px] text-[#00e676]">{dets} det{dets !== 1 ? 's' : ''}</span>
        )}

        {ms !== null && (
          <span className="text-[10px] text-[#555]">{ms} ms</span>
        )}

        {scenes.length > 0 && (
          <select
            value={selectedScene ?? ''}
            onChange={(e) => setSelectedScene(e.target.value || null)}
            className="text-[10px] bg-[#111] border border-white/[0.06] text-[#f0f0f0] rounded px-2 py-1 outline-none focus:border-[#00e676]/40 cursor-pointer"
          >
            <option value="">Upload files</option>
            {scenes.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        )}
      </div>

      <button
        onClick={() => setChatOpen(!chatOpen)}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-[10px] font-medium border transition-colors ${
          chatOpen
            ? 'bg-[#00e676]/15 border-[#00e676]/30 text-[#00e676]'
            : 'bg-white/[0.04] border-white/[0.06] text-[#555] hover:text-[#f0f0f0]'
        }`}
      >
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 0 1-2.555-.337A5.972 5.972 0 0 1 5.41 20.97a5.969 5.969 0 0 1-.474-.065 4.48 4.48 0 0 0 .978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25Z" />
        </svg>
        Chat
      </button>
    </header>
  )
}
