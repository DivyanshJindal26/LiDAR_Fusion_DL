import { useRef, useState } from 'react'
import useAppStore from '../../store/appStore'
import { runBulkInference } from '../../api/inferApi'

export default function BulkUploadPanel() {
  const { bulkStatus, bulkError, bulkFrames, setBulkStatus, setBulkFrames, setResult, setBulkSelectedIdx } =
    useAppStore()
  const inputRef = useRef()
  const [zipFile, setZipFile] = useState(null)
  const [maxFrames, setMaxFrames] = useState(20)
  const [dragging, setDragging] = useState(false)

  const processing = bulkStatus === 'processing'

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.name.endsWith('.zip')) setZipFile(f)
  }

  async function run() {
    if (!zipFile || processing) return
    setBulkStatus('processing')
    try {
      const data = await runBulkInference(zipFile, maxFrames)
      setBulkFrames(data.frames)
      if (data.frames.length > 0) {
        setResult(data.frames[0])
        setBulkSelectedIdx(0)
      }
    } catch (e) {
      setBulkStatus('error', e.message)
    }
  }

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="text-[9px] uppercase tracking-widest text-[#555]">Dataset ZIP</div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`border border-dashed rounded text-center cursor-pointer transition-colors p-4 ${
          dragging
            ? 'border-[#00e676]/60 bg-[#00e676]/10'
            : zipFile
              ? 'border-[#00e676]/30 bg-[#00e676]/5'
              : 'border-white/[0.06] bg-[#111] hover:border-white/[0.12]'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".zip"
          className="hidden"
          onChange={(e) => setZipFile(e.target.files[0] || null)}
        />
        {zipFile ? (
          <div>
            <div className="text-[#00e676] text-xs mb-1">✓ {zipFile.name}</div>
            <div className="text-[#555] text-[10px]">{(zipFile.size / 1024 / 1024).toFixed(1)} MB</div>
          </div>
        ) : (
          <div>
            <div className="text-[#555] text-xs mb-1">Drop KITTI ZIP</div>
            <div className="text-[#333] text-[10px]">or click to browse</div>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[10px]">
        <span className="text-[#555]">Max frames</span>
        <span className="text-[#f0f0f0] tabular-nums">{maxFrames}</span>
      </div>
      <input
        type="range" min={1} max={50} value={maxFrames}
        onChange={(e) => setMaxFrames(Number(e.target.value))}
        className="w-full h-0.5 accent-[#00e676] cursor-pointer"
      />

      <button
        onClick={run}
        disabled={!zipFile || processing}
        className={`w-full py-2 rounded text-xs font-semibold transition-colors ${
          zipFile && !processing
            ? 'bg-[#00e676] text-black hover:bg-[#00ff84]'
            : 'bg-[#111] text-[#555] cursor-not-allowed border border-white/[0.06]'
        }`}
      >
        {processing ? (
          <span className="flex items-center justify-center gap-1.5">
            <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Processing…
          </span>
        ) : 'Process Dataset'}
      </button>

      {bulkError && (
        <p className="text-[10px] text-[#ff3d71] bg-[#ff3d71]/10 border border-[#ff3d71]/20 rounded px-2 py-1.5">
          {bulkError}
        </p>
      )}

      {bulkStatus === 'done' && (
        <p className="text-[10px] text-[#00e676] text-center">
          {bulkFrames.length} frame{bulkFrames.length !== 1 ? 's' : ''} ready
        </p>
      )}
    </div>
  )
}
