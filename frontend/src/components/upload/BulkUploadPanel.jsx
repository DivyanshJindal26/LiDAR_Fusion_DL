import { useRef, useState, useCallback } from 'react'
import useAppStore from '../../store/appStore'
import { runBulkInference } from '../../api/inferApi'
import BulkResultsGallery from '../bulk/BulkResultsGallery'

function VideoBlock({ label, src }) {
  if (!src) return null
  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-[9px] uppercase tracking-widest text-[#555]">{label}</p>
      <video
        className="w-full rounded border border-white/[0.08] bg-black"
        controls
        loop
        src={`data:video/mp4;base64,${src}`}
      />
    </div>
  )
}

export default function BulkUploadPanel() {
  const {
    bulkStatus,
    bulkError,
    bulkFrames,
    bulkIsTimeSeries,
    bulkAnnotatedVideo,
    bulkLidarVideo,
    bulkBevVideo,
    setBulkStatus,
    setBulkFrames,
    setResult,
    setBulkSelectedIdx,
    setBulkIsTimeSeries,
    setBulkVideos,
  } = useAppStore()
  const inputRef = useRef()
  const [zipFile, setZipFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0, frameId: null, encoding: false })

  const processing = bulkStatus === 'processing'

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.name.endsWith('.zip')) setZipFile(f)
  }

  const onProgress = useCallback((evt) => {
    if (evt.type === 'start') {
      setProgress({ current: 0, total: evt.total, frameId: null, encoding: false })
    } else if (evt.type === 'progress') {
      setProgress(p => ({ ...p, current: evt.current, total: evt.total, frameId: evt.frame_id, encoding: false }))
    } else if (evt.type === 'encoding') {
      setProgress(p => ({ ...p, encoding: true, frameId: null }))
    }
  }, [])

  async function run() {
    if (!zipFile || processing) return
    setBulkStatus('processing')
    setProgress({ current: 0, total: 0, frameId: null, encoding: false })
    try {
      const data = await runBulkInference(zipFile, bulkIsTimeSeries, onProgress)
      setBulkIsTimeSeries(bulkIsTimeSeries)
      setBulkFrames(data.frames)
      setBulkVideos(
        data.video_annotated_mp4 ?? null,
        data.video_lidar_mp4 ?? null,
        data.video_bev_mp4 ?? null,
      )
      if (data.frames.length > 0) {
        setResult(data.frames[0])
        setBulkSelectedIdx(0)
      }
    } catch (e) {
      setBulkStatus('error', e.message)
    }
  }

  const showVideos = bulkStatus === 'done' && bulkIsTimeSeries

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

      <div className="text-[9px] uppercase tracking-widest text-[#555] mt-1">Sequence mode</div>
      <div className="grid grid-cols-2 gap-1.5">
        <button
          type="button"
          onClick={() => setBulkIsTimeSeries(true)}
          className={`py-1.5 rounded text-[10px] uppercase tracking-wider border transition-colors ${
            bulkIsTimeSeries
              ? 'bg-[#00e676]/15 border-[#00e676]/40 text-[#00e676]'
              : 'bg-[#111] border-white/[0.08] text-[#666] hover:text-[#aaa]'
          }`}
        >
          Time Series
        </button>
        <button
          type="button"
          onClick={() => setBulkIsTimeSeries(false)}
          className={`py-1.5 rounded text-[10px] uppercase tracking-wider border transition-colors ${
            !bulkIsTimeSeries
              ? 'bg-[#00e676]/15 border-[#00e676]/40 text-[#00e676]'
              : 'bg-[#111] border-white/[0.08] text-[#666] hover:text-[#aaa]'
          }`}
        >
          Independent
        </button>
      </div>

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

      {processing && (
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between text-[9px] text-[#555]">
            {progress.encoding ? (
              <span className="text-[#ffab00]">Encoding video…</span>
            ) : progress.total > 0 ? (
              <>
                <span>
                  Frame <span className="text-[#f0f0f0]">{progress.current}</span> / {progress.total}
                  {progress.frameId && <span className="text-[#555]"> · {progress.frameId}</span>}
                </span>
                <span className="text-[#f0f0f0]">
                  {Math.round((progress.current / progress.total) * 100)}%
                </span>
              </>
            ) : (
              <span>Initialising…</span>
            )}
          </div>
          <div className="w-full h-1 rounded-full bg-white/[0.06] overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                progress.encoding ? 'bg-[#ffab00] animate-pulse w-full' : 'bg-[#00e676]'
              }`}
              style={!progress.encoding ? {
                width: progress.total > 0 ? `${(progress.current / progress.total) * 100}%` : '0%'
              } : undefined}
            />
          </div>
        </div>
      )}

      {bulkError && (
        <p className="text-[10px] text-[#ff3d71] bg-[#ff3d71]/10 border border-[#ff3d71]/20 rounded px-2 py-1.5">
          {bulkError}
        </p>
      )}

      {bulkStatus === 'done' && (
        <p className="text-[10px] text-[#00e676] text-center">
          {bulkFrames.length} frame{bulkFrames.length !== 1 ? 's' : ''} ready
          {bulkIsTimeSeries ? ' · videos below' : ''}
        </p>
      )}

      {showVideos && (
        <div className="flex flex-col gap-3 pt-1 border-t border-white/[0.06]">
          <p className="text-[9px] uppercase tracking-widest text-[#555]">Sequence videos</p>
          <VideoBlock label="Annotated camera" src={bulkAnnotatedVideo} />
          <VideoBlock label="LiDAR overlay"    src={bulkLidarVideo} />
          <VideoBlock label="Bird's eye view"  src={bulkBevVideo} />
        </div>
      )}

      {bulkStatus === 'done' && bulkFrames.length > 0 && (
        <div className="pt-1 border-t border-white/[0.06]">
          <p className="text-[9px] uppercase tracking-widest text-[#555] mb-2">
            {bulkIsTimeSeries ? 'Frames (sequence)' : 'Independent frame results'}
          </p>
          <BulkResultsGallery />
        </div>
      )}
    </div>
  )
}
