import { useRef, useState } from 'react'
import useAppStore from '../../store/appStore'
import { runBulkInference } from '../../api/inferApi'
import clsx from 'clsx'

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
      // Auto-load first frame into the main view
      if (data.frames.length > 0) {
        setResult(data.frames[0])
        setBulkSelectedIdx(0)
      }
    } catch (e) {
      setBulkStatus('error', e.message)
    }
  }

  return (
    <div className="flex flex-col gap-4 p-4">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
        Dataset ZIP
      </p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={clsx(
          'border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-colors',
          dragging
            ? 'border-blue-400 bg-blue-500/10'
            : zipFile
              ? 'border-green-500/50 bg-green-500/5'
              : 'border-slate-600 hover:border-slate-500'
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".zip"
          className="hidden"
          onChange={(e) => setZipFile(e.target.files[0] || null)}
        />
        {zipFile ? (
          <>
            <div className="text-green-400 text-lg mb-1">✓</div>
            <p className="text-xs text-slate-300 font-mono truncate">{zipFile.name}</p>
            <p className="text-xs text-slate-500 mt-1">
              {(zipFile.size / 1024 / 1024).toFixed(1)} MB
            </p>
          </>
        ) : (
          <>
            <div className="text-slate-500 text-2xl mb-2">📦</div>
            <p className="text-xs text-slate-400">Drop KITTI ZIP here</p>
            <p className="text-xs text-slate-600 mt-1">or click to browse</p>
          </>
        )}
      </div>

      {/* Max frames */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <p className="text-xs text-slate-400">Max frames</p>
          <span className="text-xs font-mono text-slate-300">{maxFrames}</span>
        </div>
        <input
          type="range" min={1} max={50} value={maxFrames}
          onChange={(e) => setMaxFrames(Number(e.target.value))}
          className="w-full accent-blue-500 cursor-pointer"
        />
        <div className="flex justify-between text-xs text-slate-600 mt-0.5">
          <span>1</span><span>50</span>
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={run}
        disabled={!zipFile || processing}
        className={clsx(
          'w-full py-2.5 rounded-xl text-sm font-semibold transition-all duration-200',
          zipFile && !processing
            ? 'bg-blue-500 hover:bg-blue-400 text-white shadow-lg shadow-blue-500/20'
            : 'bg-slate-800 text-slate-600 cursor-not-allowed'
        )}
      >
        {processing ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Processing…
          </span>
        ) : 'Process Dataset'}
      </button>

      {bulkError && (
        <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {bulkError}
        </p>
      )}

      {bulkStatus === 'done' && (
        <p className="text-xs text-green-400 text-center">
          {bulkFrames.length} frame{bulkFrames.length !== 1 ? 's' : ''} processed
        </p>
      )}
    </div>
  )
}
