import FileSlot from './FileSlot'
import BulkUploadPanel from './BulkUploadPanel'
import BulkResultsGallery from '../bulk/BulkResultsGallery'
import useAppStore from '../../store/appStore'
import { useInference } from '../../hooks/useInference'
import clsx from 'clsx'

export default function UploadPanel() {
  const {
    files, setFile, uploadStatus, uploadError,
    selectedScene, confidenceThreshold, setConfidenceThreshold,
    bulkMode, setBulkMode,
  } = useAppStore()
  const { run } = useInference()

  const canRun = selectedScene || (files.bin && files.image && files.calib)
  const uploading = uploadStatus === 'uploading'

  return (
    <div className="flex flex-col gap-0">
      {/* Mode tabs */}
      <div className="flex border-b border-slate-700/50 px-2 pt-2 gap-1">
        {['Single', 'Bulk'].map((label) => {
          const active = label === 'Bulk' ? bulkMode : !bulkMode
          return (
            <button
              key={label}
              onClick={() => setBulkMode(label === 'Bulk')}
              className={clsx(
                'flex-1 py-1.5 text-xs font-semibold rounded-t-lg transition-colors',
                active
                  ? 'text-blue-400 border-b-2 border-blue-500 -mb-px'
                  : 'text-slate-500 hover:text-slate-300'
              )}
            >
              {label}
            </button>
          )
        })}
      </div>

      {bulkMode ? (
        /* ── Bulk tab ─────────────────────────────────────────── */
        <div className="flex flex-col gap-4">
          <BulkUploadPanel />
          <div className="px-3 pb-3">
            <BulkResultsGallery />
          </div>
        </div>
      ) : (
        /* ── Single-frame tab ─────────────────────────────────── */
        <div className="flex flex-col gap-4 p-4">
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Input Files</p>
            <div className="grid grid-cols-1 gap-2">
              <FileSlot type="bin"   file={files.bin}   onFile={(f) => setFile('bin', f)} />
              <FileSlot type="image" file={files.image} onFile={(f) => setFile('image', f)} />
              <FileSlot type="calib" file={files.calib} onFile={(f) => setFile('calib', f)} />
              <FileSlot type="label" file={files.label} onFile={(f) => setFile('label', f)} />
            </div>
          </div>

          <button
            onClick={run}
            disabled={!canRun || uploading}
            className={clsx(
              'w-full py-2.5 rounded-xl text-sm font-semibold transition-all duration-200',
              canRun && !uploading
                ? 'bg-blue-500 hover:bg-blue-400 text-white shadow-lg shadow-blue-500/20'
                : 'bg-slate-800 text-slate-600 cursor-not-allowed'
            )}
          >
            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Running inference…
              </span>
            ) : 'Run Inference'}
          </button>

          {uploadError && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {uploadError}
            </p>
          )}

          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Confidence</p>
              <span className="text-xs font-mono text-slate-300">{(confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range" min={0} max={1} step={0.05} value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
              className="w-full accent-blue-500 cursor-pointer"
            />
            <div className="flex justify-between text-xs text-slate-600 mt-1">
              <span>0%</span><span>100%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
