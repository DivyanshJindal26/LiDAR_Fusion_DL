import FileSlot from './FileSlot'
import BulkUploadPanel from './BulkUploadPanel'
import useAppStore from '../../store/appStore'
import { useInference } from '../../hooks/useInference'

export default function UploadPanel() {
  const {
    files, setFile, uploadStatus, uploadError,
    selectedScene, bulkMode, setBulkMode,
  } = useAppStore()
  const { run } = useInference()

  const canRun = selectedScene || (files.bin && files.image && files.calib)
  const uploading = uploadStatus === 'uploading'

  return (
    <div className="flex flex-col h-full">
      {/* Mode toggle */}
      <div className="flex border-b border-white/[0.06]">
        {['Single', 'Bulk'].map((label) => {
          const active = label === 'Bulk' ? bulkMode : !bulkMode
          return (
            <button
              key={label}
              onClick={() => setBulkMode(label === 'Bulk')}
              className={`flex-1 py-2 text-[10px] uppercase tracking-widest font-medium transition-colors ${
                active ? 'tab-active' : 'tab-inactive'
              }`}
            >
              {label}
            </button>
          )
        })}
      </div>

      {bulkMode ? (
        <div className="flex-1 overflow-y-auto">
          <BulkUploadPanel />
        </div>
      ) : (
        <div className="flex flex-col gap-3 p-3 flex-1">
          <div className="text-[9px] uppercase tracking-widest text-[#555]">Input files</div>
          <div className="flex flex-col gap-2">
            <FileSlot type="bin"   file={files.bin}   onFile={(f) => setFile('bin', f)} />
            <FileSlot type="image" file={files.image} onFile={(f) => setFile('image', f)} />
            <FileSlot type="calib" file={files.calib} onFile={(f) => setFile('calib', f)} />
            <FileSlot type="label" file={files.label} onFile={(f) => setFile('label', f)} />
          </div>

          <button
            onClick={run}
            disabled={!canRun || uploading}
            className={`w-full py-2.5 rounded text-xs font-semibold transition-all mt-auto ${
              canRun && !uploading
                ? 'bg-[#00e676] text-black hover:bg-[#00ff84]'
                : 'bg-[#111] text-[#555] cursor-not-allowed border border-white/[0.06]'
            }`}
          >
            {uploading ? (
              <span className="flex items-center justify-center gap-1.5">
                <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Running…
              </span>
            ) : 'Run Inference'}
          </button>

          {uploadError && (
            <p className="text-[10px] text-[#ff3d71] bg-[#ff3d71]/10 border border-[#ff3d71]/20 rounded px-2 py-1.5">
              {uploadError}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
