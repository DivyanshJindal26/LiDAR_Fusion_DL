import { useRef, useState } from 'react'

const LABELS = { bin: 'LiDAR .bin', image: 'Image .png', calib: 'Calib .txt', label: 'Labels .txt (opt)' }
const ACCEPT = { bin: '.bin', image: '.png,.jpg,.jpeg', calib: '.txt', label: '.txt' }

export default function FileSlot({ type, file, onFile }) {
  const inputRef = useRef()
  const [drag, setDrag] = useState(false)

  function handleDrop(e) {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f) onFile(f)
  }

  return (
    <div
      className={`relative flex items-center gap-2 rounded border px-2 py-2 cursor-pointer transition-colors text-xs ${
        drag
          ? 'border-[#00e676]/50 bg-[#00e676]/10'
          : file
          ? 'border-[#00e676]/30 bg-[#00e676]/5'
          : 'border-white/[0.06] bg-[#111] hover:border-white/[0.12]'
      }`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT[type]}
        className="hidden"
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
      <div className="flex-1 min-w-0">
        <div className="text-[#555] text-[9px] uppercase tracking-wider">{LABELS[type]}</div>
        {file ? (
          <div className="text-[#00e676] truncate mt-0.5">{file.name}</div>
        ) : (
          <div className="text-[#333]">drop or click</div>
        )}
      </div>
      {file && (
        <button
          className="text-[#333] hover:text-[#ff3d71] transition-colors flex-shrink-0"
          onClick={(e) => { e.stopPropagation(); onFile(null) }}
        >
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  )
}
