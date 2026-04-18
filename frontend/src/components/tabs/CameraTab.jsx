import { useRef, useState } from 'react'
import useAppStore from '../../store/appStore'

const CLASS_COLORS = {
  car:        '#2979ff',
  pedestrian: '#00e676',
  cyclist:    '#ffab00',
  van:        '#e040fb',
  truck:      '#ff3d71',
}

function objectDistance(a, b) {
  const [x1, y1, z1] = a.center
  const [x2, y2, z2] = b.center
  return Math.hypot(x2 - x1, y2 - y1, z2 - z1).toFixed(2)
}

export default function CameraTab() {
  const { result, selectedObjects } = useAppStore()
  const imgRef = useRef(null)
  const [imgRect, setImgRect] = useState(null)

  if (!result?.camera_image) {
    return (
      <div className="flex-1 flex items-center justify-center text-[#555]">
        <span className="text-sm">No camera frame yet — run inference first</span>
      </div>
    )
  }

  const updateRect = () => {
    if (imgRef.current) setImgRect(imgRef.current.getBoundingClientRect())
  }

  const dets = result.detections || []

  const dotPositions = selectedObjects.map((det) => {
    const b = det.bbox_2d
    if (!b || !imgRect) return null
    const imgNatW = imgRef.current?.naturalWidth || 1
    const imgNatH = imgRef.current?.naturalHeight || 1
    const scaleX = imgRect.width / imgNatW
    const scaleY = imgRect.height / imgNatH
    return {
      x: ((b[0] + b[2]) / 2) * scaleX,
      y: ((b[1] + b[3]) / 2) * scaleY,
      color: CLASS_COLORS[det.label?.toLowerCase()] || '#f0f0f0',
    }
  }).filter(Boolean)

  return (
    <div className="flex-1 flex items-center justify-center overflow-hidden relative bg-black">
      <div className="relative">
        <img
          ref={imgRef}
          src={`data:image/png;base64,${result.camera_image}`}
          alt="Camera frame with 3D boxes"
          className="max-h-full max-w-full object-contain"
          onLoad={updateRect}
        />

        {selectedObjects.length === 2 && dotPositions.length === 2 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ width: imgRect?.width, height: imgRect?.height }}
          >
            <line
              x1={dotPositions[0].x} y1={dotPositions[0].y}
              x2={dotPositions[1].x} y2={dotPositions[1].y}
              stroke="#00e676" strokeWidth="1.5" strokeDasharray="5 4"
              opacity="0.85"
            />
            {dotPositions.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r={5} fill={p.color} stroke="#000" strokeWidth={1.5} />
            ))}
            <text
              x={(dotPositions[0].x + dotPositions[1].x) / 2}
              y={(dotPositions[0].y + dotPositions[1].y) / 2 - 8}
              fill="#00e676"
              fontSize="11"
              fontWeight="600"
              textAnchor="middle"
              stroke="#000"
              strokeWidth="3"
              paintOrder="stroke"
            >
              {objectDistance(selectedObjects[0], selectedObjects[1])} m
            </text>
          </svg>
        )}
      </div>
    </div>
  )
}
