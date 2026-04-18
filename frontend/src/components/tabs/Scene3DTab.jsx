import { useEffect, useRef } from 'react'
import useAppStore from '../../store/appStore'

const BOX_COLORS = {
  car:        '#2979ff',
  pedestrian: '#00e676',
  cyclist:    '#ffab00',
  van:        '#e040fb',
  truck:      '#ff3d71',
}

const BOX_EDGES = [
  [0,1],[1,2],[2,3],[3,0],
  [4,5],[5,6],[6,7],[7,4],
  [0,4],[1,5],[2,6],[3,7],
]

function buildBoxTraces(detections) {
  const traces = []
  for (const det of detections) {
    if (!det.corners || det.corners.length !== 8) continue
    const c = det.corners
    const color = BOX_COLORS[det.label?.toLowerCase()] || '#aaa'

    const xs = [], ys = [], zs = []
    for (const [a, b] of BOX_EDGES) {
      xs.push(c[a][0], c[b][0], null)
      ys.push(c[a][1], c[b][1], null)
      zs.push(c[a][2], c[b][2], null)
    }

    traces.push({
      type: 'scatter3d',
      mode: 'lines',
      x: xs, y: ys, z: zs,
      line: { color, width: 2 },
      name: `${det.label} ${(det.distance_m || 0).toFixed(1)}m`,
      showlegend: true,
      hoverinfo: 'name',
    })
  }
  return traces
}

export default function Scene3DTab() {
  const { result } = useAppStore()
  const containerRef = useRef(null)
  const plotRef = useRef(null)

  useEffect(() => {
    if (!result?.detections || !containerRef.current) return

    import('plotly.js-dist-min').then((Plotly) => {
      const traces = buildBoxTraces(result.detections)

      const centerX = result.detections.map((d) => d.center?.[0] || 0)
      const centerY = result.detections.map((d) => d.center?.[1] || 0)
      const centerZ = result.detections.map((d) => d.center?.[2] || 0)
      const labels  = result.detections.map(
        (d) => `${d.label}<br>${(d.distance_m || 0).toFixed(1)}m<br>${(d.score * 100).toFixed(0)}%`
      )
      const colors  = result.detections.map(
        (d) => BOX_COLORS[d.label?.toLowerCase()] || '#aaa'
      )

      if (centerX.length > 0) {
        traces.unshift({
          type: 'scatter3d',
          mode: 'markers+text',
          x: centerX, y: centerY, z: centerZ,
          text: labels,
          textposition: 'top center',
          textfont: { size: 9, color: '#f0f0f0' },
          marker: { size: 4, color: colors, opacity: 0.9 },
          name: 'centers',
          showlegend: false,
          hoverinfo: 'text',
        })
      }

      const layout = {
        paper_bgcolor: '#000',
        plot_bgcolor:  '#000',
        font: { color: '#f0f0f0', size: 10 },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        legend: {
          bgcolor: '#0a0a0a',
          bordercolor: 'rgba(255,255,255,0.06)',
          borderwidth: 1,
          font: { size: 9, color: '#f0f0f0' },
        },
        scene: {
          bgcolor: '#000',
          xaxis: { title: 'X fwd (m)', gridcolor: '#1a1a1a', zerolinecolor: '#333', color: '#555' },
          yaxis: { title: 'Y left (m)', gridcolor: '#1a1a1a', zerolinecolor: '#333', color: '#555' },
          zaxis: { title: 'Z up (m)',  gridcolor: '#1a1a1a', zerolinecolor: '#333', color: '#555' },
          aspectmode: 'data',
          camera: {
            eye: { x: -0.3, y: -1.8, z: 0.9 },
            up:  { x: 0, y: 0, z: 1 },
          },
        },
      }

      const config = {
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
        responsive: true,
      }

      if (plotRef.current) {
        Plotly.react(containerRef.current, traces, layout, config)
      } else {
        Plotly.newPlot(containerRef.current, traces, layout, config)
        plotRef.current = true
      }
    })
  }, [result])

  if (!result?.detections) {
    return (
      <div className="flex-1 flex items-center justify-center text-[#555]">
        <span className="text-sm">No 3D scene yet — run inference first</span>
      </div>
    )
  }

  return <div ref={containerRef} className="flex-1 w-full" style={{ minHeight: 0 }} />
}
