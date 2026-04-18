import { useEffect, useRef } from 'react'
import useAppStore from '../../store/appStore'

const BOX_COLORS = {
  car:        'lime',
  pedestrian: 'cyan',
  cyclist:    'yellow',
  truck:      'orange',
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
    const detClass = det.label ?? det.class ?? 'object'
    const color = BOX_COLORS[detClass.toLowerCase()] || '#aaa'

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
      line: { color, width: 5 },
      name: `${detClass} ${(det.distance_m || 0).toFixed(1)}m`,
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

      traces.push({
        type: 'scatter3d',
        mode: 'markers+text',
        x: [0], y: [0], z: [0],
        marker: { size: 6, color: 'red', symbol: 'diamond' },
        text: ['EGO'],
        textposition: 'top center',
        name: 'Ego',
        showlegend: true,
        hoverinfo: 'name',
      })

      const layout = {
        title: 'PointPillars — 3D Detections',
        paper_bgcolor: '#111',
        plot_bgcolor:  '#111',
        font: { color: 'white', size: 10 },
        margin: { l: 0, r: 0, t: 36, b: 0 },
        legend: {
          bgcolor: '#222',
          bordercolor: '#555',
          borderwidth: 1,
          font: { size: 10, color: 'white' },
        },
        scene: {
          bgcolor: '#111',
          xaxis: { title: 'X fwd(m)', color: 'white' },
          yaxis: { title: 'Y left(m)', color: 'white' },
          zaxis: { title: 'Z up(m)', color: 'white' },
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
