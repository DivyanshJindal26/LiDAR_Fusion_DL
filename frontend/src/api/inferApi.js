export async function getChromaStats() {
  const res = await fetch('/api/chroma-stats')
  if (!res.ok) return { count: 0 }
  return res.json()
}

export async function runInference({ bin, image, calib, label }) {
  const form = new FormData()
  form.append('bin_file', bin)
  form.append('image_file', image)
  form.append('calib_file', calib)
  if (label) form.append('label_file', label)

  const res = await fetch('/api/infer', { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Inference failed (${res.status}): ${text}`)
  }
  return res.json()
}

export async function runSceneInference(sceneId) {
  const res = await fetch(`/api/infer-scene/${encodeURIComponent(sceneId)}`, {
    method: 'POST',
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Scene inference failed (${res.status}): ${text}`)
  }
  return res.json()
}


export async function runBulkInference(zipFile, isTimeSeries = true, onProgress = null) {
  const form = new FormData()
  form.append('zip_file', zipFile)

  const qs = new URLSearchParams({ is_timeseries: String(isTimeSeries) })
  const res = await fetch(`/api/infer-bulk?${qs.toString()}`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Bulk inference failed (${res.status}): ${text}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const frames = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const event = JSON.parse(line.slice(6))
      if (event.type === 'start' && onProgress) {
        onProgress({ type: 'start', total: event.total })
      } else if (event.type === 'progress') {
        if (event.frame) frames.push(event.frame)
        if (onProgress) onProgress({ type: 'progress', current: event.current, total: event.total, frame_id: event.frame_id, error: event.error })
      } else if (event.type === 'encoding' && onProgress) {
        onProgress({ type: 'encoding' })
      } else if (event.type === 'done') {
        return { ...event, frames }
      } else if (event.type === 'error') {
        throw new Error(event.message)
      }
    }
  }
  throw new Error('SSE stream ended without done event')
}
