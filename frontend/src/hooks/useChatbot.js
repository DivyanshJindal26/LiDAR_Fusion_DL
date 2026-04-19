import { useCallback } from 'react'
import { v4 as uuid } from 'uuid'
import { sendChat } from '../api/chatApi'
import { queryScene } from '../api/queryApi'
import useAppStore from '../store/appStore'

// ── Detection helpers ──────────────────────────────────────────────────────────

function getClassCounts(detections = []) {
  return detections.reduce((acc, d) => {
    const cls = String(d.label ?? d.class ?? 'unknown').toLowerCase()
    acc[cls] = (acc[cls] || 0) + 1
    return acc
  }, {})
}

function fmtDet(d) {
  const cls  = d.label ?? d.class ?? 'unknown'
  const conf = d.score ?? d.confidence
  const dist = typeof d.distance_m === 'number' ? d.distance_m.toFixed(1) : '?'
  const pos  = d.center ?? d.xyz ?? []
  const posStr = Array.isArray(pos)
    ? pos.map((v) => (typeof v === 'number' ? v.toFixed(1) : v)).join(', ')
    : ''
  const confStr = conf == null ? '' : ` (${Math.round(conf * 100)}%)`
  const tierStr = d.confidence_tier ? ` [${d.confidence_tier}]` : ''
  return `  - ${cls}${confStr}: ${dist}m @ [${posStr}]${tierStr}`
}

function frameDetectionLine(frame) {
  const dets = frame.detections ?? []
  const counts = getClassCounts(dets)
  const summary = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([cls, n]) => `${cls}×${n}`)
    .join(', ')
  const avgDist = dets.length
    ? (dets.reduce((s, d) => s + (d.distance_m ?? 0), 0) / dets.length).toFixed(1)
    : '?'
  return `  Frame ${frame.frame_id ?? '?'}: ${dets.length} obj — ${summary || 'none'} | avg dist ${avgDist}m | ${frame.inference_time_ms ?? '?'}ms`
}

// ── Consistency check ──────────────────────────────────────────────────────────

function enforceSceneConsistency(answer, detections = []) {
  if (!detections.length || !answer) return answer
  const counts = getClassCounts(detections)
  const checks = [
    { cls: 'car',        re: /(no\s+cars?|0\s+cars?|zero\s+cars?)/i },
    { cls: 'pedestrian', re: /(no\s+pedestrians?|0\s+pedestrians?|zero\s+pedestrians?)/i },
    { cls: 'cyclist',    re: /(no\s+cyclists?|0\s+cyclists?|zero\s+cyclists?)/i },
    { cls: 'truck',      re: /(no\s+trucks?|0\s+trucks?|zero\s+trucks?)/i },
  ]
  const contradictions = checks
    .filter(({ cls, re }) => (counts[cls] ?? 0) > 0 && re.test(answer))
    .map(({ cls }) => `${counts[cls]} ${cls}(s)`)
  if (!contradictions.length) return answer
  return `[Data correction: scene contains ${contradictions.join(', ')}]\n\n${answer}`
}

// ── System prompt builders ─────────────────────────────────────────────────────

const RULES = `OPERATING RULES (STRICT):
1. Use only the provided scene data and tool outputs — never invent objects or counts.
2. For count/presence questions call query_scene before answering if any ambiguity exists.
3. Never claim zero detections of a class that appears in the data.
4. Keep answers concise: finding → evidence → conclusion. Include exact numbers.
5. Distances are in metres. Confidence scores are 0–1 (multiply by 100 for %).

SCHEMA: label, score, distance_m, center [x,y,z], bbox_2d, confidence_tier, source.

CLASS ALIASES: car/vehicle → car | person/pedestrian → pedestrian | bicycle/cyclist → cyclist`

function buildSingleFramePrompt(result) {
  if (!result) {
    return 'You are a LiDAR+Camera Fusion perception assistant. No scene loaded — ask the user to upload one.'
  }
  const dets   = result.detections ?? []
  const counts = getClassCounts(dets)
  const stats  = result.pipeline_stats ?? {}

  const countLines = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([cls, n]) => `  ${cls}: ${n}`)
    .join('\n') || '  (none)'

  const detLines = dets.map(fmtDet).join('\n') || '  (none)'

  return `You are a LiDAR+Camera Fusion perception assistant analyzing a KITTI autonomous driving scene.

${RULES}

SCENE STATS:
  LiDAR points: ${result.num_points?.toLocaleString() ?? '?'}
  Inference time: ${result.inference_time_ms ?? '?'} ms
  Pipeline: YOLO ${stats.yolo_n ?? '?'} | PP_raw ${stats.pp_raw_n ?? '?'} | T1_fused ${stats.tier1_fused_n ?? '?'} | T2_gated ${stats.pp_gated_n ?? '?'} | T3_obb ${stats.obb_n ?? '?'} | final ${stats.final_n ?? '?'}

CLASS COUNTS (${dets.length} total detections):
${countLines}

ALL DETECTIONS:
${detLines}

Use query_scene to answer semantic or filtered questions. Reference actual numbers from the data above.`
}

function buildTemporalPrompt(result, bulkFrames, bulkSelectedIdx) {
  const totalFrames = bulkFrames.length
  const currentFrame = bulkFrames[bulkSelectedIdx ?? 0] ?? result

  // Global aggregated stats across all frames
  const globalCounts = {}
  const globalDistances = {}
  for (const f of bulkFrames) {
    for (const d of f.detections ?? []) {
      const cls = String(d.label ?? d.class ?? 'unknown').toLowerCase()
      globalCounts[cls] = (globalCounts[cls] ?? 0) + 1
      if (!globalDistances[cls]) globalDistances[cls] = []
      if (typeof d.distance_m === 'number') globalDistances[cls].push(d.distance_m)
    }
  }

  const globalLines = Object.entries(globalCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([cls, total]) => {
      const dists = globalDistances[cls] ?? []
      if (!dists.length) return `  ${cls}: ${total} total`
      const avg  = (dists.reduce((s, v) => s + v, 0) / dists.length).toFixed(1)
      const min  = Math.min(...dists).toFixed(1)
      const max  = Math.max(...dists).toFixed(1)
      return `  ${cls}: ${total} total | avg ${avg}m | range ${min}–${max}m`
    })
    .join('\n') || '  (none)'

  // Per-frame compact summary (capped at 200 frames to avoid token overflow)
  const frameLines = bulkFrames
    .slice(0, 200)
    .map(frameDetectionLine)
    .join('\n')

  // Current frame full detail
  const curDets    = currentFrame?.detections ?? []
  const curCounts  = getClassCounts(curDets)
  const curStats   = currentFrame?.pipeline_stats ?? {}
  const curCountLines = Object.entries(curCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([cls, n]) => `  ${cls}: ${n}`)
    .join('\n') || '  (none)'
  const curDetLines = curDets.map(fmtDet).join('\n') || '  (none)'

  return `You are a LiDAR+Camera Fusion perception assistant analyzing a KITTI temporal driving sequence.

${RULES}

═══ TEMPORAL DATASET OVERVIEW ═══
Total frames processed: ${totalFrames}
Total detections across all frames: ${Object.values(globalCounts).reduce((s, n) => s + n, 0)}

GLOBAL CLASS STATS (all ${totalFrames} frames):
${globalLines}

PER-FRAME SUMMARY:
${frameLines}

═══ CURRENT FRAME (index ${bulkSelectedIdx ?? 0} / frame_id: ${currentFrame?.frame_id ?? '?'}) ═══
LiDAR points: ${currentFrame?.num_points?.toLocaleString() ?? '?'}
Inference time: ${currentFrame?.inference_time_ms ?? '?'} ms
Pipeline: YOLO ${curStats.yolo_n ?? '?'} | T1 ${curStats.tier1_fused_n ?? '?'} | T2 ${curStats.pp_gated_n ?? '?'} | T3 ${curStats.obb_n ?? '?'} | final ${curStats.final_n ?? '?'}

Class counts (${curDets.length} detections):
${curCountLines}

All detections:
${curDetLines}

Use query_scene to answer questions about the current frame. For temporal/cross-frame questions, use the dataset overview above.`
}

function buildSystemPrompt(result, bulkFrames, bulkIsTimeSeries, bulkSelectedIdx) {
  if (bulkIsTimeSeries && bulkFrames.length > 1) {
    return buildTemporalPrompt(result, bulkFrames, bulkSelectedIdx)
  }
  return buildSingleFramePrompt(result)
}

// ── Tool definition ────────────────────────────────────────────────────────────

const QUERY_TOOL = {
  type: 'function',
  function: {
    name: 'query_scene',
    description:
      'Semantically search the current scene\'s detected objects. Use for presence, count, distance, and filtered questions.',
    parameters: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'Natural language query, e.g. "cars within 15 meters"',
        },
        max_distance_m: {
          type: 'number',
          description: 'Optional: filter to objects within this many metres',
        },
      },
      required: ['text'],
    },
  },
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useChatbot() {
  const {
    result,
    bulkFrames,
    bulkIsTimeSeries,
    bulkSelectedIdx,
    conversationHistory,
    addChatMessage,
    updateLastChatMessage,
    setConversationHistory,
    setChatLoading,
  } = useAppStore()

  const sendMessage = useCallback(
    async (userText) => {
      addChatMessage({ id: uuid(), role: 'user', content: userText })
      setChatLoading(true)

      const history = [...conversationHistory, { role: 'user', content: userText }]
      const asstId  = uuid()
      addChatMessage({ id: asstId, role: 'assistant', content: '', loading: true })

      const systemPrompt = buildSystemPrompt(result, bulkFrames, bulkIsTimeSeries, bulkSelectedIdx)
      const sceneContext = { system: systemPrompt, tools: [QUERY_TOOL] }

      try {
        let response = await sendChat({ messages: history, sceneContext })

        // Agentic tool-use loop — keep the spinner showing the whole time
        while (response.tool_calls?.length > 0) {
          const call = response.tool_calls[0]

          let toolResult
          try {
            toolResult = call.name === 'query_scene'
              ? await queryScene(call.input)
              : { error: `Unknown tool: ${call.name}` }
          } catch (err) {
            toolResult = { error: err.message }
          }

          history.push({ role: 'assistant', content: response.content ?? '', tool_calls: response.tool_calls })
          history.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(toolResult) })

          response = await sendChat({ messages: history, sceneContext })
        }

        // Consistency check against the current frame's detections
        const currentDets = (bulkIsTimeSeries && bulkFrames.length > 1)
          ? (bulkFrames[bulkSelectedIdx ?? 0]?.detections ?? result?.detections ?? [])
          : (result?.detections ?? [])

        const finalText = enforceSceneConsistency(response.content ?? '', currentDets)
        updateLastChatMessage({ content: finalText, loading: false })
        history.push({ role: 'assistant', content: finalText })
        setConversationHistory(history)
      } catch (err) {
        updateLastChatMessage({ content: `Error: ${err.message}`, loading: false, error: true })
      } finally {
        setChatLoading(false)
      }
    },
    [result, bulkFrames, bulkIsTimeSeries, bulkSelectedIdx,
     conversationHistory, addChatMessage, updateLastChatMessage,
     setConversationHistory, setChatLoading],
  )

  return { sendMessage }
}
