import { create } from 'zustand'

const useAppStore = create((set, get) => ({
  // ── Files ──────────────────────────────────────────────────────────────────
  files: { bin: null, image: null, calib: null, label: null },
  uploadStatus: 'idle', // 'idle' | 'uploading' | 'success' | 'error'
  uploadError: null,

  setFile: (type, file) =>
    set((s) => ({ files: { ...s.files, [type]: file } })),

  setUploadStatus: (status, error = null) =>
    set({ uploadStatus: status, uploadError: error }),

  // ── Inference result ───────────────────────────────────────────────────────
  result: null,
  // result shape:
  // { annotated_image: string, bev_image: string, detections: Detection[],
  //   inference_time_ms: number, num_points: number }

  setResult: (result) => set({ result, uploadStatus: 'success' }),

  // ── Scenes ─────────────────────────────────────────────────────────────────
  scenes: [],
  selectedScene: null,

  setScenes: (scenes) => set({ scenes }),
  setSelectedScene: (scene) => set({ selectedScene: scene }),

  // ── UI interaction ─────────────────────────────────────────────────────────
  hoveredDetectionId: null,
  clickedDetectionId: null,
  confidenceThreshold: 0.3,

  setHoveredDetectionId: (id) => set({ hoveredDetectionId: id }),
  setClickedDetectionId: (id) => set({ clickedDetectionId: id }),
  setConfidenceThreshold: (t) => set({ confidenceThreshold: t }),

  // ── Canvas measure tool ────────────────────────────────────────────────────
  canvasMode: 'view', // 'view' | 'measure'
  canvasPoints: [],   // [{ x, y, detectionIndex: number|null }]

  toggleCanvasMode: () =>
    set((s) => ({
      canvasMode: s.canvasMode === 'view' ? 'measure' : 'view',
      canvasPoints: [],
    })),

  addCanvasPoint: (point) =>
    set((s) => ({ canvasPoints: [...s.canvasPoints, point] })),

  removeLastCanvasPoint: () =>
    set((s) => ({ canvasPoints: s.canvasPoints.slice(0, -1) })),

  clearCanvasPoints: () => set({ canvasPoints: [] }),

  // ── Bulk dataset ───────────────────────────────────────────────────────────
  bulkMode: false,          // true = bulk tab is active
  bulkStatus: 'idle',       // 'idle' | 'processing' | 'done' | 'error'
  bulkError: null,
  bulkFrames: [],           // array of per-frame result objects
  bulkSelectedIdx: null,    // index into bulkFrames currently loaded in main view

  setBulkMode: (on) => set({ bulkMode: on }),
  setBulkStatus: (status, error = null) => set({ bulkStatus: status, bulkError: error }),
  setBulkFrames: (frames) => set({ bulkFrames: frames, bulkStatus: 'done' }),
  setBulkSelectedIdx: (idx) => set({ bulkSelectedIdx: idx }),

  // ── Chat ───────────────────────────────────────────────────────────────────
  chatOpen: false,
  chatMessages: [],      // [{ id, role, content, toolCall?, toolResult? }]
  chatLoading: false,
  conversationHistory: [], // raw OpenAI-format messages for API

  setChatOpen: (open) => set({ chatOpen: open }),
  setChatLoading: (loading) => set({ chatLoading: loading }),

  addChatMessage: (msg) =>
    set((s) => ({ chatMessages: [...s.chatMessages, msg] })),

  updateLastChatMessage: (patch) =>
    set((s) => {
      const msgs = [...s.chatMessages]
      msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], ...patch }
      return { chatMessages: msgs }
    }),

  setConversationHistory: (history) => set({ conversationHistory: history }),

  clearChat: () =>
    set({ chatMessages: [], conversationHistory: [] }),
}))

export default useAppStore
