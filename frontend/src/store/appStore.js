import { create } from 'zustand'

const useAppStore = create((set, get) => ({
  // ── Files ──────────────────────────────────────────────────────────────────
  files: { bin: null, image: null, calib: null, label: null },
  uploadStatus: 'idle',
  uploadError: null,

  setFile: (type, file) =>
    set((s) => ({ files: { ...s.files, [type]: file } })),

  setUploadStatus: (status, error = null) =>
    set({ uploadStatus: status, uploadError: error }),

  // ── Inference result ───────────────────────────────────────────────────────
  result: null,
  setResult: (result) => set({ result, uploadStatus: 'success' }),

  // ── Scenes ─────────────────────────────────────────────────────────────────
  scenes: [],
  selectedScene: null,
  setScenes: (scenes) => set({ scenes }),
  setSelectedScene: (scene) => set({ selectedScene: scene }),

  // ── Tabs ───────────────────────────────────────────────────────────────────
  activeTab: 'camera',   // 'camera' | 'lidar' | 'scene3d' | 'metrics'
  setActiveTab: (tab) => set({ activeTab: tab }),

  // ── Object distance selection (click up to 2 detections) ──────────────────
  selectedObjects: [],   // array of detection objects (max 2)
  toggleSelectedObject: (det) =>
    set((s) => {
      const exists = s.selectedObjects.findIndex(
        (d) => d.label === det.label && d.distance_m === det.distance_m &&
               JSON.stringify(d.center) === JSON.stringify(det.center)
      )
      if (exists >= 0) {
        return { selectedObjects: s.selectedObjects.filter((_, i) => i !== exists) }
      }
      if (s.selectedObjects.length >= 2) {
        return { selectedObjects: [s.selectedObjects[1], det] }
      }
      return { selectedObjects: [...s.selectedObjects, det] }
    }),
  clearSelectedObjects: () => set({ selectedObjects: [] }),

  // ── RAG query ─────────────────────────────────────────────────────────────
  ragQuery: '',
  ragLoading: false,
  ragResult: null,
  setRagQuery: (q) => set({ ragQuery: q }),
  setRagLoading: (v) => set({ ragLoading: v }),
  setRagResult: (r) => set({ ragResult: r }),

  // ── Confidence filter ─────────────────────────────────────────────────────
  confidenceThreshold: 0.3,
  setConfidenceThreshold: (t) => set({ confidenceThreshold: t }),

  // ── Bulk dataset ───────────────────────────────────────────────────────────
  bulkMode: false,
  bulkStatus: 'idle',
  bulkError: null,
  bulkFrames: [],
  bulkSelectedIdx: null,

  setBulkMode: (on) => set({ bulkMode: on }),
  setBulkStatus: (status, error = null) => set({ bulkStatus: status, bulkError: error }),
  setBulkFrames: (frames) => set({ bulkFrames: frames, bulkStatus: 'done' }),
  setBulkSelectedIdx: (idx) => set({ bulkSelectedIdx: idx }),

  // ── Chat ───────────────────────────────────────────────────────────────────
  chatOpen: false,
  chatMessages: [],
  chatLoading: false,
  conversationHistory: [],

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
  clearChat: () => set({ chatMessages: [], conversationHistory: [] }),
}))

export default useAppStore
