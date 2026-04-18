import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'
import useAppStore from '../../store/appStore'
import { useChatbot } from '../../hooks/useChatbot'

export default function ChatPanel() {
  const { chatOpen, setChatOpen, chatMessages, clearChat } = useAppStore()
  const { sendMessage } = useChatbot()
  const bottomRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  if (!chatOpen) return null

  return (
    <div className="fixed right-0 top-0 h-full w-80 z-50 flex flex-col border-l border-white/[0.06] bg-[#0a0a0a] shadow-2xl shadow-black">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06]">
        <div>
          <p className="text-xs font-semibold text-[#f0f0f0]">AI Assistant</p>
          <p className="text-[9px] text-[#555]">claude-sonnet-4-6 · OpenRouter</p>
        </div>
        <div className="flex items-center gap-1">
          {chatMessages.length > 0 && (
            <button
              onClick={clearChat}
              className="text-[10px] text-[#555] hover:text-[#f0f0f0] px-2 py-1 rounded transition-colors"
            >
              Clear
            </button>
          )}
          <button
            onClick={() => setChatOpen(false)}
            className="text-[#555] hover:text-[#f0f0f0] w-7 h-7 flex items-center justify-center rounded transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 min-h-0">
        {chatMessages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            <p className="text-xs text-[#555] max-w-[200px]">
              Ask anything about the scene — detected objects, distances, or spatial relationships.
            </p>
            <div className="flex flex-col gap-1.5 w-full mt-2">
              {[
                'How many cars are detected?',
                "What's closest to the ego vehicle?",
                'Any pedestrians within 20m?',
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-[10px] text-left px-3 py-2 rounded bg-[#111] border border-white/[0.06] text-[#555] hover:text-[#f0f0f0] hover:border-white/[0.12] transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {chatMessages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      <ChatInput onSend={sendMessage} />
    </div>
  )
}
