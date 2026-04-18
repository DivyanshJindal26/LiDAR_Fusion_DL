import { useState, useRef } from 'react'
import useAppStore from '../../store/appStore'

export default function ChatInput({ onSend }) {
  const [text, setText] = useState('')
  const { chatLoading } = useAppStore()
  const textareaRef = useRef()

  function submit() {
    const trimmed = text.trim()
    if (!trimmed || chatLoading) return
    onSend(trimmed)
    setText('')
    textareaRef.current?.focus()
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex items-end gap-2 p-3 border-t border-white/[0.06]">
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Ask about the scene…"
        rows={1}
        disabled={chatLoading}
        className="flex-1 resize-none bg-[#111] border border-white/[0.06] rounded px-3 py-2 text-xs text-[#f0f0f0] placeholder-[#333] outline-none focus:border-[#00e676]/40 transition-colors disabled:opacity-50 min-h-[36px] max-h-[100px]"
        style={{ fieldSizing: 'content' }}
      />
      <button
        onClick={submit}
        disabled={!text.trim() || chatLoading}
        className="flex-shrink-0 w-8 h-8 rounded bg-[#00e676] hover:bg-[#00ff84] disabled:bg-[#111] disabled:text-[#555] disabled:border disabled:border-white/[0.06] text-black transition-colors flex items-center justify-center"
      >
        {chatLoading ? (
          <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        ) : (
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
          </svg>
        )}
      </button>
    </div>
  )
}
