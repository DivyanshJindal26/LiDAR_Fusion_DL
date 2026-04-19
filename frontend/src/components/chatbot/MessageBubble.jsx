import clsx from 'clsx'

function renderMarkdown(text) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\n)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**'))
      return <strong key={i} className="font-semibold text-slate-100">{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`'))
      return <code key={i} className="bg-slate-800 text-cyan-300 rounded px-1 text-xs">{part.slice(1, -1)}</code>
    if (part === '\n')
      return <br key={i} />
    return <span key={i}>{part}</span>
  })
}

export default function MessageBubble({ message }) {
  const { role, content, loading, error } = message
  const isUser = role === 'user'

  if (!loading && !content) return null

  return (
    <div className={clsx('flex flex-col gap-2 animate-fade-in', isUser ? 'items-end' : 'items-start')}>
      <div
        className={clsx(
          'max-w-[85%] rounded-2xl px-3.5 py-2.5 text-xs leading-relaxed',
          isUser
            ? 'bg-blue-500/20 border border-blue-500/30 text-blue-100 rounded-tr-sm'
            : error
            ? 'bg-red-500/10 border border-red-500/20 text-red-300 rounded-tl-sm'
            : 'glass border-slate-700/40 text-slate-200 rounded-tl-sm'
        )}
      >
        {loading ? (
          <span className="flex items-center gap-1.5 text-slate-500">
            <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '300ms' }} />
          </span>
        ) : (
          renderMarkdown(content)
        )}
      </div>
    </div>
  )
}
