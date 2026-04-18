import clsx from 'clsx'

export default function GlassPanel({ children, className, elevated = false }) {
  return (
    <div className={clsx(elevated ? 'amoled-panel-elevated' : 'amoled-panel', 'overflow-hidden', className)}>
      {children}
    </div>
  )
}
