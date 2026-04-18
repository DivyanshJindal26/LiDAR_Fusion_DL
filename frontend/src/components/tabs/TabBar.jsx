import useAppStore from '../../store/appStore'

const TABS = [
  { id: 'camera',  label: 'Camera' },
  { id: 'lidar',   label: 'LiDAR BEV' },
  { id: 'scene3d', label: '3D Scene' },
  { id: 'metrics', label: 'Metrics' },
]

export default function TabBar() {
  const { activeTab, setActiveTab } = useAppStore()

  return (
    <div className="flex border-b border-white/[0.06] bg-[#0a0a0a]">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={`px-5 py-2.5 text-xs font-medium tracking-wide transition-colors ${
            activeTab === tab.id ? 'tab-active' : 'tab-inactive'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}
