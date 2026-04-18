import Header from './components/layout/Header'
import UploadPanel from './components/upload/UploadPanel'
import TabBar from './components/tabs/TabBar'
import CameraTab from './components/tabs/CameraTab'
import LidarBevTab from './components/tabs/LidarBevTab'
import Scene3DTab from './components/tabs/Scene3DTab'
import MetricsTab from './components/tabs/MetricsTab'
import DetectionSidebar from './components/DetectionSidebar'
import FrameScrubber from './components/timeline/FrameScrubber'
import ChatPanel from './components/chatbot/ChatPanel'
import useAppStore from './store/appStore'

export default function App() {
  const { activeTab, bulkMode, bulkFrames } = useAppStore()

  const tabContent = {
    camera:  <CameraTab />,
    lidar:   <LidarBevTab />,
    scene3d: <Scene3DTab />,
    metrics: <MetricsTab />,
  }

  const showScrubber = bulkMode && bulkFrames.length > 0

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#000]">
      <Header />

      <div className="flex flex-1 min-h-0">
        {/* Left sidebar: upload */}
        <div className="w-52 border-r border-white/[0.06] bg-[#0a0a0a] flex flex-col flex-shrink-0">
          <UploadPanel />
        </div>

        {/* Main area */}
        <div className="flex-1 flex flex-col min-w-0">
          <TabBar />

          <div className="flex flex-1 min-h-0">
            {/* Tab content */}
            <div className="flex-1 flex flex-col min-h-0 min-w-0">
              <div className="flex-1 flex min-h-0">
                {tabContent[activeTab]}
              </div>
              {showScrubber && <FrameScrubber />}
            </div>

            {/* Right detection sidebar */}
            <DetectionSidebar />
          </div>
        </div>
      </div>

      <ChatPanel />
    </div>
  )
}
