import useAppStore from '../../store/appStore'

export default function LidarBevTab() {
  const { result } = useAppStore()

  if (!result?.lidar_bev) {
    return (
      <div className="flex-1 flex items-center justify-center text-[#555]">
        <span className="text-sm">No LiDAR BEV yet — run inference first</span>
      </div>
    )
  }

  return (
    <div className="flex-1 flex items-center justify-center overflow-hidden bg-black">
      <img
        src={`data:image/png;base64,${result.lidar_bev}`}
        alt="LiDAR BEV (white background)"
        className="max-h-full max-w-full object-contain"
      />
    </div>
  )
}
