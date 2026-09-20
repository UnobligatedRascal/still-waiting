export default function StatusBanner({ jobs }) {
  const running = jobs.filter((j) => j.status === 'running').length
  const queued = jobs.filter((j) => j.status === 'queued').length
  const failed = jobs.filter((j) => j.status === 'failed').length
  const paused = jobs.filter((j) => j.status === 'paused').length

  const pills = []

  if (running > 0)
    pills.push(
      <span
        key="running"
        className="flex items-center gap-1.5 rounded-full border border-green-800/50 bg-green-900/20 px-2 py-0.5 text-xs text-green-400"
      >
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-400" />
        {running} running
      </span>
    )

  if (queued > 0)
    pills.push(
      <span
        key="queued"
        className="flex items-center gap-1.5 rounded-full border border-yellow-800/50 bg-yellow-900/20 px-2 py-0.5 text-xs text-yellow-400"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-yellow-400" />
        {queued} queued
      </span>
    )

  if (paused > 0)
    pills.push(
      <span
        key="paused"
        className="flex items-center gap-1.5 rounded-full border border-orange-800/50 bg-orange-900/20 px-2 py-0.5 text-xs text-orange-400"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-orange-400" />
        {paused} paused
      </span>
    )

  if (failed > 0)
    pills.push(
      <span
        key="failed"
        className="flex items-center gap-1.5 rounded-full border border-red-800/50 bg-red-900/20 px-2 py-0.5 text-xs text-red-400"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
        {failed} failed
      </span>
    )

  if (pills.length === 0) {
    return <span className="text-xs text-gray-600">No active jobs</span>
  }

  return <div className="flex flex-wrap gap-2">{pills}</div>
}
