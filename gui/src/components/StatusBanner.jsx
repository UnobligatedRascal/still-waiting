export default function StatusBanner({ jobs }) {
  const running = jobs.filter((j) => j.status === 'running').length
  const queued = jobs.filter((j) => j.status === 'queued').length
  const failed = jobs.filter((j) => j.status === 'failed').length

  return (
    <div className="flex gap-3 text-xs">
      <span className="flex items-center gap-1 text-green-400">
        <span className="h-2 w-2 rounded-full bg-green-400" />
        {running} running
      </span>
      {queued > 0 && (
        <span className="flex items-center gap-1 text-yellow-400">
          <span className="h-2 w-2 rounded-full bg-yellow-400" />
          {queued} queued
        </span>
      )}
      {failed > 0 && (
        <span className="flex items-center gap-1 text-red-400">
          <span className="h-2 w-2 rounded-full bg-red-400" />
          {failed} failed
        </span>
      )}
    </div>
  )
}
