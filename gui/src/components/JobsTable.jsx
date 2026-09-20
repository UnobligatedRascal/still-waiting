/* UnobligatedRascal — still-waiting JobsTable with status pills */

const STATUS_PILL = {
  queued: 'bg-yellow-900/20 border-yellow-800/50 text-yellow-400',
  running: 'bg-green-900/20 border-green-800/50 text-green-400',
  paused: 'bg-orange-900/20 border-orange-800/50 text-orange-400',
  completed: 'bg-blue-900/20 border-blue-800/50 text-blue-400',
  failed: 'bg-red-900/20 border-red-800/50 text-red-400',
  checkpointed: 'bg-purple-900/20 border-purple-800/50 text-purple-400',
  surgically_edited: 'bg-pink-900/20 border-pink-800/50 text-pink-400',
}

export default function JobsTable({ jobs, onSelect }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-800 bg-panel">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-gray-800 text-xs text-gray-500">
            <th className="px-4 py-3">Model</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Progress</th>
            <th className="px-4 py-3">Checkpoints</th>
            <th className="px-4 py-3">Created</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => {
            const pillClass = STATUS_PILL[job.status] || 'bg-gray-900/20 border-gray-800/50 text-gray-400'
            const isRunning = job.status === 'running'

            return (
              <tr
                key={job.id}
                onClick={() => onSelect(job)}
                className="cursor-pointer border-b border-gray-800/50 transition-colors hover:bg-gray-800/50"
              >
                <td className="px-4 py-3 font-medium">{job.model_ref}</td>
                <td className="px-4 py-3">
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs ${pillClass}`}
                  >
                    {isRunning && (
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-400" />
                    )}
                    {job.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-400">
                  {job.current_step} / {job.target_steps || '∞'}
                </td>
                <td className="px-4 py-3 text-purple-400">
                  {job.checkpoints?.length || 0}
                </td>
                <td className="px-4 py-3 text-xs text-gray-600">
                  {new Date(job.created_at).toLocaleString()}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
