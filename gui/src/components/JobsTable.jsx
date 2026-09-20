export default function JobsTable({ jobs, getStatusColor, onSelect }) {
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
          {jobs.map((job) => (
            <tr
              key={job.id}
              onClick={() => onSelect(job)}
              className="cursor-pointer border-b border-gray-800/50 transition-colors hover:bg-gray-800/50"
            >
              <td className="px-4 py-3 font-medium">{job.model_ref}</td>
              <td className={`px-4 py-3 ${getStatusColor(job.status)}`}>
                {job.status}
              </td>
              <td className="px-4 py-3 text-gray-400">
                {job.current_step} / {job.target_steps || '∞'}
              </td>
              <td className="px-4 py-3 text-gray-500">
                {job.checkpoints?.length || 0}
              </td>
              <td className="px-4 py-3 text-xs text-gray-600">
                {new Date(job.created_at).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
