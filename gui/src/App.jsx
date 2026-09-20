import { useState, useEffect } from 'react'
import JobsTable from './components/JobsTable'
import JobForm from './components/JobForm'
import StatusBanner from './components/StatusBanner'

const ORCH_URL = import.meta.env.VITE_ORCH_URL || '/v1'

export default function App() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showForm, setShowForm] = useState(false)
  const [selectedJob, setSelectedJob] = useState(null)

  const fetchJobs = async () => {
    try {
      const res = await fetch(`${ORCH_URL}/training/jobs`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setJobs(data.data || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 5000)
    return () => clearInterval(interval)
  }, [])

  const handleJobCreated = () => {
    setShowForm(false)
    fetchJobs()
  }

  const getStatusColor = (status) => {
    const colors = {
      queued: 'text-yellow-400',
      running: 'text-green-400',
      paused: 'text-orange-400',
      completed: 'text-blue-400',
      failed: 'text-red-400',
      checkpointed: 'text-purple-400',
      surgically_edited: 'text-pink-400',
    }
    return colors[status] || 'text-gray-400'
  }

  return (
    <div className="min-h-screen bg-dark text-gray-200">
      {/* Header */}
      <header className="border-b border-gray-800 bg-panel px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-3 w-3 rounded-full bg-kepler animate-pulse" />
            <h1 className="text-xl font-bold tracking-tight">
              still-waiting <span className="text-kepler">◈</span>
            </h1>
            <span className="text-xs text-gray-500">NOUGHT Training Control</span>
          </div>
          <div className="flex items-center gap-4">
            <StatusBanner jobs={jobs} />
            <button
              onClick={() => setShowForm(!showForm)}
              className="rounded-md bg-kepler px-4 py-1.5 text-sm font-medium text-white hover:bg-kepler/80"
            >
              + New Job
            </button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="p-6">
        {error && (
          <div className="mb-4 rounded-md border border-red-800 bg-red-900/20 px-4 py-2 text-sm text-red-400">
            Connection error: {error}. Is NOUGHT orchestrator running on port 9999?
          </div>
        )}

        {showForm && (
          <JobForm onCreate={handleJobCreated} onCancel={() => setShowForm(false)} />
        )}

        {loading ? (
          <div className="flex items-center gap-2 text-gray-500">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-gray-600 border-t-kepler" />
            Loading jobs...
          </div>
        ) : jobs.length === 0 ? (
          <div className="rounded-lg border border-gray-800 bg-panel px-6 py-12 text-center">
            <p className="text-gray-400">No training jobs yet.</p>
            <p className="mt-2 text-sm text-gray-600">
              Click "New Job" to start training on Kepler.
            </p>
          </div>
        ) : (
          <JobsTable
            jobs={jobs}
            getStatusColor={getStatusColor}
            onSelect={setSelectedJob}
          />
        )}
      </main>

      {/* Job detail panel */}
      {selectedJob && (
        <JobDetail
          job={selectedJob}
          onClose={() => setSelectedJob(null)}
          onRefresh={fetchJobs}
        />
      )}

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-2 text-xs text-gray-600">
        Built by UnobligatedRascal — Ancient hardware, fresh ambition.
      </footer>
    </div>
  )
}

function JobDetail({ job, onClose, onRefresh }) {
  const handleAction = async (action) => {
    try {
      await fetch(`/v1/training/jobs/${job.id}/${action}`, { method: 'POST' })
      onRefresh()
    } catch (err) {
      alert(`Action failed: ${err.message}`)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className="w-full max-w-2xl rounded-lg border border-gray-700 bg-panel p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-bold">{job.model_ref}</h2>
            <p className="text-xs text-gray-500">ID: {job.id}</p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white">✕</button>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Status:</span>{' '}
            <span className={job.status === 'running' ? 'text-green-400' : 'text-gray-300'}>
              {job.status}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Progress:</span>{' '}
            {job.current_step} / {job.target_steps || '∞'} steps
          </div>
          <div className="text-gray-500">Created:</div>
          <div>{new Date(job.created_at).toLocaleString()}</div>
        </div>

        {/* Checkpoints */}
        {job.checkpoints && job.checkpoints.length > 0 && (
          <div className="mt-4">
            <h3 className="text-sm font-semibold text-gray-400">Checkpoints</h3>
            <div className="mt-2 max-h-48 overflow-y-auto">
              {job.checkpoints.map((cp) => (
                <div key={cp.id} className="flex justify-between border-b border-gray-800 py-1 text-xs">
                  <span>Step {cp.step}</span>
                  <span className="text-gray-500">
                    loss: {cp.metrics?.loss?.toFixed(4) || 'N/A'}
                  </span>
                  <span className="text-gray-600">{new Date(cp.created_at).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="mt-4 flex gap-2">
          {job.status === 'running' && (
            <button
              onClick={() => handleAction('pause')}
              className="rounded bg-yellow-600/20 px-3 py-1 text-xs text-yellow-400 hover:bg-yellow-600/40"
            >
              Pause
            </button>
          )}
          {job.status === 'paused' && (
            <button
              onClick={() => handleAction('resume')}
              className="rounded bg-green-600/20 px-3 py-1 text-xs text-green-400 hover:bg-green-600/40"
            >
              Resume
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
