import { useState, useEffect } from 'react'
import JobsTable from './components/JobsTable'
import JobForm from './components/JobForm'
import StatusBanner from './components/StatusBanner'
import SystemStatus from './components/SystemStatus'
import LossChart from './components/LossChart'
import LogPane from './components/LogPane'

const ORCH_URL = import.meta.env.VITE_ORCH_URL || '/v1'

/* UnobligatedRascal — still-waiting main app */

export default function App() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showForm, setShowForm] = useState(false)
  const [selectedJob, setSelectedJob] = useState(null)
  const [systemStatus, setSystemStatus] = useState(null)

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

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${ORCH_URL}/system/status`)
      if (res.ok) {
        const data = await res.json()
        setSystemStatus(data)
      }
    } catch {
      // Silently fail — SystemStatus component handles its own loading
    }
  }

  useEffect(() => {
    fetchJobs()
    fetchSystemStatus()
    const jobsInterval = setInterval(fetchJobs, 5000)
    const statusInterval = setInterval(fetchSystemStatus, 10000)
    return () => {
      clearInterval(jobsInterval)
      clearInterval(statusInterval)
    }
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

  const getStatusPillClass = (status) => {
    const classes = {
      queued: 'bg-yellow-900/20 border-yellow-800/50 text-yellow-400',
      running: 'bg-green-900/20 border-green-800/50 text-green-400',
      paused: 'bg-orange-900/20 border-orange-800/50 text-orange-400',
      completed: 'bg-blue-900/20 border-blue-800/50 text-blue-400',
      failed: 'bg-red-900/20 border-red-800/50 text-red-400',
      checkpointed: 'bg-purple-900/20 border-purple-800/50 text-purple-400',
      surgically_edited: 'bg-pink-900/20 border-pink-800/50 text-pink-400',
    }
    return classes[status] || 'bg-gray-900/20 border-gray-800/50 text-gray-400'
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
            <span className="hidden sm:inline text-xs text-gray-500">
              NOUGHT Training Control
            </span>
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
          <div className="mb-4 rounded-md border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
            <span className="font-semibold">Connection error:</span> {error}
            <br />
            <span className="text-xs text-red-500">
              Is NOUGHT orchestrator running on port 9999?
            </span>
          </div>
        )}

        {showForm && (
          <JobForm
            onCreate={handleJobCreated}
            onCancel={() => setShowForm(false)}
            systemStatus={systemStatus}
          />
        )}

        {loading ? (
          <SkeletonJobsTable />
        ) : jobs.length === 0 ? (
          <EmptyState />
        ) : (
          <JobsTable
            jobs={jobs}
            getStatusColor={getStatusColor}
            onSelect={setSelectedJob}
          />
        )}

        <SystemStatus />
      </main>

      {/* Job detail panel */}
      {selectedJob && (
        <JobDetail
          job={selectedJob}
          onClose={() => setSelectedJob(null)}
          onRefresh={fetchJobs}
          getStatusPillClass={getStatusPillClass}
        />
      )}

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-2 text-xs text-gray-600">
        Built by UnobligatedRascal — Ancient hardware, fresh ambition.
      </footer>
    </div>
  )
}

/* Skeleton loader for jobs table */
function SkeletonJobsTable() {
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-800 bg-panel">
      <div className="border-b border-gray-800 px-4 py-3">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-gray-600 border-t-kepler" />
          Loading jobs...
        </div>
      </div>
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="flex animate-pulse items-center justify-between border-b border-gray-800/50 px-4 py-3"
        >
          <div className="h-4 w-48 rounded bg-gray-800" />
          <div className="h-4 w-16 rounded bg-gray-800" />
          <div className="h-4 w-20 rounded bg-gray-800" />
          <div className="h-4 w-12 rounded bg-gray-800" />
          <div className="h-4 w-24 rounded bg-gray-800" />
        </div>
      ))}
    </div>
  )
}

/* Empty state with Sonic branding */
function EmptyState() {
  return (
    <div className="rounded-lg border border-gray-800 bg-panel px-6 py-12 text-center">
      {/* Sonic-style "waiting" illustration */}
      <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center">
        <div className="relative">
          {/* Tapping foot animation */}
          <div className="absolute -bottom-1 left-2 h-2 w-4 rounded-full bg-kepler/60 animate-bounce" />
          <div className="text-4xl text-kepler/80">⏳</div>
        </div>
      </div>
      <h3 className="text-lg font-bold text-gray-400">Still waiting...</h3>
      <p className="mt-2 text-sm text-gray-500">
        No training jobs yet. The Kepler GPUs are idle and ready.
      </p>
      <p className="mt-1 text-xs text-gray-600">
        Click <span className="rounded bg-kepler/20 px-1.5 py-0.5 text-kepler">+ New Job</span> to
        start training on NOUGHT.
      </p>
    </div>
  )
}

function JobDetail({ job, onClose, onRefresh, getStatusPillClass }) {
  const handleAction = async (action) => {
    try {
      await fetch(`/v1/training/jobs/${job.id}/${action}`, { method: 'POST' })
      onRefresh()
    } catch (err) {
      alert(`Action failed: ${err.message}`)
    }
  }

  const progress =
    job.target_steps && job.current_step
      ? Math.min(100, (job.current_step / job.target_steps) * 100)
      : 0

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl rounded-lg border border-gray-700 bg-panel p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-bold">{job.model_ref}</h2>
            <p className="text-xs text-gray-500">ID: {job.id}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md px-2 py-1 text-gray-500 hover:bg-gray-800 hover:text-white"
          >
            ✕
          </button>
        </div>

        {/* Status and progress */}
        <div className="mb-4 space-y-2">
          <div className="flex items-center gap-3">
            <span
              className={`rounded-full border px-2.5 py-0.5 text-xs ${getStatusPillClass(job.status)}`}
            >
              {job.status}
            </span>
            <span className="text-xs text-gray-500">
              {job.current_step} / {job.target_steps || '∞'} steps
            </span>
          </div>

          {/* Progress bar */}
          {job.target_steps && (
            <div className="h-2 w-full overflow-hidden rounded-full bg-gray-800">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  job.status === 'running'
                    ? 'bg-green-500 animate-pulse'
                    : job.status === 'failed'
                    ? 'bg-red-500'
                    : 'bg-kepler'
                }`}
                style={{ width: `${progress}%` }}
              />
            </div>
          )}
        </div>

        {/* Details grid */}
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <span className="text-gray-500">Created:</span>{' '}
            <span className="text-gray-300">
              {new Date(job.created_at).toLocaleString()}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Checkpoints:</span>{' '}
            <span className="text-purple-400">
              {job.checkpoints?.length || 0}
            </span>
          </div>
        </div>

        {/* Loss chart */}
        {job.checkpoints && job.checkpoints.length > 0 && (
          <div className="mt-4">
            <h3 className="mb-2 text-xs font-semibold text-gray-400">
              Loss Curve
            </h3>
            <LossChart checkpoints={job.checkpoints} />
          </div>
        )}

        {/* Checkpoints list */}
        {job.checkpoints && job.checkpoints.length > 0 && (
          <div className="mt-3">
            <h3 className="mb-2 text-xs font-semibold text-gray-400">
              Checkpoint Timeline
            </h3>
            <div className="max-h-36 overflow-y-auto rounded border border-gray-800 bg-dark/50">
              {job.checkpoints
                .slice()
                .reverse()
                .map((cp) => (
                  <div
                    key={cp.id}
                    className="flex items-center justify-between border-b border-gray-800/50 px-3 py-1.5 text-xs last:border-b-0"
                  >
                    <span className="text-gray-300">Step {cp.step}</span>
                    <span className="text-kepler">
                      loss:{' '}
                      {cp.metrics?.loss != null
                        ? cp.metrics.loss.toFixed(4)
                        : 'N/A'}
                    </span>
                    <span className="text-gray-600">
                      {new Date(cp.created_at).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Live logs */}
        <div className="mt-4">
          <LogPane jobId={job.id} />
        </div>

        {/* Actions */}
        <div className="mt-4 flex gap-2">
          {job.status === 'running' && (
            <button
              onClick={() => handleAction('pause')}
              className="rounded bg-yellow-600/20 px-3 py-1.5 text-xs text-yellow-400 hover:bg-yellow-600/40"
            >
              Pause
            </button>
          )}
          {job.status === 'paused' && (
            <button
              onClick={() => handleAction('resume')}
              className="rounded bg-green-600/20 px-3 py-1.5 text-xs text-green-400 hover:bg-green-600/40"
            >
              Resume
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
