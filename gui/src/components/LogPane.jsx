import { useState, useEffect, useRef } from 'react'

const ORCH_URL = import.meta.env.VITE_ORCH_URL || '/v1'

/* UnobligatedRascal — Real-time training logs */

export default function LogPane({ jobId, autoScroll = true }) {
  const [logs, setLogs] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [polling, setPolling] = useState(true)
  const scrollRef = useRef(null)
  const logsEndRef = useRef(null)

  // Track from offset for pagination
  const fromRef = useRef(0)

  const fetchLogs = async () => {
    if (!jobId || !polling) return
    try {
      const limit = 100
      const res = await fetch(
        `${ORCH_URL}/training/jobs/${jobId}/logs?from=${fromRef.current}&limit=${limit}`
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      
      setLogs((prev) => {
        const newLogs = data.logs || []
        if (newLogs.length > 0) {
          fromRef.current += newLogs.length
        }
        return newLogs.length > 0 ? [...prev, ...newLogs] : prev
      })
      setTotal(data.total || 0)
      setError(null)
    } catch (err) {
      if (!error) setError(err.message) // Only show first error
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLogs()
    const interval = setInterval(fetchLogs, 2000)
    return () => clearInterval(interval)
  }, [jobId, polling])

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  // Scroll handler: pause auto-scroll if user scrolls up
  const handleScroll = () => {
    const el = scrollRef.current
    if (el) {
      const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50
      setPolling(isAtBottom)
    }
  }

  const levelClass = (level) => {
    switch (level) {
      case 'ERROR':
        return 'text-red-400'
      case 'WARN':
        return 'text-yellow-400'
      case 'DEBUG':
        return 'text-gray-500'
      default:
        return 'text-gray-300'
    }
  }

  const levelBadge = (level) => {
    const badges = {
      ERROR: 'bg-red-900/30 text-red-400',
      WARN: 'bg-yellow-900/30 text-yellow-400',
      DEBUG: 'bg-gray-800 text-gray-500',
      INFO: 'bg-green-900/20 text-green-400',
    }
    return badges[level] || 'bg-gray-800 text-gray-400'
  }

  const formatTimestamp = (ts) => {
    const d = new Date(ts)
    return d.toLocaleTimeString()
  }

  const clearLogs = () => {
    setLogs([])
    fromRef.current = 0
  }

  return (
    <div className="flex flex-col">
      {/* Log pane header */}
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold text-gray-400">Training Logs</h3>
          {loading && (
            <div className="h-3 w-3 animate-spin rounded-full border-2 border-gray-600 border-t-kepler" />
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-600">
            {logs.length} / {total} lines
          </span>
          <button
            onClick={clearLogs}
            className="rounded px-2 py-0.5 text-xs text-gray-500 hover:bg-gray-800 hover:text-white"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="mb-2 text-xs text-red-500">
          {error} — retrying...
        </div>
      )}

      {/* Log entries */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="max-h-64 overflow-y-auto rounded border border-gray-800 bg-dark/70 font-mono text-xs"
      >
        {logs.length === 0 && !loading && (
          <div className="px-3 py-4 text-center text-gray-600">
            No logs yet. Waiting for worker output...
          </div>
        )}

        {logs.map((entry, i) => (
          <div
            key={i}
            className={`flex items-start gap-2 border-b border-gray-800/30 px-3 py-0.5 ${levelClass(
              entry.level
            )}`}
          >
            {/* Timestamp */}
            <span className="shrink-0 text-gray-600">
              {formatTimestamp(entry.timestamp)}
            </span>
            {/* Level badge */}
            <span
              className={`shrink-0 rounded px-1 py-px text-[10px] uppercase ${levelBadge(
                entry.level
              )}`}
            >
              {entry.level}
            </span>
            {/* Step hint */}
            {entry.step != null && (
              <span className="shrink-0 text-purple-400">
                [{entry.step}]
              </span>
            )}
            {/* Message */}
            <span className="break-all">{entry.message}</span>
          </div>
        ))}

        {/* Scroll anchor */}
        <div ref={logsEndRef} />
      </div>

      {/* Auto-scroll indicator */}
      <div className="mt-1 flex justify-end">
        {!polling && (
          <button
            onClick={() => {
              setPolling(true)
              logsEndRef.current?.scrollIntoView({ behavior: 'instant' })
            }}
            className="rounded bg-kepler/20 px-2 py-0.5 text-[10px] text-kepler hover:bg-kepler/40"
          >
            ↓ Show latest
          </button>
        )}
      </div>
    </div>
  )
}
