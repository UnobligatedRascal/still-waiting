import { useState, useEffect } from 'react'

export default function SystemStatus() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/v1/system/status')
        if (res.ok) setStatus(await res.json())
      } catch { /* ignore */ } finally {
        setLoading(false)
      }
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return null

  const formatUptime = (secs) => {
    const d = Math.floor(secs / 86400)
    const h = Math.floor((secs % 86400) / 3600)
    const m = Math.floor((secs % 3600) / 60)
    if (d > 0) return `${d}d ${h}h ${m}m`
    if (h > 0) return `${h}h ${m}m`
    return `${m}m`
  }

  const parseGpuInfo = (info) => {
    if (!info) return null
    return info.trim().split('\n').map(line => {
      const [idx, name, used, total, temp] = line.split(',').map(s => s.trim())
      const usedMb = parseInt(used)
      const totalMb = parseInt(total)
      const freeMb = totalMb - usedMb
      const pct = totalMb ? Math.round((usedMb / totalMb) * 100) : 0
      const shortName = name.replace('Tesla K80', 'K80').trim()
      return { idx: parseInt(idx), name: shortName, used: usedMb, total: totalMb, free: freeMb, temp: parseInt(temp), pct }
    }).filter(g => g.total > 0)
  }

  const gpus = parseGpuInfo(status.gpu_info)

  return (
    <div className="mt-6 rounded-lg border border-gray-800 bg-panel p-4">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-bold">
        <span className="text-kepler">⚙</span>
        <span className="text-gray-400">System Status</span>
        {loading && <span className="ml-auto text-xs text-gray-600">updating...</span>}
      </h3>

      <div className="grid gap-4 md:grid-cols-3">
        {/* Orchestrator info */}
        <div className="space-y-1.5 text-xs">
          <div className="font-semibold text-kepler">Orchestrator</div>
          <div className="flex justify-between">
            <span className="text-gray-500">Version</span>
            <span className="text-gray-400">{status.orchestrator.version}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Uptime</span>
            <span className="text-green-400">{formatUptime(status.orchestrator.uptime_seconds)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Bind</span>
            <span className="text-gray-600">{status.orchestrator.bind_addr}</span>
          </div>
        </div>

        {/* Job counts */}
        <div className="space-y-1.5 text-xs">
          <div className="font-semibold text-kepler">Jobs</div>
          <div className="flex justify-between">
            <span className="text-gray-500">Total</span>
            <span className="text-gray-300">{status.jobs.total}</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {status.jobs.running > 0 && (
              <span className="rounded-full bg-green-900/20 px-2 py-0.5 text-[9px] text-green-400">
                {status.jobs.running} running
              </span>
            )}
            {status.jobs.queued > 0 && (
              <span className="rounded-full bg-yellow-900/20 px-2 py-0.5 text-[9px] text-yellow-400">
                {status.jobs.queued} queued
              </span>
            )}
            {status.jobs.paused > 0 && (
              <span className="rounded-full bg-orange-900/20 px-2 py-0.5 text-[9px] text-orange-400">
                {status.jobs.paused} paused
              </span>
            )}
            {status.jobs.completed > 0 && (
              <span className="rounded-full bg-blue-900/20 px-2 py-0.5 text-[9px] text-blue-400">
                {status.jobs.completed} done
              </span>
            )}
            {status.jobs.failed > 0 && (
              <span className="rounded-full bg-red-900/20 px-2 py-0.5 text-[9px] text-red-400">
                {status.jobs.failed} failed
              </span>
            )}
          </div>
        </div>

        {/* GPU info */}
        {gpus && (
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-kepler">GPUs ({gpus.length})</span>
              <span className="text-[9px] text-gray-600">Tesla K80 (sm_37)</span>
            </div>
            <div className="max-h-32 overflow-y-auto space-y-1">
              {gpus.map(g => {
                const freeGb = (g.free / 1024).toFixed(1)
                const statusColor = g.pct > 90 ? '#ef4444' : g.pct > 60 ? '#e8491d' : '#22c55e'
                const statusText = g.pct < 30 ? 'idle' : g.pct < 70 ? 'warm' : 'hot'

                return (
                  <div key={g.idx} className="flex items-center gap-2">
                    <span className="w-5 text-gray-400">#{g.idx}</span>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-[9px] text-gray-500">
                          {freeGb}GB free
                        </span>
                        <span
                          className="text-[8px]"
                          style={{ color: statusColor }}
                        >
                          {statusText}
                        </span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-gray-700 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${g.pct}%`,
                            backgroundColor: statusColor,
                          }}
                        />
                      </div>
                    </div>
                    <span className="w-8 text-[9px] text-gray-500 text-right">
                      {g.temp}°
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* Restart instructions */}
      <div className="mt-3 flex flex-wrap gap-3 rounded border border-gray-800 bg-dark/50 p-2 text-[10px] text-gray-600">
        <div>
          <span className="text-gray-500">Script:</span>{' '}
          <code className="text-gray-400">sudo ./deploy/start_still_waiting.sh restart</code>
        </div>
        <div>
          <span className="text-gray-500">Systemd:</span>{' '}
          <code className="text-gray-400">sudo systemctl restart still-waiting-orchestrator</code>
        </div>
      </div>
    </div>
  )
}
