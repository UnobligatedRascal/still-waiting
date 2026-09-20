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
      const pct = total ? Math.round((parseInt(used) / parseInt(total)) * 100) : 0
      return { idx: parseInt(idx), name, used: parseInt(used), total: parseInt(total), temp: parseInt(temp), pct }
    }).filter(g => g.total > 0)
  }

  const gpus = parseGpuInfo(status.gpu_info)

  return (
    <div className="mt-6 rounded-lg border border-gray-800 bg-panel p-4">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-bold text-gray-400">
        <span>⚙ System</span>
      </h3>

      <div className="grid gap-4 md:grid-cols-3">
        {/* Orchestrator info */}
        <div className="space-y-2 text-xs">
          <div className="font-semibold text-kepler">Orchestrator</div>
          <div>
            Version: <span className="text-gray-400">{status.orchestrator.version}</span>
          </div>
          <div>
            Uptime: <span className="text-gray-400">{formatUptime(status.orchestrator.uptime_seconds)}</span>
          </div>
          <div>
            Bind: <span className="text-gray-400">{status.orchestrator.bind_addr}</span>
          </div>
        </div>

        {/* Job counts */}
        <div className="space-y-2 text-xs">
          <div className="font-semibold text-kepler">Jobs</div>
          <div>Total: <span className="text-gray-400">{status.jobs.total}</span></div>
          <div className="flex gap-3">
            <span className="text-green-400">▸ {status.jobs.running} running</span>
            <span className="text-yellow-400">▸ {status.jobs.queued} queued</span>
          </div>
          <div className="flex gap-3">
            <span className="text-orange-400">▸ {status.jobs.paused} paused</span>
            <span className="text-blue-400">▸ {status.jobs.completed} done</span>
            {status.jobs.failed > 0 && <span className="text-red-400">▸ {status.jobs.failed} failed</span>}
          </div>
        </div>

        {/* GPU info */}
        {gpus && (
          <div className="space-y-2 text-xs">
            <div className="font-semibold text-kepler">GPUs ({gpus.length})</div>
            <div className="max-h-24 overflow-y-auto space-y-1">
              {gpus.map(g => (
                <div key={g.idx} className="flex items-center justify-between">
                  <span className="text-gray-400">#{g.idx}</span>
                  <span className="w-20 text-gray-500">{g.used}MB/{g.total}MB</span>
                  <div className="w-16 h-1.5 rounded-full bg-gray-700 overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${g.pct}%`,
                        backgroundColor: g.pct > 90 ? '#ef4444' : g.pct > 60 ? '#e8491d' : '#22c55e',
                      }}
                    />
                  </div>
                  <span className="text-gray-500 w-10 text-right">{g.temp}°C</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Restart instructions */}
      <div className="mt-3 rounded border border-gray-800 bg-dark/50 p-2 text-[10px] text-gray-600">
        <span className="text-gray-500 font-semibold">Restart orchestrator:</span>
        <code className="ml-1 text-gray-400">sudo /home/whistler/still-waiting/deploy/start_still_waiting.sh restart</code>
        <br />
        <span className="text-gray-500 font-semibold mt-1 block">Systemd:</span>
        <code className="ml-1 text-gray-400">sudo systemctl restart still-waiting-orchestrator</code>
      </div>
    </div>
  )
}
