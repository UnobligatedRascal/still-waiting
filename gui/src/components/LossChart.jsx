import { useEffect, useRef, useState } from 'react'
import { createChart } from 'lightweight-charts'

/*
 * UnobligatedRascal — still-waiting LossChart
 * Phase 2: live loss curve using lightweight-charts (TradingView).
 */

const CHART_WIDTH = 600
const CHART_HEIGHT = 180

// Kepler orange as primary series color
const KEPLER_ORANGE = '#e8491d'

export default function LossChart({ checkpoints, autoRefresh = true }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  const [error, setError] = useState(null)

  // Convert checkpoints to chart data points
  const data = useMemoizedChartData(checkpoints)

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: CHART_WIDTH,
      height: CHART_HEIGHT,
      layout: {
        background: { type: 'solid', color: '#0a0a0f' },
        textColor: '#9ca3af',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(55, 55, 55, 0.3)' },
        horzLines: { color: 'rgba(55, 55, 55, 0.3)' },
      },
      crosshair: {
        mode: 0, // normal
        vertLine: { color: 'rgba(232, 73, 29, 0.4)', width: 1, labelBackgroundColor: '#e8491d' },
        horzLine: { color: 'rgba(232, 73, 29, 0.4)', width: 1, labelBackgroundColor: '#e8491d' },
      },
      rightPriceScale: {
        borderColor: 'rgba(55, 55, 55, 0.5)',
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
      timeScale: {
        borderColor: 'rgba(55, 55, 55, 0.5)',
        timeVisible: false,
        rightOffset: 1,
        barSpacing: 8,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: false,
      },
    })

    const areaSeries = chart.addAreaSeries({
      topColor: 'rgba(232, 73, 29, 0.4)',
      bottomColor: 'rgba(232, 73, 29, 0.05)',
      lineColor: KEPLER_ORANGE,
      lineWidth: 2,
      priceLineVisible: false,
    })

    chartRef.current = chart
    seriesRef.current = areaSeries

    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Update data when checkpoints change
  useEffect(() => {
    if (!seriesRef.current || data.length === 0) return

    try {
      seriesRef.current.setData(data)
      seriesRef.current.applyOptions({
        lastValueVisible: true,
        priceLineVisible: true,
      })

      // Auto-scroll to latest
      if (chartRef.current) {
        chartRef.current.timeScale().scrollToPosition(data.length - 2, false)
      }
    } catch (err) {
      setError(err.message)
    }
  }, [data])

  // Resize handler
  useEffect(() => {
    if (!chartRef.current || !containerRef.current) return

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect
        if (width > 0) {
          chartRef.current.applyOptions({ width: Math.max(width, 300) })
        }
      }
    })

    resizeObserver.observe(containerRef.current)
    return () => resizeObserver.disconnect()
  }, [])

  if (error) {
    return <div className="text-xs text-red-400">Chart error: {error}</div>
  }

  if (!checkpoints || checkpoints.length < 2) {
    return (
      <div className="rounded border border-gray-800 bg-dark/50 px-3 py-4 text-center text-xs text-gray-600">
        {checkpoints && checkpoints.length === 1
          ? '1 checkpoint — need 2+ for a loss curve'
          : 'No checkpoints yet — loss chart will appear when training reports'}
      </div>
    )
  }

  return (
    <div className="overflow-hidden rounded border border-gray-800 bg-dark">
      <div ref={containerRef} style={{ width: '100%', height: CHART_HEIGHT }} />
    </div>
  )
}

// Module-level cache for chart data
let _chartCache = { key: null, data: [] }

function useMemoizedChartData(checkpoints) {
  if (!checkpoints || checkpoints.length === 0) return []

  const key = `${checkpoints.length}-${checkpoints[checkpoints.length - 1]?.step}-${checkpoints[checkpoints.length - 1]?.metrics?.loss}`
  if (_chartCache.key === key) return _chartCache.data

  const newData = checkpoints
    .filter((cp) => cp.metrics?.loss != null)
    .map((cp, i) => ({
      time: i,
      value: parseFloat(cp.metrics.loss),
      step: cp.step,
    }))

  _chartCache = { key, data: newData }
  return newData
}
