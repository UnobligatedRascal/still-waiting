import { useState, useMemo, useRef, useEffect } from 'react'

/*
 * UnobligatedRascal — still-waiting JobForm
 * Phase 1: Progressive disclosure, model picker, VRAM estimates, presets.
 */

// Pre-listed models safe for Kepler K80 (sm_37, ~11.5GB per chip, partial GPUs ~1.5-2GB)
const KNOWN_MODELS = [
  {
    name: 'Qwen/Qwen2.5-0.5B-Instruct',
    params: 0.5,
    vramF32: 1.0,
    badge: 'fits partial GPUs',
    badgeClass: 'text-green-400',
    recommended: true,
  },
  {
    name: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    params: 1.1,
    vramF32: 2.2,
    badge: 'tight on partial GPU',
    badgeClass: 'text-yellow-400',
  },
  {
    name: 'Qwen/Qwen2.5-1.5B-Instruct',
    params: 1.5,
    vramF32: 3.0,
    badge: 'needs full K80',
    badgeClass: 'text-orange-400',
  },
  {
    name: 'meta-llama/Llama-3.2-3B-Instruct',
    params: 3,
    vramF32: 6.0,
    badge: 'needs full K80',
    badgeClass: 'text-orange-400',
  },
  {
    name: 'Qwen/Qwen2.5-7B-Instruct',
    params: 7,
    vramF32: 14.0,
    badge: 'will OOM (single GPU)',
    badgeClass: 'text-red-400',
  },
]

// Preset configs — one-click Kepler-safe defaults
const PRESETS = [
  {
    id: 'quick-test',
    label: 'Quick Test',
    desc: 'Minimal config for validation',
    config: {
      model: 'Qwen/Qwen2.5-0.5B-Instruct',
      targetSteps: 100,
      datasetPath: '',
      maxSeqLength: 256,
      batchSize: 2,
      gradientAccumulation: 4,
      learningRate: 2e-4,
      loraR: 8,
      loraAlpha: 16,
      warmupRatio: 0.05,
      checkpointInterval: 50,
    },
  },
  {
    id: 'kepler-safe',
    label: 'Kepler Safe',
    desc: 'Conservative defaults for sub-billion models',
    config: {
      model: 'Qwen/Qwen2.5-0.5B-Instruct',
      targetSteps: 1000,
      datasetPath: '',
      maxSeqLength: 512,
      batchSize: 2,
      gradientAccumulation: 8,
      learningRate: 1e-4,
      loraR: 16,
      loraAlpha: 32,
      warmupRatio: 0.1,
      checkpointInterval: 200,
    },
  },
  {
    id: 'conversational',
    label: 'Conversational',
    desc: 'Chat-style fine-tuning (longer context)',
    config: {
      model: 'Qwen/Qwen2.5-0.5B-Instruct',
      targetSteps: 2000,
      datasetPath: '',
      maxSeqLength: 1024,
      batchSize: 1,
      gradientAccumulation: 16,
      learningRate: 5e-5,
      loraR: 16,
      loraAlpha: 32,
      warmupRatio: 0.1,
      checkpointInterval: 256,
      targetModules: 'q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
    },
  },
]

// Estimate VRAM: model weights + optimizer states (2x for AdamW) + batch buffer
const estimateVRAM = (modelSizeGB, batchSize, seqLength, loraR) => {
  const model = modelSizeGB || 1.0
  const optimizer = model * 2 // AdamW states
  const batch = Math.min(0.5, (batchSize * seqLength) / 8192)
  // LoRA adds minimal overhead
  const lora = model * 0.05 * (loraR / 16)
  return Math.round((model + optimizer + batch + lora) * 100) / 100
}

// Searchable combobox for model selection
function ModelPicker({ value, onChange }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const ref = useRef(null)

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = useMemo(() => {
    const q = query.toLowerCase()
    if (!q) return KNOWN_MODELS
    return KNOWN_MODELS.filter((m) => m.name.toLowerCase().includes(q))
  }, [query])

  const selected = useMemo(() => {
    return KNOWN_MODELS.find((m) => m.name === value) || null
  }, [value])

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between rounded-md border border-gray-700 bg-dark px-3 py-2 text-sm text-left focus:border-kepler focus:outline-none"
      >
        <span className="truncate">
          {selected ? selected.name : (query || 'Select or type a model...')}
        </span>
        <span className="ml-2 text-gray-500">{open ? '▴' : '▾'}</span>
      </button>

      {open && (
        <div className="absolute z-40 mt-1 w-full rounded-md border border-gray-700 bg-panel shadow-xl">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search models..."
            className="w-full border-b border-gray-700 bg-dark px-3 py-2 text-xs text-gray-300 placeholder:text-gray-600 focus:border-kepler focus:outline-none"
            autoFocus
          />
          <div className="max-h-48 overflow-y-auto">
            {filtered.length === 0 && (
              <div className="px-3 py-2 text-xs text-gray-500">
                Type a HuggingFace model path (e.g., Qwen/Qwen2.5-0.5B-Instruct)
              </div>
            )}
            {filtered.map((m) => (
              <button
                key={m.name}
                type="button"
                onClick={() => {
                  onChange(m.name)
                  setQuery('')
                  setOpen(false)
                }}
                className={`flex w-full items-center justify-between px-3 py-2 text-xs hover:bg-gray-800 ${
                  value === m.name ? 'bg-kepler/20 text-kepler' : 'text-gray-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="truncate">{m.name}</span>
                  {m.recommended && (
                    <span className="rounded bg-green-900/30 px-1.5 py-0.5 text-[9px] text-green-400">
                      recommended
                    </span>
                  )}
                </div>
                <span className={`ml-2 text-[9px] ${m.badgeClass}`}>
                  {m.badge}
                </span>
              </button>
            ))}
            {query && !KNOWN_MODELS.find((m) => m.name === query) && (
              <button
                type="button"
                onClick={() => {
                  onChange(query)
                  setQuery('')
                  setOpen(false)
                }}
                className="flex w-full items-center justify-between border-t border-gray-700 px-3 py-2 text-xs hover:bg-gray-800 text-gray-400"
              >
                <span>Use custom: {query}</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default function JobForm({ onCreate, onCancel, systemStatus }) {
  // Basic state
  const [model, setModel] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [targetSteps, setTargetSteps] = useState('1000')
  const [datasetPath, setDatasetPath] = useState('')
  const [presetId, setPresetId] = useState(null)

  // Advanced state
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [maxSeqLength, setMaxSeqLength] = useState('512')
  const [batchSize, setBatchSize] = useState('2')
  const [gradientAccumulation, setGradientAccumulation] = useState('8')
  const [learningRate, setLearningRate] = useState('1e-4')
  const [loraR, setLoraR] = useState('16')
  const [loraAlpha, setLoraAlpha] = useState('32')
  const [warmupRatio, setWarmupRatio] = useState('0.1')
  const [checkpointInterval, setCheckpointInterval] = useState('2048')
  const [targetModules, setTargetModules] = useState('')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Derived values
  const selectedModelInfo = useMemo(
    () => KNOWN_MODELS.find((m) => m.name === model) || null,
    [model]
  )

  const vramEstimate = useMemo(() => {
    const modelSize = selectedModelInfo?.vramF32 || 2.0
    return estimateVRAM(
      modelSize,
      parseInt(batchSize) || 2,
      parseInt(maxSeqLength) || 512,
      parseInt(loraR) || 16
    )
  }, [model, batchSize, maxSeqLength, loraR, selectedModelInfo])

  const vramBadge = useMemo(() => {
    if (vramEstimate <= 2.5)
      return { text: `~${vramEstimate}GB — safe on partial GPUs ✓`, color: 'text-green-400' }
    if (vramEstimate <= 11)
      return { text: `~${vramEstimate}GB — needs full K80`, color: 'text-orange-400' }
    return { text: `~${vramEstimate}GB — will OOM on single K80`, color: 'text-red-400' }
  }, [vramEstimate])

  const applyPreset = (preset) => {
    setModel(preset.config.model)
    setTargetSteps(preset.config.targetSteps.toString())
    setDatasetPath(preset.config.datasetPath)
    setMaxSeqLength(preset.config.maxSeqLength.toString())
    setBatchSize(preset.config.batchSize.toString())
    setGradientAccumulation(preset.config.gradientAccumulation.toString())
    setLearningRate(preset.config.learningRate.toString())
    setLoraR(preset.config.loraR.toString())
    setLoraAlpha(preset.config.loraAlpha.toString())
    setWarmupRatio(preset.config.warmupRatio.toString())
    setCheckpointInterval(preset.config.checkpointInterval.toString())
    setTargetModules(preset.config.targetModules || '')
    setPresetId(preset.id)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const config = {
        dataset_path: datasetPath || null,
        lora_r: parseInt(loraR) || 16,
        lora_alpha: parseInt(loraAlpha) || 32,
        max_seq_length: parseInt(maxSeqLength) || 512,
        batch_size: parseInt(batchSize) || 2,
        gradient_accumulation_steps: parseInt(gradientAccumulation) || 8,
        learning_rate: parseFloat(learningRate) || 1e-4,
        warmup_ratio: parseFloat(warmupRatio) || 0.1,
        checkpoint_interval: parseInt(checkpointInterval) || 2048,
      }

      if (targetModules) {
        config.target_modules = targetModules
      }

      const res = await fetch('/v1/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          target_steps: parseInt(targetSteps) || 1000,
          config,
        }),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.message || `HTTP ${res.status}`)
      }
      onCreate()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mb-6 rounded-lg border border-kepler/50 bg-panel p-5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-bold text-kepler">New Training Job</h3>

        {/* Preset selector */}
        <div className="flex items-center gap-2">
          <label className="text-[10px] text-gray-500">Preset:</label>
          <select
            value={presetId || ''}
            onChange={(e) => {
              const p = PRESETS.find((pr) => pr.id === e.target.value)
              if (p) applyPreset(p)
            }}
            className="rounded border border-gray-700 bg-dark px-2 py-1 text-[10px] text-gray-400 focus:border-kepler focus:outline-none"
          >
            <option value="">— choose —</option>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label} — {p.desc}
              </option>
            ))}
          </select>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* QUICK FORM — always visible */}
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <label className="mb-1 block text-xs text-gray-500">Model</label>
            <ModelPicker value={model} onChange={setModel} />
            {selectedModelInfo && (
              <div className={`mt-1 text-[9px] ${selectedModelInfo.badgeClass}`}>
                {selectedModelInfo.badge} ({selectedModelInfo.params}B params)
              </div>
            )}
          </div>

          <div>
            <label className="mb-1 block text-xs text-gray-500">Target Steps</label>
            <input
              type="number"
              value={targetSteps}
              onChange={(e) => setTargetSteps(e.target.value)}
              className="w-full rounded-md border border-gray-700 bg-dark px-3 py-2 text-sm focus:border-kepler focus:outline-none"
              min="10"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs text-gray-500">
              Dataset Path <span className="text-gray-600">(optional)</span>
            </label>
            <input
              type="text"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="w-full rounded-md border border-gray-700 bg-dark px-3 py-2 text-sm focus:border-kepler focus:outline-none"
              placeholder="/data/training.jsonl or user/dataset"
            />
          </div>
        </div>

        {/* VRAM estimate */}
        <div className={`mt-3 flex items-center gap-2 text-[10px] ${vramBadge.color}`}>
          <span className="font-semibold">VRAM estimate:</span> {vramBadge.text}
        </div>

        {/* Advanced toggle */}
        <div className="mt-4">
          <button
            type="button"
            onClick={() => setAdvancedOpen(!advancedOpen)}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-kepler"
          >
            <span>{advancedOpen ? '▾' : '▸'}</span> Advanced settings
          </button>
        </div>

        {/* ADVANCED FORM — collapsible */}
        {advancedOpen && (
          <div className="mt-4 grid gap-4 rounded-md border border-gray-800 bg-dark/50 p-4 md:grid-cols-3">
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">Max Sequence Length</label>
              <input
                type="number"
                value={maxSeqLength}
                onChange={(e) => setMaxSeqLength(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">Batch Size</label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
                min="1"
              />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">Gradient Accumulation</label>
              <input
                type="number"
                value={gradientAccumulation}
                onChange={(e) => setGradientAccumulation(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
                min="1"
              />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">Learning Rate</label>
              <input
                type="text"
                value={learningRate}
                onChange={(e) => setLearningRate(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
                placeholder="1e-4"
              />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">LoRA Rank (r)</label>
              <input
                type="number"
                value={loraR}
                onChange={(e) => setLoraR(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
                min="1"
              />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">LoRA Alpha</label>
              <input
                type="number"
                value={loraAlpha}
                onChange={(e) => setLoraAlpha(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">Warmup Ratio</label>
              <input
                type="text"
                value={warmupRatio}
                onChange={(e) => setWarmupRatio(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
                placeholder="0.1"
              />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">Checkpoint Interval</label>
              <input
                type="number"
                value={checkpointInterval}
                onChange={(e) => setCheckpointInterval(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-[10px] text-gray-500">Target Modules (comma-separated)</label>
              <input
                type="text"
                value={targetModules}
                onChange={(e) => setTargetModules(e.target.value)}
                className="w-full rounded border border-gray-700 bg-panel px-2 py-1.5 text-xs focus:border-kepler focus:outline-none"
                placeholder="q_proj,v_proj,k_proj,o_proj"
              />
            </div>
          </div>
        )}

        {error && <div className="mt-3 text-xs text-red-400">Error: {error}</div>}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-gray-700 px-4 py-1.5 text-sm hover:bg-gray-800"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading}
            className="rounded-md bg-kepler px-4 py-1.5 text-sm font-medium text-white hover:bg-kepler/80 disabled:opacity-50"
          >
            {loading ? 'Creating...' : 'Create Job'}
          </button>
        </div>
      </form>

      <div className="mt-3 flex items-start gap-2 text-[10px] text-gray-600">
        <span>ℹ</span>
        <span>
          GPUs 1,2,5,6 available (~1.5-2GB free each). Use sub-billion models for testing.
          Training uses F32 (Kepler constraint). NUMA pinning applied automatically by worker.
        </span>
      </div>
    </div>
  )
}
