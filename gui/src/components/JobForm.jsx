import { useState } from 'react'

export default function JobForm({ onCreate, onCancel }) {
  const [model, setModel] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [targetSteps, setTargetSteps] = useState('1000')
  const [datasetPath, setDatasetPath] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const config = {
        dataset_path: datasetPath || null,
        lora_r: 16,
        lora_alpha: 32,
        max_seq_length: 512,
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

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      onCreate()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mb-6 rounded-lg border border-kepler/50 bg-panel p-5">
      <h3 className="mb-4 text-sm font-bold text-kepler">New Training Job</h3>

      <form onSubmit={handleSubmit} className="grid gap-4 md:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs text-gray-500">Model (HuggingFace ref)</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full rounded-md border border-gray-700 bg-dark px-3 py-2 text-sm focus:border-kepler focus:outline-none"
            placeholder="Qwen/Qwen2.5-0.5B-Instruct"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs text-gray-500">Target Steps</label>
          <input
            type="number"
            value={targetSteps}
            onChange={(e) => setTargetSteps(e.target.value)}
            className="w-full rounded-md border border-gray-700 bg-dark px-3 py-2 text-sm focus:border-kepler focus:outline-none"
          />
        </div>

        <div className="md:col-span-2">
          <label className="mb-1 block text-xs text-gray-500">
            Dataset Path (optional — JSONL file, directory, or HF dataset)
          </label>
          <input
            type="text"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            className="w-full rounded-md border border-gray-700 bg-dark px-3 py-2 text-sm focus:border-kepler focus:outline-none"
            placeholder="/data/training.jsonl or user/dataset-name"
          />
        </div>

        {error && (
          <div className="md:col-span-2 text-xs text-red-400">Error: {error}</div>
        )}

        <div className="flex justify-end gap-2 md:col-span-2">
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

      <p className="mt-3 text-[10px] text-gray-600">
        Note: Only GPUs 1,2,5,6 available (~1.5-2GB free each). Use sub-billion models for testing.
      </p>
    </div>
  )
}
