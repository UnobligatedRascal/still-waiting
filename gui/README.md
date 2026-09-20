# still-waiting GUI

Control panel for NOUGHT training operations. Built with Tauri (Rust backend) + SvelteKit (frontend).

## Architecture

```
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  GUI Client     │ HTTP   │  NOUGHT         │ SSH    │  Admin/Dev      │
│  (Tauri +       │ ◄────► │  orchestrator   │ ◄────► │  (TUI / direct) │
│   SvelteKit)    │ WS     │  (:8000)        │        │                 │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

## Setup

### Prerequisites
- Node.js 20+
- Rust (for Tauri)
- System deps (Windows): Windows SDK, WebView2 runtime

### Install
```bash
cd gui
npm install
npm run tauri dev
```

### Connect to NOUGHT
Set the orchestrator URL in `.env`:
```
VITE_ORCH_URL=http://192.168.137.29:8000
```

Or configure at runtime in settings.

## Features (planned)

- Job dashboard: live status of all training jobs
- Job creation wizard: model selection, LoRA config, dataset upload
- Checkpoint browser: view metrics, compare checkpoints
- Conductor panel: apply surgical edits, inject preferences
- GPU monitoring: VRAM, utilization per GPU (via nvidia-smi polling)
- Model library: list trained/exported models

## Notes

- GUI connects over network; NOUGHT runs headless
- WebSocket for real-time job status updates
- Tauri keeps the binary small (~5MB) vs Electron (~150MB)

---
UnobligatedRascal
