# still-waiting GUI

Browser-based control panel for training operations. Served directly from the orchestrator at `http://<YOUR_TRAINING_NODE_IP>:9999/`.

Built with React + Vite + TailwindCSS.

## Architecture

```
┌─────────────────┐ HTTP+WS   ┌──────────────────┐        ┌─────────────────┐
│  Your Browser   │ ────────► │  orchestrator    │ ───────│  Python workers │
│  (:9999/)       │           │  (:9999/v1/api)  │        │  (CUDA GPUs)    │
└─────────────────┘           └──────────────────┘        └─────────────────┘
```

- GUI served as static files from `/` (SPA with fallback routing)
- API at `/v1/` — OpenAI-compatible training endpoints
- Single deployment: one orchestrator binary serves everything

## Development

### Prerequisites
- Node.js 20+
- npm

### Install & run locally
```bash
cd gui
npm install
npm run dev
```

Dev server proxies `/v1` requests to your training node (set `VITE_PROXY_TARGET=http://<ip>:9999`).

### Build for deployment
```bash
npm run build
# Output: gui/dist/ — deploy to your training node alongside the orchestrator
```

## Features

- **Dashboard**: Live status of all training jobs (auto-refresh every 5s)
- **Job creation**: Model selection, target steps, dataset path
- **Job detail**: Checkpoint history with loss metrics, pause/resume controls
- **Status banner**: Running/queued/failed job counts at a glance

## Deployment

Build output is served by orchestrator. Configure with:
```bash
ORCH_STATIC_DIR=<PATH_TO_PROJECT>/gui/dist ./agent-orchestrator
```

## Why browser-based (not Tauri)?

Training nodes are typically headless — the GUI always runs remotely. Browser gives:
- Zero install friction — open browser, go
- Accessible from any device on LAN (laptop, tablet, phone)
- Simpler deployment — no native builds, no per-platform packaging
- Same origin as API — no CORS configuration needed

---
Built by UnobligatedRascal — Ancient hardware, fresh ambition.
