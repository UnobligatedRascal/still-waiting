# still-waiting GUI

Browser-based control panel for NOUGHT training operations. Served directly from the orchestrator at `http://NOUGHT:9999/`.

Built with React + Vite + TailwindCSS.

## Architecture

```
┌─────────────────┐ HTTP+WS   ┌──────────────────┐        ┌─────────────────┐
│  Your Browser   │ ────────► │  orchestrator    │ ───────│  Python workers │
│  (:9999/)       │           │  (:9999/v1/api)  │        │  (Kepler GPUs)  │
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

Dev server proxies `/v1` requests to NOUGHT at `http://192.168.137.29:9999`.

### Build for deployment
```bash
npm run build
# Output: gui/dist/ — copy to NOUGHT /home/whistler/still-waiting/gui/dist/
```

## Features

- **Dashboard**: Live status of all training jobs (auto-refresh every 5s)
- **Job creation**: Model selection, target steps, dataset path
- **Job detail**: Checkpoint history with loss metrics, pause/resume controls
- **Status banner**: Running/queued/failed job counts at a glance

## Deployment

Build output is served by orchestrator. Configure with:
```bash
ORCH_STATIC_DIR=/home/whistler/still-waiting/gui/dist ./agent-orchestrator
```

## Why browser-based (not Tauri)?

NOUGHT is headless on LAN — the GUI always runs remotely anyway. Browser gives:
- Zero install friction — open browser, go
- Accessible from any device on LAN (laptop, tablet, phone)
- Simpler deployment — no native builds, no per-platform packaging
- Same origin as API — no CORS configuration needed

---
Built by UnobligatedRascal — Ancient hardware, fresh ambition.
