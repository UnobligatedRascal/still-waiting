#!/bin/bash
# still-waiting startup script
# Run as: sudo ./start_still_waiting.sh (requires root for daemon mode)
# Or: ./start_still_waiting.sh user (runs in foreground, no root)
#
# Built by UnobligatedRascal — Ancient hardware, fresh ambition.

set -e

# Resolve project root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ORCH_BINARY="$PROJECT_DIR/orchestrator/target/release/agent-orchestrator"
ORCH_STATIC="$PROJECT_DIR/gui/dist"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="/tmp/still-waiting-orch.pid"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[still-waiting]${NC} $1"; }
warn() { echo -e "${YELLOW}[still-waiting]${NC} $1"; }
error() { echo -e "${RED}[still-waiting]${NC} $1"; }

usage() {
    echo "Usage: $0 [start|stop|restart|status|user]"
    echo ""
    echo "Commands:"
    echo "  start            Start orchestrator in background (requires root)"
    echo "  stop             Stop orchestrator"
    echo "  restart          Restart orchestrator"
    echo "  status           Check orchestrator status"
    echo "  user             Run in foreground (for testing, no root needed)"
    echo ""
    echo "Or just run: sudo $0  (defaults to start)"
    exit 1
}

check_prerequisites() {
    local missing=0

    if [ ! -f "$ORCH_BINARY" ]; then
        error "Orchestrator binary not found: $ORCH_BINARY"
        error "Run: cd $PROJECT_DIR/orchestrator && . ~/.cargo/env && cargo build --release"
        missing=1
    fi

    if [ ! -d "$ORCH_STATIC" ]; then
        error "GUI dist not found: $ORCH_STATIC"
        error "Run: cd $PROJECT_DIR/gui && npm install && npm run build"
        missing=1
    fi

    if ! command -v rustup &> /dev/null; then
        warn "Rust not found in PATH. Run: . ~/.cargo/env"
    fi

    if ! command -v python3 &> /dev/null; then
        error "Python3 not found"
        missing=1
    fi

    if [ $missing -eq 1 ]; then
        exit 1
    fi
}

start_background() {
    info "Starting orchestrator..."

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Kill existing instance
    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            warn "Killing existing instance (PID $old_pid)"
            kill "$old_pid"
            sleep 1
        fi
    fi

    # Start with nohup
    cd "$PROJECT_DIR/orchestrator"
    ORCH_STATIC_DIR="$ORCH_STATIC" nohup ./target/release/agent-orchestrator \
        > "$LOG_DIR/orchestrator.log" 2>&1 &
    local new_pid=$!
    echo $new_pid > "$PID_FILE"

    # Wait and verify
    sleep 2
    if kill -0 "$new_pid" 2>/dev/null; then
        info "Orchestrator started (PID $new_pid)"
        local ip=$(hostname -I | awk '{print $1}')
        info "GUI: http://${ip}:9999/"
        info "API: http://${ip}:9999/v1/"
        info "Logs: $LOG_DIR/orchestrator.log"
    else
        error "Orchestrator failed to start. Check logs: $LOG_DIR/orchestrator.log"
        exit 1
    fi
}

stop_background() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping orchestrator (PID $pid)..."
            kill "$pid"
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                warn "Force killing..."
                kill -9 "$pid"
            fi
            info "Stopped"
            rm -f "$PID_FILE"
        else
            info "Not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        info "Not running (no PID file)"
    fi
}

status_check() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}Running${NC} (PID $pid)"
            curl -s http://127.0.0.1:9999/v1/training/jobs 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Jobs: {len(d.get(\"data\",[]))}')" 2>/dev/null || echo "API: unreachable"
        else
            echo -e "${RED}Not running${NC} (stale PID file)"
        fi
    else
        echo -e "${RED}Not running${NC}"
    fi
}

case "${1:-}" in
    start)
        if [ "$(id -u)" -ne 0 ]; then
            error "Requires root. Run: sudo $0 start"
            exit 1
        fi
        check_prerequisites
        start_background
        ;;
    stop)
        stop_background
        ;;
    restart)
        stop_background
        sleep 1
        if [ "$(id -u)" -ne 0 ]; then
            error "Requires root. Run: sudo $0 restart"
            exit 1
        fi
        check_prerequisites
        start_background
        ;;
    status)
        status_check
        ;;
    user)
        check_prerequisites
        info "Running in foreground (Ctrl+C to stop)..."
        cd "$PROJECT_DIR/orchestrator"
        exec ORCH_STATIC_DIR="$ORCH_STATIC" ./target/release/agent-orchestrator
        ;;
    *)
        usage
        ;;
esac
