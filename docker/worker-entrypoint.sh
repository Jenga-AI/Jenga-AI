#!/bin/bash
"""
JengaHub Worker Entrypoint Script

This script handles different worker modes and initialization tasks.
Supports training workers, processing workers, and utility workers.
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Initialize worker environment
init_worker() {
    log "Initializing JengaHub Worker..."
    
    # Create worker status file
    mkdir -p /app/tmp
    cat > /app/tmp/worker_status.json << EOF
{
    "worker_type": "${WORKER_TYPE:-unknown}",
    "started_at": $(date +%s),
    "last_heartbeat": $(date +%s),
    "pid": $$,
    "version": "${JENGAHUB_VERSION:-1.0.0}"
}
EOF
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers
    
    # Initialize logging
    setup_logging
    
    # Validate environment
    validate_environment
}

# Setup signal handlers
setup_signal_handlers() {
    trap 'handle_shutdown TERM' TERM
    trap 'handle_shutdown INT' INT
    trap 'handle_shutdown QUIT' QUIT
}

handle_shutdown() {
    signal=$1
    log "Received $signal signal, initiating graceful shutdown..."
    
    # Update worker status
    if [ -f /app/tmp/worker_status.json ]; then
        python3 -c "
import json
import time
status_file = '/app/tmp/worker_status.json'
try:
    with open(status_file, 'r') as f:
        status = json.load(f)
    status['shutdown_initiated'] = int(time.time())
    status['shutdown_signal'] = '$signal'
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
except Exception as e:
    print(f'Failed to update status file: {e}')
"
    fi
    
    # Kill child processes gracefully
    if [ ! -z "$WORKER_PID" ]; then
        log "Stopping worker process (PID: $WORKER_PID)..."
        kill -TERM $WORKER_PID 2>/dev/null || true
        sleep 5
        kill -KILL $WORKER_PID 2>/dev/null || true
    fi
    
    log "Shutdown complete"
    exit 0
}

# Setup logging
setup_logging() {
    export JENGAHUB_LOG_LEVEL=${JENGAHUB_LOG_LEVEL:-INFO}
    
    # Create log directory
    mkdir -p ${JENGAHUB_LOG_DIR:-/app/logs}
    
    # Set log file based on worker type
    export JENGAHUB_LOG_FILE="${JENGAHUB_LOG_DIR:-/app/logs}/worker-${WORKER_TYPE:-unknown}-$(date +%Y%m%d).log"
    
    log "Logging to: $JENGAHUB_LOG_FILE"
}

# Validate environment
validate_environment() {
    log "Validating worker environment..."
    
    # Check required directories
    for dir in "${JENGAHUB_DATA_DIR:-/app/data}" "${JENGAHUB_MODEL_DIR:-/app/models}" "${JENGAHUB_LOG_DIR:-/app/logs}"; do
        if [ ! -d "$dir" ]; then
            error "Required directory not found: $dir"
            exit 1
        fi
        
        if [ ! -w "$dir" ]; then
            error "Directory not writable: $dir"
            exit 1
        fi
    done
    
    # Check Python environment
    if ! python -c "import jengahub" 2>/dev/null; then
        error "JengaHub package not found in Python path"
        exit 1
    fi
    
    # Check GPU if required
    if [ "${JENGAHUB_REQUIRE_GPU:-false}" = "true" ]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            error "GPU required but not available"
            exit 1
        fi
        info "GPU validation passed: $(python -c "import torch; print(torch.cuda.device_count())") device(s) available"
    fi
    
    log "Environment validation passed"
}

# Update heartbeat
update_heartbeat() {
    while true; do
        sleep 30
        if [ -f /app/tmp/worker_status.json ]; then
            python3 -c "
import json
import time
status_file = '/app/tmp/worker_status.json'
try:
    with open(status_file, 'r') as f:
        status = json.load(f)
    status['last_heartbeat'] = int(time.time())
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
except Exception:
    pass
"
        fi
    done &
    HEARTBEAT_PID=$!
}

# Training worker mode
run_training_worker() {
    log "Starting training worker..."
    export WORKER_TYPE="training"
    
    # Setup distributed training if configured
    if [ ! -z "$WORLD_SIZE" ] && [ ! -z "$RANK" ]; then
        info "Distributed training mode: rank $RANK of $WORLD_SIZE"
        export MASTER_ADDR=${MASTER_ADDR:-localhost}
        export MASTER_PORT=${MASTER_PORT:-29500}
        info "Master: $MASTER_ADDR:$MASTER_PORT"
    fi
    
    # Start training worker
    python -m jengahub.training.worker \
        --worker-type training \
        --log-level ${JENGAHUB_LOG_LEVEL:-INFO} \
        --config ${JENGAHUB_CONFIG_FILE:-/app/configs/training.yaml} &
    
    WORKER_PID=$!
    wait $WORKER_PID
}

# Processing worker mode
run_processing_worker() {
    log "Starting processing worker..."
    export WORKER_TYPE="processing"
    
    # Start processing worker
    python -m jengahub.processing.worker \
        --worker-type processing \
        --log-level ${JENGAHUB_LOG_LEVEL:-INFO} \
        --config ${JENGAHUB_CONFIG_FILE:-/app/configs/processing.yaml} &
    
    WORKER_PID=$!
    wait $WORKER_PID
}

# Data preparation worker mode
run_data_worker() {
    log "Starting data preparation worker..."
    export WORKER_TYPE="data"
    
    # Start data worker
    python -m jengahub.data.worker \
        --worker-type data \
        --log-level ${JENGAHUB_LOG_LEVEL:-INFO} \
        --config ${JENGAHUB_CONFIG_FILE:-/app/configs/data.yaml} &
    
    WORKER_PID=$!
    wait $WORKER_PID
}

# Notebook server mode
run_notebook_server() {
    log "Starting Jupyter notebook server..."
    export WORKER_TYPE="notebook"
    
    # Configure Jupyter
    export JUPYTER_TOKEN=${JUPYTER_TOKEN:-jengahub}
    export JUPYTER_PORT=${JUPYTER_PORT:-8888}
    
    # Start Jupyter Lab
    jupyter lab \
        --ip=0.0.0.0 \
        --port=$JUPYTER_PORT \
        --no-browser \
        --allow-root \
        --token=$JUPYTER_TOKEN \
        --notebook-dir=/app/notebooks &
    
    WORKER_PID=$!
    wait $WORKER_PID
}

# Utility worker mode (for maintenance tasks)
run_utility_worker() {
    log "Starting utility worker..."
    export WORKER_TYPE="utility"
    
    # Run utility command or keep container alive
    if [ ! -z "$UTILITY_COMMAND" ]; then
        log "Executing utility command: $UTILITY_COMMAND"
        eval "$UTILITY_COMMAND"
    else
        log "Utility worker ready for commands. Use 'docker exec' to run tasks."
        # Keep container alive
        tail -f /dev/null &
        WORKER_PID=$!
        wait $WORKER_PID
    fi
}

# Ray worker mode (for distributed computing)
run_ray_worker() {
    log "Starting Ray worker..."
    export WORKER_TYPE="ray"
    
    # Ray head or worker node
    if [ "${RAY_NODE_TYPE:-worker}" = "head" ]; then
        info "Starting Ray head node..."
        ray start --head \
            --port=${RAY_PORT:-10001} \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=${RAY_DASHBOARD_PORT:-8265} &
    else
        info "Starting Ray worker node..."
        ray start --address=${RAY_HEAD_ADDRESS:-ray-head:10001} &
    fi
    
    WORKER_PID=$!
    wait $WORKER_PID
}

# Main entrypoint logic
main() {
    log "JengaHub Worker Container Starting..."
    
    # Initialize worker environment
    init_worker
    
    # Start heartbeat updater
    update_heartbeat
    
    # Determine worker mode
    WORKER_MODE=${1:-${WORKER_MODE:-training-worker}}
    
    case "$WORKER_MODE" in
        "training-worker"|"training")
            run_training_worker
            ;;
        "processing-worker"|"processing")
            run_processing_worker
            ;;
        "data-worker"|"data")
            run_data_worker
            ;;
        "notebook-server"|"notebook"|"jupyter")
            run_notebook_server
            ;;
        "utility-worker"|"utility"|"bash")
            run_utility_worker
            ;;
        "ray-worker"|"ray")
            run_ray_worker
            ;;
        *)
            error "Unknown worker mode: $WORKER_MODE"
            error "Valid modes: training-worker, processing-worker, data-worker, notebook-server, utility-worker, ray-worker"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"