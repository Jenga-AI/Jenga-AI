#!/bin/bash
# MLflow Server Startup Script for Jenga-AI
# This script starts the MLflow tracking server with configuration from mlflow_config.yaml

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Starting MLflow Tracking Server${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Load configuration (using Python to parse YAML)
if [ -f "mlflow_config.yaml" ]; then
    echo -e "${GREEN}✓${NC} Loading configuration from mlflow_config.yaml"
    
    # Parse YAML using Python
    CONFIG=$(python3 -c "
import yaml
try:
    with open('mlflow_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(config['server']['host'], config['server']['port'], config['server']['backend_store_uri'])
except Exception as e:
    print('127.0.0.1 5000 ./mlruns')
")
    
    read HOST PORT BACKEND_STORE <<< "$CONFIG"
else
    echo -e "${YELLOW}⚠${NC} Configuration file not found, using defaults"
    HOST="127.0.0.1"
    PORT="5000"
    BACKEND_STORE="./mlruns"
fi

# Override with environment variables if set
HOST="${MLFLOW_HOST:-$HOST}"
PORT="${MLFLOW_PORT:-$PORT}"
BACKEND_STORE="${MLFLOW_BACKEND_STORE_URI:-$BACKEND_STORE}"

# Create mlruns directory if it doesn't exist
if [ "$BACKEND_STORE" = "./mlruns" ]; then
    mkdir -p "./mlruns"
    echo -e "${GREEN}✓${NC} MLruns directory ready: ./mlruns"
fi

echo -e "\n${BLUE}Server Configuration:${NC}"
echo -e "  Host: ${GREEN}$HOST${NC}"
echo -e "  Port: ${GREEN}$PORT${NC}"
echo -e "  Backend Store: ${GREEN}$BACKEND_STORE${NC}"

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "\n${RED}✗${NC} Port $PORT is already in use!"
    echo -e "${YELLOW}  Tip: Stop the existing MLflow server or use a different port${NC}"
    echo -e "${YELLOW}  To stop: kill \$(lsof -t -i:$PORT)${NC}\n"
    exit 1
fi

echo -e "\n${BLUE}Starting MLflow UI...${NC}\n"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} Activating virtual environment"
    source .venv/bin/activate
fi

# Start MLflow server
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  MLflow UI is starting...${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "Access the UI at: ${BLUE}http://$HOST:$PORT${NC}\n"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}\n"

# Start the server
mlflow server \
    --host "$HOST" \
    --port "$PORT" \
    --backend-store-uri "$BACKEND_STORE" \
    --default-artifact-root "$BACKEND_STORE"

# This line will only execute if the server stops
echo -e "\n${YELLOW}MLflow server stopped${NC}\n"
