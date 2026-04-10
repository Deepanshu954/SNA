#!/bin/bash
# ============================================================================
# launch.sh — Community Detection Showcase — One-click launcher
# ============================================================================
# Installs all dependencies, verifies the dataset, and launches the
# Streamlit web app.
#
# Usage:
#   chmod +x launch.sh
#   ./launch.sh
# ============================================================================

set -e  # Exit on any error

# ── Colors for output ──────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Helper functions ───────────────────────────────────────────────────────
info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[✓]${NC}     $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[✗]${NC}     $1"; }

# ── Navigate to project root ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   🔬 Community Detection & Information Diffusion Showcase  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check Python ──────────────────────────────────────────────────
info "Checking Python installation..."

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.9+ is required but not found."
    echo "  Please install Python 3.9 or later:"
    echo "    macOS:  brew install python@3.11"
    echo "    Linux:  sudo apt install python3.11"
    exit 1
fi

success "Found $($PYTHON --version 2>&1)"

# ── Step 2: Check/add user bin to PATH ────────────────────────────────────
USER_BIN="$($PYTHON -m site --user-base 2>/dev/null)/bin"
if [ -d "$USER_BIN" ] && [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
    export PATH="$USER_BIN:$PATH"
    info "Added $USER_BIN to PATH"
fi

# ── Step 3: Install dependencies ─────────────────────────────────────────
info "Installing Python dependencies..."

$PYTHON -m pip install --user --upgrade pip -q 2>/dev/null || true

if $PYTHON -m pip install --user -r requirements.txt -q 2>&1 | tail -5; then
    success "All dependencies installed"
else
    warn "Some packages may have had issues. Trying individual installs..."
    while IFS= read -r pkg || [ -n "$pkg" ]; do
        pkg=$(echo "$pkg" | tr -d '[:space:]')
        [ -z "$pkg" ] && continue
        [ "${pkg:0:1}" = "#" ] && continue
        $PYTHON -m pip install --user "$pkg" -q 2>/dev/null || warn "Could not install: $pkg"
    done < requirements.txt
    success "Dependency installation complete"
fi

# ── Step 4: Verify dataset ───────────────────────────────────────────────
info "Checking for dataset..."

DATA_FILE="data/facebook_combined.txt"
if [ -f "$DATA_FILE" ]; then
    LINES=$(wc -l < "$DATA_FILE" | tr -d ' ')
    success "Dataset found: $DATA_FILE ($LINES edges)"
else
    warn "Dataset not found at $DATA_FILE"
    info "Attempting to download from SNAP Stanford..."
    
    mkdir -p data
    if command -v curl &>/dev/null; then
        curl -sL "https://snap.stanford.edu/data/facebook_combined.txt.gz" -o data/facebook_combined.txt.gz 2>/dev/null
        if [ -f "data/facebook_combined.txt.gz" ]; then
            gunzip -f data/facebook_combined.txt.gz 2>/dev/null
            if [ -f "$DATA_FILE" ]; then
                success "Dataset downloaded successfully!"
            else
                error "Download failed. Please download manually:"
                echo "  https://snap.stanford.edu/data/ego-Facebook.html"
                echo "  Place facebook_combined.txt in the data/ folder"
                exit 1
            fi
        fi
    elif command -v wget &>/dev/null; then
        wget -q "https://snap.stanford.edu/data/facebook_combined.txt.gz" -O data/facebook_combined.txt.gz 2>/dev/null
        gunzip -f data/facebook_combined.txt.gz 2>/dev/null
        if [ -f "$DATA_FILE" ]; then
            success "Dataset downloaded successfully!"
        else
            error "Download failed"
            exit 1
        fi
    else
        error "Neither curl nor wget found. Please download the dataset manually."
        exit 1
    fi
fi

# ── Step 5: Verify imports ───────────────────────────────────────────────
info "Verifying Python imports..."

if $PYTHON -c "
import sys; sys.path.insert(0, '.')
import streamlit, networkx, community, pyvis, matplotlib, plotly, pandas, numpy, sklearn
from src.graph_utils import load_graph
from src.community_detection import detect_communities
from src.diffusion import simulate_ic
from src.dw_louvain import run_dw_louvain
from src.evaluate import compute_nmi
from src.visualize import plot_degree_dist
print('OK')
" 2>/dev/null | grep -q "OK"; then
    success "All imports verified"
else
    error "Import verification failed. Running diagnostic..."
    $PYTHON -c "
import sys; sys.path.insert(0, '.')
try:
    import streamlit; print('  streamlit:', streamlit.__version__)
except: print('  ✗ streamlit missing')
try:
    import networkx; print('  networkx:', networkx.__version__)
except: print('  ✗ networkx missing')
try:
    import community; print('  python-louvain: OK')
except: print('  ✗ python-louvain missing')
try:
    import pyvis; print('  pyvis: OK')
except: print('  ✗ pyvis missing')
try:
    import matplotlib; print('  matplotlib:', matplotlib.__version__)
except: print('  ✗ matplotlib missing')
try:
    import plotly; print('  plotly:', plotly.__version__)
except: print('  ✗ plotly missing')
try:
    import pandas; print('  pandas:', pandas.__version__)
except: print('  ✗ pandas missing')
try:
    import numpy; print('  numpy:', numpy.__version__)
except: print('  ✗ numpy missing')
try:
    import sklearn; print('  scikit-learn:', sklearn.__version__)
except: print('  ✗ scikit-learn missing')
"
    echo ""
    warn "Some dependencies are missing. Attempting to fix..."
    $PYTHON -m pip install --user streamlit networkx python-louvain pyvis matplotlib plotly pandas numpy scikit-learn 2>&1 | tail -5
fi

# ── Step 6: Find streamlit command ───────────────────────────────────────
STREAMLIT=""
if command -v streamlit &>/dev/null; then
    STREAMLIT="streamlit"
elif [ -x "$USER_BIN/streamlit" ]; then
    STREAMLIT="$USER_BIN/streamlit"
else
    STREAMLIT="$PYTHON -m streamlit"
fi

# ── Step 7: Launch! ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                  🚀 Launching Streamlit App                 ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Starting server on http://localhost:8501"
info "Press Ctrl+C to stop the server"
echo ""

$STREAMLIT run app/main.py \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    2>&1
