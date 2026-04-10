#!/bin/bash
# ============================================================
# launch.sh — Community Detection Showcase Launcher
# Runs precompute (if needed), then starts Streamlit app
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Community Detection & Information Diffusion Showcase"
echo "  Bennett University 2024-25"
echo "============================================================"
echo ""

# ── Step 1: Check Python dependencies ──────────────────────────────────
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "  Installing dependencies..."
    pip install -r requirements.txt
else
    echo "  All dependencies OK ✅"
fi

# ── Step 2: Check dataset ──────────────────────────────────────────────
if [ ! -f "data/facebook_combined.txt" ]; then
    echo ""
    echo "📥 Downloading Facebook Social Circles dataset..."
    curl -sL "https://snap.stanford.edu/data/facebook_combined.txt.gz" \
        -o "data/facebook_combined.txt.gz"
    gunzip -f "data/facebook_combined.txt.gz"
    echo "  Dataset downloaded ✅"
fi

echo "  Dataset: $(wc -l < data/facebook_combined.txt) edges"

# ── Step 3: Precompute (if not already done) ───────────────────────────
if [ ! -f "data/precomputed/summary.json" ]; then
    echo ""
    echo "🔬 Running precomputation (this takes 2-5 minutes, one time only)..."
    echo ""
    python3 precompute.py
    echo ""
    echo "  Precomputation complete ✅"
else
    echo "  Precomputed data found ✅"
fi

# ── Step 4: Launch Streamlit ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🚀 Launching Streamlit App..."
echo "  Open: http://localhost:8501"
echo "============================================================"
echo ""

python3 -m streamlit run app/main.py --server.headless=true --server.port=8501
