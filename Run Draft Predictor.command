#!/bin/bash
# Double-click to launch the app in your browser.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/streamlit ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
fi
.venv/bin/streamlit run draft_predictor_app.py --server.headless false
