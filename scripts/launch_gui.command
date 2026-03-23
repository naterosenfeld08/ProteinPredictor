#!/bin/bash
# Double-click in Finder (macOS) to open the Streamlit GUI in your browser.
cd "$(dirname "$0")/.." || exit 1
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "venv/bin/activate"
fi
exec streamlit run gui/app.py
