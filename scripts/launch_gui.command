#!/bin/bash
# Double-click in Finder (macOS) to open the Streamlit GUI in your browser.
cd "$(dirname "$0")/.." || exit 1
for _venv in .venv venv; do
  if [[ -f "${_venv}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${_venv}/bin/activate"
    break
  fi
done
exec streamlit run gui/app.py
