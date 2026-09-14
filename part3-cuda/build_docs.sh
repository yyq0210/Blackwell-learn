#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
DOC_PYTHON="$ROOT/.venvs/modern-gpu-docs312/bin/python"
"$DOC_PYTHON" scripts/v01_api_atlas.py
"$DOC_PYTHON" scripts/v01_walkthrough.py
"$DOC_PYTHON" scripts/docs.py
"$DOC_PYTHON" scripts/testing_docs.py
"$DOC_PYTHON" scripts/reference_docs.py
"$DOC_PYTHON" scripts/report.py
"$DOC_PYTHON" scripts/check_docs.py
