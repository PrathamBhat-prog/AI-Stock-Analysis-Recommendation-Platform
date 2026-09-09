#!/bin/sh
set -e
python scripts/setup_model.py
uvicorn src.main:app --host 0.0.0.0 --port 8000 &
python -m src.ui.gradio_app
