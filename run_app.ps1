$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
}

python scripts/setup_model.py

Write-Host "Starting FastAPI on :8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"

Write-Host "Starting Gradio on :7860 ..."
python -m src.ui.gradio_app
