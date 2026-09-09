Write-Host "=== Stock Analyser — Local Test ===" -ForegroundColor Cyan

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
}

pip install -r requirements.txt --quiet
pip install pytest httpx --quiet

python scripts/setup_model.py
python verify_pipeline.py
if ($LASTEXITCODE -ne 0) { exit 1 }

pytest tests/ -v
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "`nOptional live test: python verify_pipeline.py --live --ticker AAPL" -ForegroundColor Green
Write-Host "Train Sniper v5: python train.py --strategy sniper --period 5y" -ForegroundColor Green
Write-Host "Backtest: python train.py --strategy backtest --period 5y" -ForegroundColor Green
