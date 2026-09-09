"""Ensure Sniper model is available under artifacts/models/ (.cbm preferred, .pkl legacy)."""

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST_DIR = os.path.join(ROOT, "artifacts", "models")
PKL_DST = os.path.join(DST_DIR, "trading_model_sniper_v5.pkl")
CBM_DST = os.path.join(DST_DIR, "trading_model_sniper_v5.cbm")
PKL_SRC = os.path.join(ROOT, "trading_model_sniper_v5.pkl")
CBM_SRC = os.path.join(ROOT, "trading_model_sniper_v5.cbm")

os.makedirs(DST_DIR, exist_ok=True)

if os.path.exists(CBM_DST):
    print(f"Native model already at {CBM_DST}")
elif os.path.exists(CBM_SRC):
    shutil.copy2(CBM_SRC, CBM_DST)
    print(f"Copied native .cbm to {CBM_DST}")
elif os.path.exists(PKL_DST):
    print(f"Legacy pickle model at {PKL_DST} (inference prefers .cbm when present)")
elif os.path.exists(PKL_SRC):
    shutil.copy2(PKL_SRC, PKL_DST)
    print(f"Copied legacy pickle to {PKL_DST}")
    print("Tip: train native .cbm: python train.py --strategy sniper --period 10y")
else:
    print("No model found.")
    print("Train production model: python train.py --strategy sniper --period 10y")
    print("Fast dev (no GDELT):     python train.py --strategy sniper --no-gdelt-backfill")
