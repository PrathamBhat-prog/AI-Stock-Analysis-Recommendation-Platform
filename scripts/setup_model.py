"""Copy root trading_model_sniper_v5.pkl into artifacts/models/ if present."""

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "trading_model_sniper_v5.pkl")
DST_DIR = os.path.join(ROOT, "artifacts", "models")
DST = os.path.join(DST_DIR, "trading_model_sniper_v5.pkl")

if os.path.exists(SRC):
    os.makedirs(DST_DIR, exist_ok=True)
    shutil.copy2(SRC, DST)
    print(f"Copied model to {DST}")
elif os.path.exists(DST):
    print(f"Model already at {DST}")
else:
    print("No model found. Train with: python train.py --strategy sniper")
