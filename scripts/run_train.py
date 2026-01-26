from __future__ import annotations
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
from irrigation_rl.train.ppo_train import train_ppo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    args = ap.parse_args()

    model_path = train_ppo(args.config)
    print(f"[OK] Model saved to: {model_path}")

if __name__ == "__main__":
    main()
