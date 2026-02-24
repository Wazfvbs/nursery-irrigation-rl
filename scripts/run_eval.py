from __future__ import annotations

import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO

from irrigation_rl.train.ppo_train import load_yaml, build_env
from irrigation_rl.train.evaluate import evaluate_policy


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="configs/train.yaml", help="Path to train.yaml")
    ap.add_argument("--model", type=str, default="", help="Path to trained model zip (optional)")
    ap.add_argument("--out", type=str, default="outputs/eval_run", help="Output directory")

    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy for evaluation")
    ap.add_argument("--max_steps", type=int, default=0, help="Optional cap steps (0 means no cap)")

    args = ap.parse_args()

    # -------- Load configs --------
    train_cfg = load_yaml(args.config)
    env_cfg = load_yaml(train_cfg["paths"]["env_config"])

    seed = int(train_cfg.get("seed", 0))

    # -------- Build env (统一入口：RewardWrapper 在这里注入) --------
    env = build_env(env_cfg, train_cfg, seed=seed)

    # -------- Resolve model path --------
    model_path = args.model.strip()
    if not model_path:
        model_path = os.path.join(train_cfg["paths"]["out_dir"], f"ppo_seed{seed}.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")

    # -------- Load model --------
    model = PPO.load(model_path)

    # -------- Evaluate --------
    max_steps = None if args.max_steps <= 0 else int(args.max_steps)

    res = evaluate_policy(
        env=env,
        model=model,
        out_dir=args.out,
        deterministic=bool(args.deterministic),
        max_steps=max_steps,
    )

    print(f"[OK] Eval finished.")
    print(f"  model: {model_path}")
    print(f"  out_dir: {res['out_dir']}")
    print(f"  trajectory: {res['trajectory_csv']}")
    print(f"  steps: {res['steps']}")


if __name__ == "__main__":
    main()
