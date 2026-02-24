from __future__ import annotations

import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import csv
import copy
import inspect
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

from irrigation_rl.metrics.metrics import compute_metrics_from_csv, compute_metrics_from_df
from irrigation_rl.train.ppo_train import load_yaml, build_env

from irrigation_rl.baselines.threshold import ThresholdPolicy
from irrigation_rl.baselines.fao_rule import FAORulePolicy
from irrigation_rl.baselines.calendar import CalendarPolicy
from irrigation_rl.robust.obs_noise_wrapper import ObsNoiseWrapper, ObsNoiseConfig


# -----------------------------
# Tuned FAO-rule (grid-search)
# -----------------------------
@dataclass
class TunedFAORulePolicy:
    """
    Tuned FAO-rule baseline (paper "Tuned FAO-rule").

    Control law:
      if Dr_t > RAW(t) - δ:
           D_refill(t) = max(0, RAW(t) - Δ)
           I_t = clip(Dr_t - D_refill(t), 0, I_max)
      else:
           I_t = 0

    - Dr_t uses *observed* depletion (obs[0]) for fairness under observation noise.
    - RAW is read from info["RAW_mm"] (usually constant per episode in this env).
    """
    delta_mm: float
    Delta_mm: float
    a_max_mm: float

    def act(self, Dr: float, day: int | None = None, info: dict | None = None) -> float:
        RAW = float(info.get("RAW_mm", 0.0)) if isinstance(info, dict) else 0.0
        if RAW <= 0.0:
            return 0.0
        if Dr <= RAW - self.delta_mm:
            return 0.0
        D_refill = max(0.0, RAW - self.Delta_mm)
        I = Dr - D_refill
        return float(np.clip(I, 0.0, self.a_max_mm))


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_env_compat(env_cfg: dict, train_cfg: dict, seed: int):
    """Compat with possible older build_env signature."""
    sig = inspect.signature(build_env)
    if len(sig.parameters) >= 3:
        return build_env(env_cfg, train_cfg, seed=seed)
    return build_env(env_cfg, seed=seed)


def apply_robust_wrappers(env, *, setting: str, noise_cfg: dict, seed: int):
    """
    Apply robustness wrappers consistently.

    setting:
      - nominal: no wrapper
      - noise_only: ObsNoiseWrapper only
      - et0_only: ET0 multiplier only (dynamics-side)
      - noise_et0: both obs noise + ET0 multiplier

    NOTE:
      ET0 multiplier wrapper must exist at irrigation_rl/robust/et0_mult_wrapper.py
      (if you already applied robust_et0_patch, you're good).
    """
    setting = setting.lower().strip()

    if setting in ("nominal", "none", ""):
        return env

    # noise-only part
    if setting in ("noise_only", "noise", "obs_noise", "noise-only", "noiseonly") or setting.startswith("noise"):
        ocfg = ObsNoiseConfig(
            enabled=True,
            Dr_sigma_mm=float(noise_cfg.get("Dr_sigma_mm", 0.0)),
            theta_sigma=float(noise_cfg.get("theta_sigma", 0.0)),
            ET0_sigma=float(noise_cfg.get("ET0_sigma", 0.0)),
        )
        env = ObsNoiseWrapper(env, cfg=ocfg, seed=seed)

    # ET0 multiplier part
    if setting in ("et0_only", "et0", "et0-only", "et0only", "noise_et0", "noise+et0") or ("et0" in setting):
        try:
            from irrigation_rl.robust.et0_mult_wrapper import ET0MultWrapper, ET0MultConfig
        except Exception as e:
            raise RuntimeError(
                "You requested an ET0 robustness setting but ET0MultWrapper is missing.\n"
                "Next step: add file irrigation_rl/robust/et0_mult_wrapper.py (I can provide it).\n"
                f"Import error: {e}"
            )

        et0_block = noise_cfg.get("et0_mult", {}) if isinstance(noise_cfg, dict) else {}
        cfg = ET0MultConfig(
            enabled=True,
            per_day=bool(et0_block.get("per_day", True)),
            low=float(et0_block.get("low", 0.9)),
            high=float(et0_block.get("high", 1.1)),
        )
        env = ET0MultWrapper(env, cfg=cfg, seed=seed)

    return env


def _policy_act(policy, *, day: int, Dr_obs: float, info: Dict[str, Any]) -> float:
    """
    Call policy.act with best-effort signature matching:
      act(Dr), act(day), act(day, Dr), act(day, Dr, info), act(Dr, info)
    """
    sig = inspect.signature(policy.act)
    params = sig.parameters

    kw = {}
    if "day" in params:
        kw["day"] = day
    if "Dr" in params:
        kw["Dr"] = Dr_obs
    if "info" in params:
        kw["info"] = info
    return float(policy.act(**kw))


def rollout_policy(
        env,
        policy_name: str,
        policy,
        out_dir: str,
        *,
        seed: int,
        scenario: str = "nominal",
        save_csv: bool = True,
) -> Dict[str, Any]:
    """
    Baseline rollout on (possibly wrapped) env.

    - Uses Dr_obs = obs[0] (important for fairness under observation noise)
    - Saves trajectory.csv with ref-interval columns and clip columns
    - Computes metrics.json from trajectory.csv (ref-based metrics + clip_rate)
    """
    ensure_dir(out_dir)

    obs, info = env.reset(seed=seed)
    done = False
    t = 0

    traj_rows: List[Dict[str, Any]] = []

    while not done:
        day = int(info.get("day", t))
        Dr_obs = float(obs[0])  # policy sees observation (possibly noisy)

        I_cmd = _policy_act(policy, day=day, Dr_obs=Dr_obs, info=info)

        # Do NOT clip here; env will clip and report I_raw/clipped.
        obs_next, reward, terminated, truncated, info_next = env.step(np.array([I_cmd], dtype=np.float32))
        done = bool(terminated or truncated)

        row: Dict[str, Any] = {}
        row["t"] = t
        row["day"] = int(info_next.get("day", day))
        row["stage_norm"] = float(info_next.get("stage_norm", 0.0))

        # true state
        row["Dr"] = float(info_next.get("Dr_mm", 0.0))
        row["theta"] = float(info_next.get("theta", 0.0))

        # obs (debug)
        row["Dr_obs"] = float(obs_next[0]) if len(obs_next) > 0 else float("nan")
        row["theta_obs"] = float(obs_next[1]) if len(obs_next) > 1 else float("nan")
        row["ET0_obs"] = float(obs_next[2]) if len(obs_next) > 2 else float("nan")

        # env constants / weather
        row["TAW"] = float(info_next.get("TAW_mm", 0.0))
        row["RAW"] = float(info_next.get("RAW_mm", 0.0))
        row["ET0"] = float(info_next.get("ET0", 0.0))
        row["ETc"] = float(info_next.get("ETc", 0.0))

        # intervals (train/debug + ref/paper)
        lo_train = float(info_next.get("Dr_lo_train", info_next.get("Dr_lo", 0.0)))
        hi_train = float(info_next.get("Dr_hi_train", info_next.get("Dr_hi", 0.0)))
        lo_ref = float(info_next.get("Dr_lo_ref", lo_train))
        hi_ref = float(info_next.get("Dr_hi_ref", hi_train))
        mid_ref = float(info_next.get("Dr_mid_ref", 0.5 * (lo_ref + hi_ref)))

        row["Dr_lo"] = float(info_next.get("Dr_lo", lo_train))
        row["Dr_hi"] = float(info_next.get("Dr_hi", hi_train))
        row["Dr_lo_train"] = lo_train
        row["Dr_hi_train"] = hi_train
        row["Dr_lo_ref"] = lo_ref
        row["Dr_hi_ref"] = hi_ref
        row["Dr_mid_ref"] = mid_ref

        # action logs (clipped by env)
        row["I"] = float(info_next.get("I_mm", I_cmd))
        row["I_raw"] = float(info_next.get("I_raw_mm", I_cmd))
        row["clipped"] = int(info_next.get("clipped", 0))

        row["reward"] = float(reward)
        row["terminated"] = int(bool(terminated))
        row["truncated"] = int(bool(truncated))

        # aux
        row["in_ref"] = int((row["Dr"] >= lo_ref) and (row["Dr"] <= hi_ref))
        row["e_mid_ref"] = float(abs(row["Dr"] - mid_ref))

        traj_rows.append(row)

        obs = obs_next
        info = info_next
        t += 1

    traj_csv = os.path.join(out_dir, "trajectory.csv")
    if save_csv:
        base_cols = [
            "t", "day", "stage_norm",
            "Dr", "theta",
            "Dr_obs", "theta_obs", "ET0_obs",
            "TAW", "RAW",
            "ET0", "ETc",
            "Dr_lo", "Dr_hi",
            "Dr_lo_train", "Dr_hi_train",
            "Dr_lo_ref", "Dr_hi_ref", "Dr_mid_ref",
            "in_ref", "e_mid_ref",
            "I", "I_raw", "clipped",
            "reward", "terminated", "truncated",
        ]
        all_keys = set()
        for r in traj_rows:
            all_keys.update(r.keys())
        extra_cols = sorted([k for k in all_keys if k not in base_cols])
        header = base_cols + extra_cols

        with open(traj_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in traj_rows:
                w.writerow({k: r.get(k, "") for k in header})

    metrics = compute_metrics_from_csv(traj_csv)
    metrics.update({
        "method": policy_name,
        "scenario": scenario,
        "seed": int(seed),
        "trajectory_csv": traj_csv,
        "out_dir": out_dir,
    })
    save_json(os.path.join(out_dir, "metrics.json"), metrics)
    return metrics


def tune_tuned_rule(
        env_cfg: dict,
        eval_train_cfg: dict,
        *,
        dev_seed_start: int,
        dev_num_seeds: int,
        out_dir: str,
        setting: str,
        noise_cfg: dict,
        grid_delta: List[float],
        grid_Delta: List[float],
) -> Dict[str, Any]:
    """
    Grid search (δ,Δ) on dev seeds to maximize mean TIR_ref,
    tie-breaker: lower mean TotalIrrigation_mm.

    Writes:
      - grid_results.csv
      - best_params.json
    """
    ensure_dir(out_dir)

    rows = []
    for delta in grid_delta:
        for Delta in grid_Delta:
            tir_list = []
            irr_list = []
            for i in range(dev_num_seeds):
                seed = dev_seed_start + i
                env = build_env_compat(env_cfg, eval_train_cfg, seed=seed)
                env = apply_robust_wrappers(env, setting=setting, noise_cfg=noise_cfg, seed=seed)

                a_max = float(np.asarray(env.action_space.high).reshape(-1)[0])
                pol = TunedFAORulePolicy(delta_mm=float(delta), Delta_mm=float(Delta), a_max_mm=a_max)

                obs, info = env.reset(seed=seed)
                done = False
                traj_rows: List[Dict[str, Any]] = []
                t = 0
                while not done:
                    day = int(info.get("day", t))
                    Dr_obs = float(obs[0])
                    I_cmd = _policy_act(pol, day=day, Dr_obs=Dr_obs, info=info)
                    obs_next, reward, terminated, truncated, info_next = env.step(np.array([I_cmd], dtype=np.float32))
                    done = bool(terminated or truncated)
                    traj_rows.append({
                        "Dr": float(info_next.get("Dr_mm", 0.0)),
                        "I": float(info_next.get("I_mm", I_cmd)),
                        "I_raw": float(info_next.get("I_raw_mm", I_cmd)),
                        "clipped": int(info_next.get("clipped", 0)),
                        "Dr_lo_ref": float(info_next.get("Dr_lo_ref", 0.0)),
                        "Dr_hi_ref": float(info_next.get("Dr_hi_ref", 0.0)),
                        "Dr_mid_ref": float(info_next.get("Dr_mid_ref", 0.0)),
                    })
                    obs, info = obs_next, info_next
                    t += 1

                import pandas as pd
                df = pd.DataFrame(traj_rows)
                m = compute_metrics_from_df(df)
                tir_list.append(float(m.get("TIR_ref", np.nan)))
                irr_list.append(float(m.get("TotalIrrigation_mm", np.nan)))

            rows.append({
                "delta_mm": float(delta),
                "Delta_mm": float(Delta),
                "mean_TIR_ref": float(np.nanmean(tir_list)),
                "mean_TotalIrrigation_mm": float(np.nanmean(irr_list)),
            })

    rows_sorted = sorted(rows, key=lambda r: (-r["mean_TIR_ref"], r["mean_TotalIrrigation_mm"]))
    best = rows_sorted[0] if rows_sorted else {"delta_mm": 0.0, "Delta_mm": 0.0}

    import pandas as pd
    df_all = pd.DataFrame(rows_sorted)
    df_all.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False, encoding="utf-8-sig")

    best_params = {"delta_mm": float(best["delta_mm"]), "Delta_mm": float(best["Delta_mm"])}
    save_json(os.path.join(out_dir, "best_params.json"), best_params)
    return best_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml", help="train.yaml")

    ap.add_argument("--seed_start", type=int, default=None, help="report seed start (e.g., 42)")
    ap.add_argument("--num_seeds", type=int, default=10, help="number of report seeds")

    ap.add_argument("--out", type=str, default="outputs/baselines_ref", help="baseline output dir")
    ap.add_argument("--setting", type=str, default="nominal", help="nominal | noise_only | et0_only | noise_et0")

    ap.add_argument("--noise_config", type=str, default="configs/noise_test.yaml")

    # tuned rule params for report
    ap.add_argument("--tuned_params", type=str, default="", help="path to best_params.json (delta_mm, Delta_mm)")
    ap.add_argument("--delta_mm", type=float, default=5.0)
    ap.add_argument("--Delta_mm", type=float, default=10.0)

    # dev grid-search
    ap.add_argument("--tune", action="store_true", help="grid search tuned rule on dev seeds")
    ap.add_argument("--dev_seed_start", type=int, default=32)
    ap.add_argument("--dev_num_seeds", type=int, default=10)

    args = ap.parse_args()

    train_cfg = load_yaml(args.config)
    env_cfg = load_yaml(train_cfg["paths"]["env_config"])

    base_seed = int(train_cfg.get("seed", 42))
    seed_start = int(args.seed_start) if args.seed_start is not None else base_seed

    # evaluation protocol: disable UCB bonus contribution
    eval_train_cfg = copy.deepcopy(train_cfg)
    if "ablation" in eval_train_cfg:
        eval_train_cfg["ablation"]["use_ucb_bonus"] = False
    if "reward" in eval_train_cfg:
        eval_train_cfg["reward"]["w_ucb"] = 0.0

    # -------- load noise_config and normalize --------
    noise_cfg: Dict[str, Any] = {}
    if os.path.exists(args.noise_config):
        noise_cfg = load_yaml(args.noise_config) or {}

    # configs/noise_test.yaml schema:
    #   sensor_noise.theta_sigma
    #   weather_bias.ET0_mult_min/max
    # We also accept a flat format (theta_sigma, Dr_sigma_mm, ...)
    sensor_noise = noise_cfg.get("sensor_noise", {}) if isinstance(noise_cfg, dict) else {}
    weather_bias = noise_cfg.get("weather_bias", {}) if isinstance(noise_cfg, dict) else {}

    Dr_sigma_mm = float(sensor_noise.get("Dr_sigma_mm", noise_cfg.get("Dr_sigma_mm", noise_cfg.get("Dr_sigma", 0.0))))
    theta_sigma = float(sensor_noise.get("theta_sigma", noise_cfg.get("theta_sigma", 0.0)))
    ET0_sigma = float(sensor_noise.get("ET0_sigma", noise_cfg.get("ET0_sigma", 0.0)))

    et0_low = float(weather_bias.get("ET0_mult_min", noise_cfg.get("ET0_mult_min", 0.9)))
    et0_high = float(weather_bias.get("ET0_mult_max", noise_cfg.get("ET0_mult_max", 1.1)))

    noise_cfg_norm = {
        "Dr_sigma_mm": Dr_sigma_mm,
        "theta_sigma": theta_sigma,
        "ET0_sigma": ET0_sigma,
        "et0_mult": {"per_day": True, "low": et0_low, "high": et0_high},
    }

    ensure_dir(args.out)

    # -------- optional tuning on dev seeds --------
    tuned_best = None
    if args.tune:
        grid_delta = [0, 2, 5, 10, 15]
        grid_Delta = [0, 5, 10, 15, 20]
        tune_out = os.path.join(args.out, "TunedFAORule", "tuning_dev")
        tuned_best = tune_tuned_rule(
            env_cfg,
            eval_train_cfg,
            dev_seed_start=int(args.dev_seed_start),
            dev_num_seeds=int(args.dev_num_seeds),
            out_dir=tune_out,
            setting="nominal",
            noise_cfg=noise_cfg_norm,
            grid_delta=[float(x) for x in grid_delta],
            grid_Delta=[float(x) for x in grid_Delta],
        )
        print(f"[TUNE] Best tuned params: {tuned_best} (saved under {tune_out})")

    # -------- report tuned params --------
    if args.tuned_params:
        with open(args.tuned_params, "r", encoding="utf-8") as f:
            tuned_best = json.load(f)
    if tuned_best is None:
        tuned_best = {"delta_mm": float(args.delta_mm), "Delta_mm": float(args.Delta_mm)}

    all_metrics: List[Dict[str, Any]] = []

    # -------- report seeds rollouts --------
    for i in range(int(args.num_seeds)):
        seed = seed_start + i

        # get constants via reset-info (robust against wrappers)
        env0 = build_env_compat(env_cfg, eval_train_cfg, seed=seed)
        a_max = float(np.asarray(env0.action_space.high).reshape(-1)[0])
        _, info0 = env0.reset(seed=seed)
        RAW = float(info0.get("RAW_mm", 0.0))

        # Threshold: Dr > 0.9*RAW => irrigate a_max
        Dr_hi = 0.9 * RAW if RAW > 0 else 1.0
        thr = ThresholdPolicy(Dr_threshold=Dr_hi, irrigation_mm=a_max)

        # FAO rule: Dr > RAW => irrigate a_max
        fao = FAORulePolicy(RAW=RAW, irrigation_mm=a_max)

        # Calendar: every k days irrigate I_fix
        cal_cfg = train_cfg.get("baselines", {}).get("calendar", {}) if isinstance(train_cfg, dict) else {}
        k_days = int(cal_cfg.get("interval_days", 3))
        I_fix = float(cal_cfg.get("irrigation_mm", 3.3))
        offset = int(cal_cfg.get("offset_days", 0))
        cal = CalendarPolicy(interval_days=k_days, irrigation_mm=I_fix, offset_days=offset, a_max_mm=a_max)

        tuned = TunedFAORulePolicy(
            delta_mm=float(tuned_best.get("delta_mm", args.delta_mm)),
            Delta_mm=float(tuned_best.get("Delta_mm", args.Delta_mm)),
            a_max_mm=a_max
        )

        policies = [
            ("Threshold", thr),
            ("FAORule", fao),
            ("Calendar", cal),
            ("TunedFAORule", tuned),
        ]

        for name, pol in policies:
            # fresh env per (seed, method) to align randomness
            env = build_env_compat(env_cfg, eval_train_cfg, seed=seed)
            env = apply_robust_wrappers(env, setting=args.setting, noise_cfg=noise_cfg_norm, seed=seed)

            out_dir = os.path.join(args.out, name, f"seed{seed}")
            m = rollout_policy(env, name, pol, out_dir, seed=seed, scenario=args.setting)
            all_metrics.append(m)

        print(f"[OK] seed {seed} done")

    save_json(os.path.join(args.out, "_baseline_metrics_all.json"), {"runs": all_metrics, "setting": args.setting})
    print(f"[OK] Baselines done. Saved: {os.path.join(args.out, '_baseline_metrics_all.json')}")


if __name__ == "__main__":
    main()
