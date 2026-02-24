from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import csv


def _set_ucb_enabled(env, enabled: bool):
    """
    Best-effort switch for UCB bonus during evaluation.
    RewardWrapper uses ActionBinUCB(cfg.enabled=...).
    We try to set `env.ucb.cfg.enabled` if present.

    Returns:
        prev_enabled (bool|None): previous state if we could read it, else None
    """
    try:
        if hasattr(env, "ucb") and hasattr(env.ucb, "cfg") and hasattr(env.ucb.cfg, "enabled"):
            prev = bool(env.ucb.cfg.enabled)
            env.ucb.cfg.enabled = bool(enabled)
            return prev
    except Exception:
        pass
    return None


def evaluate_policy(
        env,
        model,
        out_dir: str,
        deterministic: bool = True,
        max_steps: Optional[int] = None,
        use_ucb: bool = False,
        **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate a trained policy and export trajectory.csv.

    ✅ Assumes reward is computed inside env.step()
       (i.e., env is wrapped by RewardWrapper).

    Parameters
    ----------
    env : gym.Env
        Environment instance (preferably RewardWrapper(env))
    model : stable_baselines3 model OR any object with .predict(obs) -> (action, state)
        Loaded SB3 model (e.g., PPO.load(...))
    out_dir : str
        Output directory where trajectory.csv will be saved
    deterministic : bool
        Whether to use deterministic action in predict()
    max_steps : Optional[int]
        Optional cap on rollout steps, otherwise run until terminated/truncated
    use_ucb : bool
        Whether to enable UCB bonus during evaluation.
        Paper protocol: evaluation should disable UCB => use_ucb=False.
    kwargs : dict
        Ignored extra args for forward/backward compatibility.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Paper protocol: evaluation disables UCB bonus (training may use it)
    prev_ucb = _set_ucb_enabled(env, enabled=use_ucb)

    obs, info = env.reset()

    rows: List[Dict[str, Any]] = []
    done = False
    t = 0

    while not done:
        # SB3 predict for a single env (non-VecEnv) accepts 1D obs
        action, _ = model.predict(obs, deterministic=deterministic)

        # Step env (reward is now computed by RewardWrapper)
        obs_next, reward, terminated, truncated, info_next = env.step(action)
        done = bool(terminated or truncated)

        # -------- Collect basic fields (safe with .get defaults) --------
        row: Dict[str, Any] = {}

        row["t"] = t
        row["day"] = int(info_next.get("day", t))

        # stage position in nursery/crop cycle (0~1). Used by dynamic target.
        row["stage_norm"] = float(info_next.get("stage_norm", 0.0))

        row["Dr"] = float(info_next.get("Dr_mm", 0.0))
        row["theta"] = float(info_next.get("theta", 0.0))

        row["TAW"] = float(info_next.get("TAW_mm", 0.0))
        row["RAW"] = float(info_next.get("RAW_mm", 0.0))

        row["ET0"] = float(info_next.get("ET0", 0.0))
        row["ETc"] = float(info_next.get("ETc", 0.0))

        row["Kc"] = float(info_next.get("Kc", 0.0))
        row["Ks"] = float(info_next.get("Ks", 0.0))

        # Action after clipping is best read from info["I_mm"]
        if "I_mm" in info_next:
            row["I"] = float(info_next["I_mm"])
        else:
            # fallback if env doesn't write it
            try:
                row["I"] = float(action[0])
            except Exception:
                row["I"] = float(action)

        # ✅ NEW: raw action and clip flag (for clip_rate)
        # nursery_env writes: I_raw_mm, clipped
        row["I_raw"] = float(info_next.get("I_raw_mm", row["I"]))
        row["clipped"] = int(info_next.get("clipped", 0))

        # target intervals (RewardWrapper injects train + reference)
        row["Dr_lo"] = float(info_next.get("Dr_lo", 0.0))        # backward-compatible (=train)
        row["Dr_hi"] = float(info_next.get("Dr_hi", 0.0))
        row["Dr_lo_train"] = float(info_next.get("Dr_lo_train", row["Dr_lo"]))
        row["Dr_hi_train"] = float(info_next.get("Dr_hi_train", row["Dr_hi"]))
        row["Dr_lo_ref"] = float(info_next.get("Dr_lo_ref", row["Dr_lo"]))
        row["Dr_hi_ref"] = float(info_next.get("Dr_hi_ref", row["Dr_hi"]))

        # midpoint of reference interval (for IAE_mid_ref)
        row["Dr_mid_ref"] = float(info_next.get(
            "Dr_mid_ref",
            0.5 * (row["Dr_lo_ref"] + row["Dr_hi_ref"])
        ))

        # per-step auxiliary values (optional but useful for plotting)
        row["in_ref"] = int((row["Dr"] >= row["Dr_lo_ref"]) and (row["Dr"] <= row["Dr_hi_ref"]))
        row["e_mid_ref"] = float(abs(row["Dr"] - row["Dr_mid_ref"]))

        # ucb bonus (optional)
        row["ucb_bonus"] = float(info_next.get("ucb_bonus", 0.0))

        # reward
        row["reward"] = float(reward)

        # termination flags
        row["terminated"] = int(bool(terminated))
        row["truncated"] = int(bool(truncated))

        # -------- Optional: flatten reward decomposition terms --------
        terms = info_next.get("reward_terms", None)
        if isinstance(terms, dict):
            # Put reward terms into columns with prefix "rt_"
            for k, v in terms.items():
                key = f"rt_{k}"
                try:
                    row[key] = float(v)
                except Exception:
                    row[key] = v

        rows.append(row)

        obs = obs_next
        t += 1

        if max_steps is not None and t >= max_steps:
            break

    # Restore UCB enabled state (best-effort)
    if prev_ucb is not None:
        _set_ucb_enabled(env, enabled=prev_ucb)

    # -------- Write trajectory.csv --------
    csv_path = os.path.join(out_dir, "trajectory.csv")

    # Build header as union of all keys (stable order: basic -> extra sorted)
    base_cols = [
        "t", "day", "stage_norm",
        "Dr", "theta",
        "TAW", "RAW",
        "ET0", "ETc",
        "Kc", "Ks",
        # actions
        "I", "I_raw", "clipped",
        # intervals
        "Dr_lo", "Dr_hi",
        "Dr_lo_train", "Dr_hi_train",
        "Dr_lo_ref", "Dr_hi_ref", "Dr_mid_ref",
        # aux
        "in_ref", "e_mid_ref",
        "ucb_bonus",
        "reward",
        "terminated", "truncated",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    extra_cols = sorted([k for k in all_keys if k not in base_cols])
    header = base_cols + extra_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            # ensure missing fields are blank
            w.writerow({k: r.get(k, "") for k in header})

    return {
        "out_dir": out_dir,
        "trajectory_csv": csv_path,
        "steps": int(t),
        "use_ucb_eval": bool(use_ucb),
    }
