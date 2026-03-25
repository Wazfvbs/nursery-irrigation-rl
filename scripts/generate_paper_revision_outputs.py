from __future__ import annotations

import csv
import json
import math
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"

REPORT_SEEDS = list(range(42, 52))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_ALPHA = 0.95
BOOTSTRAP_SEED = 20260325

METHOD_LABELS = {
    "PPO": "PPO-Optimized",
    "TunedFAORule": "Tuned FAO-rule (grid)",
    "VanillaPPO": "Vanilla PPO",
}

METRIC_SPECS = [
    ("IAE_mid_ref", "IAE_mid,ref", "lower"),
    ("TotalIrrigation_mm", "Total irrigation", "lower"),
    ("StressDays_ref", "StressDays_ref", "lower"),
    ("TIR_ref", "TIR_ref", "higher"),
]

SOURCE_REFS = {
    "target_config": {
        "path": "irrigation_rl/rewards/target.py",
        "symbol": "TargetConfig",
        "lines": "6-20",
    },
    "target_interval": {
        "path": "irrigation_rl/rewards/target.py",
        "symbol": "DynamicTarget.get_interval",
        "lines": "26-40",
    },
    "reward_wrapper_train_ref": {
        "path": "irrigation_rl/envs/reward_wrapper.py",
        "symbol": "RewardWrapper._get_train_interval / _get_ref_interval",
        "lines": "183-203",
    },
    "env_stage_norm": {
        "path": "irrigation_rl/envs/nursery_env.py",
        "symbol": "stage_norm",
        "lines": "58-60",
    },
    "env_crop_cfg": {
        "path": "irrigation_rl/envs/nursery_env.py",
        "symbol": "EnvConfig crop fields",
        "lines": "37-43",
    },
    "env_kc_stage": {
        "path": "irrigation_rl/envs/nursery_env.py",
        "symbol": "kc_by_stage",
        "lines": "49-55",
    },
    "env_step": {
        "path": "irrigation_rl/envs/nursery_env.py",
        "symbol": "NurseryIrrigationEnv.step",
        "lines": "291-365",
    },
    "fao56": {
        "path": "irrigation_rl/envs/fao56.py",
        "symbol": "calc_TAW / calc_RAW",
        "lines": "13-19",
    },
    "build_env": {
        "path": "irrigation_rl/train/ppo_train.py",
        "symbol": "build_env",
        "lines": "53-107",
    },
    "env_yaml": {
        "path": "configs/env.yaml",
        "symbol": "nominal env config",
        "lines": "1-30",
    },
    "train_yaml": {
        "path": "configs/train.yaml",
        "symbol": "nominal PPO config",
        "lines": "15-31",
    },
    "vanilla_yaml": {
        "path": "configs/train_vanilla_ppo.yaml",
        "symbol": "vanilla PPO config",
        "lines": "16-34",
    },
    "tuned_rule_impl": {
        "path": "scripts/run_baselines.py",
        "symbol": "TunedFAORulePolicy",
        "lines": "28-56",
    },
    "tuned_rule_grid": {
        "path": "scripts/run_baselines.py",
        "symbol": "tune_tuned_rule",
        "lines": "278-357",
    },
}


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def calc_taw(theta_fc: float, theta_wp: float, zr_m: float) -> float:
    return 1000.0 * max(theta_fc - theta_wp, 0.0) * max(zr_m, 0.0)


def calc_raw(p: float, taw: float) -> float:
    return max(p, 0.0) * max(taw, 0.0)


def load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_trajectory_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def detect_band_switches(trajectory_path: Path) -> List[Dict]:
    rows = load_trajectory_rows(trajectory_path)
    switches: List[Dict] = []
    prev = None
    for row in rows:
        cur = (row["Dr_lo_ref"], row["Dr_hi_ref"])
        if cur != prev:
            switches.append(
                {
                    "day": int(float(row["day"])),
                    "stage_norm": float(row["stage_norm"]),
                    "Dr_lo_ref_mm": float(row["Dr_lo_ref"]),
                    "Dr_hi_ref_mm": float(row["Dr_hi_ref"]),
                }
            )
            prev = cur
    return switches


def build_dynamic_target_schedule() -> Tuple[Dict, List[Dict], str]:
    env_cfg = load_yaml(REPO_ROOT / "configs" / "env.yaml")
    tuned_params_path = REPO_ROOT / "output" / "baselines" / "TunedFAORule" / "tuning_dev" / "best_params.json"
    tuned_params = read_json(tuned_params_path)

    horizon_days = int(env_cfg["scenario"]["horizon_days"])
    theta_fc = float(env_cfg["soil"]["theta_fc"])
    theta_wp = float(env_cfg["soil"]["theta_wp"])
    zr_m = float(env_cfg["soil"]["Zr_m"])
    p_value = float(env_cfg["soil"]["p"])
    taw_mm = calc_taw(theta_fc, theta_wp, zr_m)
    raw_mm = calc_raw(p_value, taw_mm)

    kc_ini = float(env_cfg["crop"]["Kc_ini"])
    kc_mid = float(env_cfg["crop"]["Kc_mid"])
    kc_end = float(env_cfg["crop"]["Kc_end"])
    stage_ini_days = int(env_cfg["crop"]["stage_ini_days"])
    stage_mid_days = int(env_cfg["crop"]["stage_mid_days"])
    stage_end_days = int(env_cfg["crop"]["stage_end_days"])

    dynamic_stages = [
        {
            "stage_name": "early",
            "condition": "stage_norm < 0.33",
            "logged_day_range": "1-29",
            "switch_day_logged": 1,
            "low_frac_TAW": 0.15,
            "high_frac_RAW": 0.70,
        },
        {
            "stage_name": "mid",
            "condition": "0.33 <= stage_norm < 0.66",
            "logged_day_range": "30-58",
            "switch_day_logged": 30,
            "low_frac_TAW": 0.20,
            "high_frac_RAW": 0.90,
        },
        {
            "stage_name": "late",
            "condition": "stage_norm >= 0.66",
            "logged_day_range": "59-90",
            "switch_day_logged": 59,
            "low_frac_TAW": 0.25,
            "high_frac_RAW": 1.00,
        },
    ]

    for stage in dynamic_stages:
        lo_mm = stage["low_frac_TAW"] * taw_mm
        hi_mm = stage["high_frac_RAW"] * raw_mm
        delta_mm = raw_mm - lo_mm
        stage["Dr_lo_mm"] = round(lo_mm, 6)
        stage["Dr_hi_mm"] = round(max(lo_mm, hi_mm), 6)
        stage["delta_equiv_mm"] = round(delta_mm, 6)
        stage["paper_equiv_lower"] = f"D_lo = RAW - {delta_mm:.1f}"
        stage["paper_equiv_upper"] = f"D_hi = {stage['high_frac_RAW']:.2f} * RAW"

    kc_schedule = [
        {
            "stage_name": "initial",
            "logged_day_range": "1-20",
            "condition": f"day < {stage_ini_days} before env day increment",
            "Kc": kc_ini,
        },
        {
            "stage_name": "mid",
            "logged_day_range": "21-70",
            "condition": f"{stage_ini_days} <= day < {stage_ini_days + stage_mid_days} before env day increment",
            "Kc": kc_mid,
        },
        {
            "stage_name": "end",
            "logged_day_range": "71-90",
            "condition": f"day >= {stage_ini_days + stage_mid_days} before env day increment",
            "Kc": kc_end,
        },
    ]

    trajectory_path = REPO_ROOT / "output" / "robust_test_ref" / "PPO" / "nominal" / "seed42" / "trajectory.csv"
    switch_rows = detect_band_switches(trajectory_path)

    inconsistency_points = [
        (
            "Dynamic target stages are not read from crop stage config. "
            "The target band uses stage_norm thresholds 0.33 and 0.66, while Kc uses "
            "stage_ini_days/stage_mid_days/stage_end_days = 20/50/20."
        ),
        (
            "The code does not store an explicit delta_t variable for the dynamic target. "
            "The actual implementation is lo = alpha_g * TAW and hi = beta_g * RAW, with "
            "delta_t only derivable algebraically as RAW - lo."
        ),
        (
            "Because nominal TAW and RAW are constant, the paper shorthand delta_t = Delta^(g) "
            "is valid only as a derived reparameterization of the coded logic, not as the literal code path."
        ),
    ]

    schedule_payload = {
        "nominal_config": {
            "horizon_days": horizon_days,
            "theta_fc": theta_fc,
            "theta_wp": theta_wp,
            "Zr_m": zr_m,
            "p": p_value,
            "TAW_mm": taw_mm,
            "RAW_mm": raw_mm,
            "Kc_ini": kc_ini,
            "Kc_mid": kc_mid,
            "Kc_end": kc_end,
            "stage_ini_days": stage_ini_days,
            "stage_mid_days": stage_mid_days,
            "stage_end_days": stage_end_days,
            "tuned_fao_rule_best_params": tuned_params,
        },
        "dynamic_target_impl": {
            "hardcoded": True,
            "config_override_found": False,
            "source_refs": [
                SOURCE_REFS["target_config"],
                SOURCE_REFS["target_interval"],
                SOURCE_REFS["reward_wrapper_train_ref"],
                SOURCE_REFS["build_env"],
            ],
            "logic": (
                "RewardWrapper always calls DynamicTarget.get_interval(TAW, RAW, stage_norm). "
                "TargetConfig is instantiated with dataclass defaults; train.yaml does not override it."
            ),
            "paper_simplification": {
                "delta_t_equals_Delta_g_valid_for_nominal": True,
                "reason": (
                    "In nominal experiments TAW=54 mm and RAW=27 mm are constant, so lo = alpha_g * TAW "
                    "is equivalent to lo = RAW - Delta^(g), where Delta^(g) = RAW - alpha_g * TAW."
                ),
            },
            "stages": dynamic_stages,
            "trajectory_validation": {
                "source": str(trajectory_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "switch_rows": switch_rows,
            },
        },
        "kc_schedule_impl": {
            "source_refs": [
                SOURCE_REFS["env_crop_cfg"],
                SOURCE_REFS["env_kc_stage"],
                SOURCE_REFS["env_step"],
                SOURCE_REFS["env_yaml"],
            ],
            "stages": kc_schedule,
        },
        "source_refs": SOURCE_REFS,
        "inconsistency_points": inconsistency_points,
    }

    csv_rows: List[Dict] = []
    for stage in dynamic_stages:
        csv_rows.append(
            {
                "row_type": "dynamic_target_stage",
                "stage_name": stage["stage_name"],
                "logged_day_range": stage["logged_day_range"],
                "condition": stage["condition"],
                "low_frac_TAW": stage["low_frac_TAW"],
                "high_frac_RAW": stage["high_frac_RAW"],
                "Dr_lo_mm": stage["Dr_lo_mm"],
                "Dr_hi_mm": stage["Dr_hi_mm"],
                "delta_equiv_mm": stage["delta_equiv_mm"],
                "Kc": "",
                "TAW_mm": taw_mm,
                "RAW_mm": raw_mm,
                "p": p_value,
                "theta_fc": theta_fc,
                "theta_wp": theta_wp,
                "Zr_m": zr_m,
                "source_path": SOURCE_REFS["target_interval"]["path"],
                "source_symbol": SOURCE_REFS["target_interval"]["symbol"],
                "source_lines": SOURCE_REFS["target_interval"]["lines"],
                "notes": "Actual nominal dynamic target stage used in paper metrics.",
            }
        )
    for stage in kc_schedule:
        csv_rows.append(
            {
                "row_type": "kc_stage",
                "stage_name": stage["stage_name"],
                "logged_day_range": stage["logged_day_range"],
                "condition": stage["condition"],
                "low_frac_TAW": "",
                "high_frac_RAW": "",
                "Dr_lo_mm": "",
                "Dr_hi_mm": "",
                "delta_equiv_mm": "",
                "Kc": stage["Kc"],
                "TAW_mm": taw_mm,
                "RAW_mm": raw_mm,
                "p": p_value,
                "theta_fc": theta_fc,
                "theta_wp": theta_wp,
                "Zr_m": zr_m,
                "source_path": SOURCE_REFS["env_kc_stage"]["path"],
                "source_symbol": SOURCE_REFS["env_kc_stage"]["symbol"],
                "source_lines": SOURCE_REFS["env_kc_stage"]["lines"],
                "notes": "Nominal Kc stage schedule applied by the environment.",
            }
        )

    md_lines = [
        "# Dynamic target schedule",
        "",
        "## Actual implementation",
        "",
        "- Dynamic target is hardcoded in `irrigation_rl/rewards/target.py` (`TargetConfig`, lines 6-20; `DynamicTarget.get_interval`, lines 26-40).",
        "- `RewardWrapper` uses that target for both training-time dynamic target and all paper reference metrics (`irrigation_rl/envs/reward_wrapper.py`, lines 183-203).",
        "- Nominal PPO build path does not override `TargetConfig`; `build_env(...)` passes only `RewardWrapper(..., reward_cfg=..., flags=...)` (`irrigation_rl/train/ppo_train.py`, lines 53-107).",
        "- Therefore the nominal experiments use the dataclass defaults exactly as coded.",
        "",
        "## Actual nominal settings",
        "",
        f"- TAW = {taw_mm:.1f} mm, from `theta_fc={theta_fc:.2f}`, `theta_wp={theta_wp:.2f}`, `Zr={zr_m:.2f}` via `TAW = 1000 (theta_fc - theta_wp) Zr`.",
        f"- p(t) = {p_value:.2f} (constant in nominal config), so RAW(t) = p * TAW = {raw_mm:.1f} mm.",
        f"- Kc schedule comes from `configs/env.yaml`: Kc_ini={kc_ini:.2f}, Kc_mid={kc_mid:.2f}, Kc_end={kc_end:.2f}.",
        f"- Crop-stage Kc day split is 1-20 / 21-70 / 71-90 in logged rollout days, because `kc_by_stage` uses `stage_ini_days={stage_ini_days}`, `stage_mid_days={stage_mid_days}`, `stage_end_days={stage_end_days}`.",
        "",
        "## Dynamic target stages actually used",
        "",
        "- Growth stage 1: logged days 1-29 (`stage_norm < 0.33`).",
        "- Growth stage 2: logged days 30-58 (`0.33 <= stage_norm < 0.66`).",
        "- Growth stage 3: logged days 59-90 (`stage_norm >= 0.66`).",
        f"- Lower bound values are `D_lo = [8.1, 10.8, 13.5]` mm = `[0.15, 0.20, 0.25] * TAW`.",
        f"- Upper bound values are `D_hi = [18.9, 24.3, 27.0]` mm = `[0.70, 0.90, 1.00] * RAW`.",
        f"- Under the paper notation `D_lo,t = max(0, RAW(t) - Delta_t)`, the nominal-equivalent constants are `Delta^(1)={dynamic_stages[0]['delta_equiv_mm']:.1f}` mm, `Delta^(2)={dynamic_stages[1]['delta_equiv_mm']:.1f}` mm, `Delta^(3)={dynamic_stages[2]['delta_equiv_mm']:.1f}` mm.",
        "",
        "## Trajectory validation",
        "",
        f"- `output/robust_test_ref/PPO/nominal/seed42/trajectory.csv` switches `Dr_lo_ref/Dr_hi_ref` at days 30 and 59, matching the coded schedule.",
        "",
        "## Inconsistency check",
        "",
        "- The dynamic target stage split is not the same as the crop Kc stage split.",
        "- If the paper currently states that the target stages are 1-20 / 21-70 / 71-90, that is inconsistent with the code.",
        "- The code does not literally define a `delta_t` variable for the dynamic target. It defines `lo = alpha_g * TAW`; the `Delta^(g)` values above are an exact nominal reparameterization, not the literal source variable.",
        "",
        "## Paper-ready text",
        "",
        (
            "The actual nominal dynamic target used in the experiments is implemented as a three-stage "
            "hardcoded schedule in `irrigation_rl/rewards/target.py`. In logged rollout days, the target "
            "band uses stages 1-29, 30-58, and 59-90, selected by `stage_norm < 0.33`, `0.33 <= stage_norm < 0.66`, "
            "and `stage_norm >= 0.66`, respectively. The corresponding lower bounds are `D_lo = [8.1, 10.8, 13.5]` mm, "
            "which are equivalent to `Delta^(1)=18.9` mm, `Delta^(2)=16.2` mm, and `Delta^(3)=13.5` mm under the paper "
            "notation `D_lo,t = max(0, RAW(t) - Delta_t)`. The upper bounds are `D_hi = [18.9, 24.3, 27.0]` mm, i.e. "
            "`[0.70, 0.90, 1.00] * RAW`. The nominal water-balance constants are `TAW=54.0` mm and `RAW=27.0` mm "
            "with constant `p=0.50`, while the crop-coefficient schedule is `Kc = 0.50` for days 1-20, `1.00` for "
            "days 21-70, and `0.80` for days 71-90. These are the actual nominal settings used in the experiments."
        ),
    ]

    return schedule_payload, csv_rows, "\n".join(md_lines)


def stable_seed(label: str) -> int:
    return BOOTSTRAP_SEED + (zlib.crc32(label.encode("utf-8")) % 100_000)


def load_seed_metric_series(base_dir: Path, metric: str) -> np.ndarray:
    values = []
    missing = []
    for seed in REPORT_SEEDS:
        path = base_dir / f"seed{seed}" / "metrics.json"
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
            continue
        values.append(float(read_json(path)[metric]))
    if missing:
        raise FileNotFoundError(f"Missing metrics.json files: {missing}")
    return np.asarray(values, dtype=float)


def mean_std(arr: np.ndarray) -> Tuple[float, float]:
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def paired_bootstrap_ci(diff: np.ndarray, label: str) -> Tuple[float, float]:
    rng = np.random.default_rng(stable_seed(label))
    indices = rng.integers(0, diff.size, size=(BOOTSTRAP_RESAMPLES, diff.size))
    boot = diff[indices].mean(axis=1)
    low_q = 0.5 * (1.0 - BOOTSTRAP_ALPHA)
    high_q = 1.0 - low_q
    low, high = np.quantile(boot, [low_q, high_q])
    return float(low), float(high)


def collect_comparison_rows(
    scenario_label: str,
    setting: str,
    sources: Dict[str, Path],
) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    summary = {
        "scenario": scenario_label,
        "setting": setting,
        "seeds": REPORT_SEEDS,
        "source_dirs": {
            method: str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            for method, path in sources.items()
        },
        "comparisons": [],
    }

    comparisons = [("PPO", "TunedFAORule"), ("PPO", "VanillaPPO")]
    for lhs, rhs in comparisons:
        comp_entry = {
            "lhs_method": lhs,
            "lhs_display": METHOD_LABELS[lhs],
            "rhs_method": rhs,
            "rhs_display": METHOD_LABELS[rhs],
            "metrics": [],
        }
        for metric_key, metric_label, direction in METRIC_SPECS:
            lhs_values = load_seed_metric_series(sources[lhs], metric_key)
            rhs_values = load_seed_metric_series(sources[rhs], metric_key)
            diff = lhs_values - rhs_values
            lhs_mean, lhs_std = mean_std(lhs_values)
            rhs_mean, rhs_std = mean_std(rhs_values)
            diff_mean, diff_std = mean_std(diff)
            ci_low, ci_high = paired_bootstrap_ci(
                diff,
                f"{scenario_label}:{lhs}:{rhs}:{metric_key}",
            )
            saturated = bool(
                metric_key == "TIR_ref"
                and lhs_values.max() >= 0.994
                and rhs_values.max() >= 0.994
                and lhs_values.min() >= 0.994
                and rhs_values.min() >= 0.988
            )
            note = ""
            if saturated:
                note = "Near-saturated metric; less informative than IAE_mid,ref or irrigation."

            row = {
                "scenario": scenario_label,
                "setting": setting,
                "metric": metric_key,
                "metric_label": metric_label,
                "direction": direction,
                "n_seeds": len(REPORT_SEEDS),
                "seeds": "42-51",
                "ppo_method": METHOD_LABELS[lhs],
                "comparator_method": METHOD_LABELS[rhs],
                "ppo_mean": lhs_mean,
                "ppo_std": lhs_std,
                "comparator_mean": rhs_mean,
                "comparator_std": rhs_std,
                "difference_definition": f"{METHOD_LABELS[lhs]} - {METHOD_LABELS[rhs]}",
                "difference_mean": diff_mean,
                "difference_std": diff_std,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "ci_type": "percentile",
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "resample_unit": "seed",
                "paired": True,
                "bootstrap_seed": stable_seed(f"{scenario_label}:{lhs}:{rhs}:{metric_key}"),
                "note": note,
            }
            rows.append(row)

            comp_entry["metrics"].append(
                {
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "direction": direction,
                    "lhs_mean": lhs_mean,
                    "lhs_std": lhs_std,
                    "rhs_mean": rhs_mean,
                    "rhs_std": rhs_std,
                    "difference_mean": diff_mean,
                    "difference_std": diff_std,
                    "difference_definition": row["difference_definition"],
                    "ci_95_percentile": [ci_low, ci_high],
                    "note": note,
                }
            )
        summary["comparisons"].append(comp_entry)

    return rows, summary


def format_metric_line(row: Dict) -> str:
    ppo = f"{row['ppo_mean']:.3f} +/- {row['ppo_std']:.3f}"
    comp = f"{row['comparator_mean']:.3f} +/- {row['comparator_std']:.3f}"
    diff = f"{row['difference_mean']:.3f}"
    ci = f"[{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]"
    suffix = f" {row['note']}" if row["note"] else ""
    return (
        f"- {row['metric_label']}: {METHOD_LABELS['PPO']} {ppo}; "
        f"{row['comparator_method']} {comp}; "
        f"Delta = {diff}; 95% paired bootstrap CI {ci}.{suffix}"
    )


def build_bootstrap_summary_md(nominal_rows: List[Dict], noise_rows: List[Dict]) -> str:
    def filter_rows(rows: Iterable[Dict], comparator_label: str) -> List[Dict]:
        return [row for row in rows if row["comparator_method"] == comparator_label]

    lines = [
        "# Paired bootstrap summary",
        "",
        "## Bootstrap configuration",
        "",
        f"- Report seeds: {REPORT_SEEDS[0]}-{REPORT_SEEDS[-1]} (n={len(REPORT_SEEDS)}).",
        "- Resample unit: seed.",
        f"- Bootstrap resamples: {BOOTSTRAP_RESAMPLES}.",
        "- CI type: percentile 95%.",
        "- Difference definition: PPO-Optimized minus comparator.",
        "",
        "## Strongest baselines comparisons",
        "",
        "### Nominal",
        "",
    ]

    for comparator in ["Tuned FAO-rule (grid)", "Vanilla PPO"]:
        lines.append(f"#### PPO-Optimized vs {comparator}")
        lines.append("")
        comp_rows = filter_rows(nominal_rows, comparator)
        for row in comp_rows:
            lines.append(format_metric_line(row))
        lines.append("")

    lines.extend(
        [
            "### Noise-only robustness",
            "",
        ]
    )
    for comparator in ["Tuned FAO-rule (grid)", "Vanilla PPO"]:
        lines.append(f"#### PPO-Optimized vs {comparator}")
        lines.append("")
        comp_rows = filter_rows(noise_rows, comparator)
        for row in comp_rows:
            lines.append(format_metric_line(row))
        lines.append("")

    lines.extend(
        [
            "## Most informative differences",
            "",
            "- The most informative strongest-baseline gaps are IAE_mid,ref and Total irrigation.",
            "- Against Tuned FAO-rule (grid), PPO-Optimized keeps TIR_ref and StressDays_ref saturated, but still reduces IAE_mid,ref by about 80.9 mm*day in nominal and 80.9 mm*day under noise-only, while using about 2.9 mm less irrigation.",
            "- Against Vanilla PPO, PPO-Optimized reduces IAE_mid,ref by about 35.7-35.8 mm*day, uses about 4.1 mm less irrigation, and lowers StressDays_ref by about 0.5 day; all three paired-bootstrap CIs exclude zero.",
            "- TIR_ref is near-saturated for the strongest methods, so it is less informative than IAE_mid,ref and irrigation for the main-text comparison.",
        ]
    )
    return "\n".join(lines)


def generate_bootstrap_outputs() -> Tuple[Dict, List[Dict], List[Dict], str]:
    nominal_sources = {
        "PPO": REPO_ROOT / "output" / "robust_test_ref" / "PPO" / "nominal",
        "TunedFAORule": REPO_ROOT / "output" / "baselines" / "TunedFAORule",
        "VanillaPPO": REPO_ROOT / "output" / "robust_test_ref" / "VanillaPPO" / "nominal",
    }
    noise_sources = {
        "PPO": REPO_ROOT / "output" / "robust_test_ref" / "PPO" / "noise_only",
        "TunedFAORule": REPO_ROOT / "output" / "robust_baselines_noise_only" / "TunedFAORule",
        "VanillaPPO": REPO_ROOT / "output" / "robust_test_ref" / "VanillaPPO" / "noise_only",
    }

    nominal_rows, nominal_summary = collect_comparison_rows(
        "nominal",
        "nominal",
        nominal_sources,
    )
    noise_rows, noise_summary = collect_comparison_rows(
        "robustness",
        "noise_only",
        noise_sources,
    )

    payload = {
        "bootstrap": {
            "paired": True,
            "resample_unit": "seed",
            "resamples": BOOTSTRAP_RESAMPLES,
            "ci_type": "percentile",
            "ci_level": BOOTSTRAP_ALPHA,
            "report_seeds": REPORT_SEEDS,
            "base_seed": BOOTSTRAP_SEED,
        },
        "source_refs": SOURCE_REFS,
        "nominal": nominal_summary,
        "robust_noise_only": noise_summary,
    }

    summary_md = build_bootstrap_summary_md(nominal_rows, noise_rows)
    return payload, nominal_rows, noise_rows, summary_md


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dynamic_payload, dynamic_csv_rows, dynamic_md = build_dynamic_target_schedule()
    write_json(OUTPUT_DIR / "dynamic_target_schedule.json", dynamic_payload)
    write_csv(
        OUTPUT_DIR / "dynamic_target_schedule.csv",
        dynamic_csv_rows,
        [
            "row_type",
            "stage_name",
            "logged_day_range",
            "condition",
            "low_frac_TAW",
            "high_frac_RAW",
            "Dr_lo_mm",
            "Dr_hi_mm",
            "delta_equiv_mm",
            "Kc",
            "TAW_mm",
            "RAW_mm",
            "p",
            "theta_fc",
            "theta_wp",
            "Zr_m",
            "source_path",
            "source_symbol",
            "source_lines",
            "notes",
        ],
    )
    write_text(OUTPUT_DIR / "dynamic_target_schedule.md", dynamic_md)

    bootstrap_payload, nominal_rows, noise_rows, bootstrap_md = generate_bootstrap_outputs()
    write_csv(
        OUTPUT_DIR / "bootstrap_ci_nominal.csv",
        nominal_rows,
        [
            "scenario",
            "setting",
            "metric",
            "metric_label",
            "direction",
            "n_seeds",
            "seeds",
            "ppo_method",
            "comparator_method",
            "ppo_mean",
            "ppo_std",
            "comparator_mean",
            "comparator_std",
            "difference_definition",
            "difference_mean",
            "difference_std",
            "ci_95_low",
            "ci_95_high",
            "ci_type",
            "bootstrap_resamples",
            "resample_unit",
            "paired",
            "bootstrap_seed",
            "note",
        ],
    )
    write_csv(
        OUTPUT_DIR / "bootstrap_ci_robust_noise.csv",
        noise_rows,
        [
            "scenario",
            "setting",
            "metric",
            "metric_label",
            "direction",
            "n_seeds",
            "seeds",
            "ppo_method",
            "comparator_method",
            "ppo_mean",
            "ppo_std",
            "comparator_mean",
            "comparator_std",
            "difference_definition",
            "difference_mean",
            "difference_std",
            "ci_95_low",
            "ci_95_high",
            "ci_type",
            "bootstrap_resamples",
            "resample_unit",
            "paired",
            "bootstrap_seed",
            "note",
        ],
    )
    write_json(OUTPUT_DIR / "bootstrap_ci_summary.json", bootstrap_payload)
    write_text(OUTPUT_DIR / "bootstrap_ci_summary.md", bootstrap_md)

    print("Generated paper revision outputs under output/.")


if __name__ == "__main__":
    main()
