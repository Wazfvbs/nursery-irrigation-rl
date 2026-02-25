from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_cmd(name: str, cmd: List[str]) -> int:
    print(f"\n[RUN] {name}")
    print(" ".join(cmd))
    p = subprocess.run(cmd, check=False)
    if p.returncode == 0:
        print(f"[OK] {name}")
    else:
        print(f"[FAIL] {name} (code={p.returncode})")
    return int(p.returncode)


def pick_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--tables", type=str, default="output/tables_ref")
    ap.add_argument("--out", type=str, default="output/figures")
    ap.add_argument("--seed", type=int, default=42, help="representative seed for single-trajectory figures")
    ap.add_argument("--clip", type=float, default=2.5, help="clip for Fig6 panel-b")
    ap.add_argument("--noise_setting", type=str, default="noise_only", help="setting used by Fig8 table")

    ap.add_argument("--skip_fig5", action="store_true")
    ap.add_argument("--skip_fig6", action="store_true")
    ap.add_argument("--skip_fig7", action="store_true")
    ap.add_argument("--skip_fig8", action="store_true")
    ap.add_argument("--skip_fig9", action="store_true")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    tables = (root / args.tables).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    here = Path(__file__).resolve().parent

    rc_all: List[int] = []

    # -------------------------
    # Fig5: representative trajectory
    # -------------------------
    if not args.skip_fig5:
        traj = pick_existing(
            [
                root / "output" / "ppo_runs" / f"seed{args.seed}" / "eval" / "trajectory.csv",
                root / "output" / "ablation_ref" / "Full" / f"seed{args.seed}" / "eval" / "trajectory.csv",
                ]
        )
        if traj is None:
            print("[WARN] Fig5 skipped: cannot find representative PPO trajectory.")
        else:
            rc_all.append(
                run_cmd(
                    "Fig5",
                    [
                        py,
                        str(here / "plot_fig5_trajectory.py"),
                        "--traj",
                        str(traj),
                        "--out",
                        str(out_dir / "fig5_representative"),
                    ],
                )
            )

    # -------------------------
    # Fig6: nominal comparison
    # -------------------------
    if not args.skip_fig6:
        rc_all.append(
            run_cmd(
                "Fig6",
                [
                    py,
                    str(here / "plot_fig6.py"),
                    "--root",
                    str(root),
                    "--out",
                    str(out_dir / "fig6_nominal_failure"),
                    "--clip",
                    str(args.clip),
                ],
            )
        )

    # -------------------------
    # Fig7: main results from Table8
    # -------------------------
    if not args.skip_fig7:
        table8 = tables / "Table8_main_results_mean_std.csv"
        if not table8.exists():
            print(f"[WARN] Fig7 skipped: missing {table8}")
        else:
            rc_all.append(
                run_cmd(
                    "Fig7",
                    [
                        py,
                        str(here / "fig7.py"),
                        "--csv",
                        str(table8),
                        "--out",
                        str(out_dir / "fig7_main"),
                    ],
                )
            )

    # -------------------------
    # Fig8: robustness (noise)
    # -------------------------
    if not args.skip_fig8:
        table9 = pick_existing(
            [
                tables / f"Table9_robust_{args.noise_setting}_mean_std.csv",
                tables / "Table9_robust_noise_only_mean_std.csv",
                tables / "Table9_robust_noise_mean_std.csv",
                ]
        )
        if table9 is None:
            print("[WARN] Fig8 skipped: cannot find Table9 robustness csv.")
        else:
            rc_all.append(
                run_cmd(
                    "Fig8",
                    [
                        py,
                        str(here / "plot_fig8.py"),
                        "--table9",
                        str(table9),
                        "--out_prefix",
                        str(out_dir / "fig8_robustness"),
                        "--stress_max",
                        "100",
                    ],
                )
            )

    # -------------------------
    # Fig9: ablation from Table10
    # -------------------------
    if not args.skip_fig9:
        t10_main = tables / "Table10_ablation_mean_std.csv"
        t10_supp = tables / "Table10_ablation_supp_mean_std.csv"
        if (not t10_main.exists()) or (not t10_supp.exists()):
            print("[WARN] Fig9 skipped: missing Table10 ablation csv files.")
        else:
            rc_all.append(
                run_cmd(
                    "Fig9",
                    [
                        py,
                        str(here / "fig9.py"),
                        "--main_csv",
                        str(t10_main),
                        "--supp_csv",
                        str(t10_supp),
                        "--out_main",
                        str(out_dir / "fig9_ablation_main.png"),
                        "--out_supp",
                        str(out_dir / "fig9_ablation_supp.png"),
                    ],
                )
            )

    if any(code != 0 for code in rc_all):
        raise SystemExit(1)
    print(f"\n[OK] figure batch finished. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
