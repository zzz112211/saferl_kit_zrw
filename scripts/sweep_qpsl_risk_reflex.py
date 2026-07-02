#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import sys
from pathlib import Path


MODEL_PREFIX = "paper_1m_20260629_qpsl_seed"


def parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def add_common_args(cmd, seed, eval_episodes, out_dir):
    cmd.extend(
        [
            "--model_tag",
            f"{MODEL_PREFIX}{seed}",
            "--seed",
            str(seed),
            "--eval_episodes",
            str(eval_episodes),
            "--out_dir",
            str(out_dir),
        ]
    )


def run_eval(cmd):
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_commands(args):
    seeds = parse_int_list(args.seeds)
    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    gammas = parse_float_list(args.gammas)
    thetas = parse_float_list(args.thetas)
    ks = parse_float_list(args.ks)
    lambda_brakes = parse_float_list(args.lambda_brakes)
    lambda_throttles = parse_float_list(args.lambda_throttles)

    variants = ["longitudinal"]
    if args.include_steering_limit:
        variants.append("steering_limit")

    evaluator = Path(__file__).resolve().parents[1] / "eval_qpsl_risk_reflex.py"
    out_dir = Path(args.out_dir)
    commands = []

    for seed in seeds:
        baseline_cmd = [sys.executable, str(evaluator), "--variant", "baseline"]
        add_common_args(baseline_cmd, seed, args.eval_episodes, out_dir)
        commands.append(baseline_cmd)

        for variant in variants:
            for alpha, beta, gamma, theta, k, lambda_brake, lambda_throttle in itertools.product(
                alphas,
                betas,
                gammas,
                thetas,
                ks,
                lambda_brakes,
                lambda_throttles,
            ):
                cmd = [sys.executable, str(evaluator), "--variant", variant]
                add_common_args(cmd, seed, args.eval_episodes, out_dir)
                cmd.extend(
                    [
                        "--alpha",
                        str(alpha),
                        "--beta",
                        str(beta),
                        "--gamma",
                        str(gamma),
                        "--theta",
                        str(theta),
                        "--k",
                        str(k),
                        "--lambda_brake",
                        str(lambda_brake),
                        "--lambda_throttle",
                        str(lambda_throttle),
                    ]
                )
                commands.append(cmd)

    return commands


def select_commands(commands, task_index, task_count):
    if task_index is None and task_count is None:
        return commands
    if task_index is None or task_count is None:
        raise ValueError("--task_index and --task_count must be provided together")
    if task_count <= 0:
        raise ValueError("--task_count must be positive")
    if task_index < 0 or task_index >= task_count:
        raise ValueError("--task_index must satisfy 0 <= task_index < task_count")
    return [cmd for index, cmd in enumerate(commands) if index % task_count == task_index]


def enforce_overwrite_guard(out_dir, overwrite):
    path = Path(out_dir)
    if overwrite or not path.exists():
        return
    existing = sorted(path.rglob("*_summary.csv"))
    if existing:
        sample = ", ".join(str(item) for item in existing[:3])
        more = "" if len(existing) <= 3 else f", ... ({len(existing)} total)"
        raise FileExistsError(
            f"{path} already contains summary CSV files ({sample}{more}); "
            "use --overwrite to append/re-run into this directory"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run QPSL risk-reflex evaluation screening sweep.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--eval_episodes", default=20, type=int)
    parser.add_argument("--out_dir", default="runs/risk_reflex_screen")
    parser.add_argument("--thetas", default="0.40,0.45,0.50")
    parser.add_argument("--lambda_brakes", default="0.20,0.30,0.40")
    parser.add_argument("--lambda_throttles", default="0.35")
    parser.add_argument("--alphas", default="0.88,0.92")
    parser.add_argument("--betas", default="0.35,0.45")
    parser.add_argument("--gammas", default="0.06,0.08")
    parser.add_argument("--ks", default="12.0")
    parser.add_argument("--include_steering_limit", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--task_index", default=None, type=int)
    parser.add_argument("--task_count", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    commands = build_commands(args)
    selected_commands = select_commands(commands, args.task_index, args.task_count)

    print(f"total commands: {len(commands)}", flush=True)
    print(f"selected commands: {len(selected_commands)}", flush=True)

    if args.dry_run:
        for cmd in selected_commands:
            print(" ".join(cmd), flush=True)
        return

    enforce_overwrite_guard(args.out_dir, args.overwrite)
    for cmd in selected_commands:
        run_eval(cmd)

    print(f"completed {len(selected_commands)} evaluation runs", flush=True)


if __name__ == "__main__":
    main()
