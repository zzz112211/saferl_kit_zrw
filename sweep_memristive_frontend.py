"""Sweep memristive risk-reflex settings against saferl_kit baselines.

The sweep intentionally reuses eval_memristive_frontend.py so that every row is
measured in the same MetaDrive v0.2.4 runtime as the trained checkpoints.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from eval_memristive_frontend import ReflexCfg, run_mode, summarize, write_csv


CANDIDATES: List[Dict[str, Any]] = [
    {
        "name": "nocicim_default",
        "frontend": "nocicim_risk_field",
        "overrides": {},
    },
    {
        "name": "nocicim_gentle",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "slow_front_risk_threshold": 0.65,
            "stop_front_risk_threshold": 0.90,
            "lateral_steer_threshold": 0.65,
            "lateral_throttle_threshold": 0.75,
            "lateral_throttle_cap": 0.45,
            "lateral_steering_cap": 0.80,
            "front_gain": 0.08,
            "front_slow_threshold": 0.82,
            "front_slow_throttle_cap": 0.85,
            "lateral_gain": 0.16,
            "lateral_guard_threshold": 0.82,
        },
    },
    {
        "name": "nocicim_light_memory",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_decay": 0.72,
            "front_gain": 0.05,
            "front_slow_threshold": 0.86,
            "front_slow_throttle_cap": 0.90,
            "lateral_decay": 0.72,
            "lateral_gain": 0.08,
            "lateral_guard_threshold": 0.90,
            "lateral_throttle_cap": 0.60,
            "lateral_steering_cap": 0.90,
        },
    },
    {
        "name": "nocicim_front_only",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "lateral_steer_threshold": 0.99,
            "lateral_throttle_threshold": 1.01,
            "lateral_throttle_cap": 0.90,
            "lateral_steering_cap": 1.00,
            "lateral_gain": 0.00,
            "lateral_guard_threshold": 1.01,
            "front_gain": 0.08,
            "front_slow_threshold": 0.82,
            "front_slow_throttle_cap": 0.85,
        },
    },
    {
        "name": "nocicim_lateral_sensor_only",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "lateral_steer_threshold": 0.99,
            "lateral_throttle_threshold": 1.01,
            "lateral_gain": 0.14,
            "lateral_guard_threshold": 0.86,
            "lateral_throttle_cap": 0.55,
            "lateral_steering_cap": 0.90,
            "front_gain": 0.08,
            "front_slow_threshold": 0.84,
        },
    },
    {
        "name": "nocicim_close_range",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 6.0,
            "stop_distance_m": 2.5,
            "lateral_distance_threshold_m": 3.5,
            "slow_front_risk_threshold": 0.70,
            "stop_front_risk_threshold": 0.92,
            "lateral_steer_threshold": 0.70,
            "lateral_throttle_threshold": 0.80,
            "lateral_throttle_cap": 0.55,
            "lateral_steering_cap": 0.90,
            "front_gain": 0.06,
            "front_slow_threshold": 0.88,
            "front_slow_throttle_cap": 0.90,
            "lateral_gain": 0.10,
            "lateral_guard_threshold": 0.90,
        },
    },
    {
        "name": "nocicim_cost_guard",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 7.0,
            "lateral_distance_threshold_m": 4.0,
            "slow_front_risk_threshold": 0.60,
            "stop_front_risk_threshold": 0.92,
            "lateral_steer_threshold": 0.72,
            "lateral_throttle_threshold": 0.78,
            "lateral_throttle_cap": 0.50,
            "lateral_steering_cap": 0.88,
            "front_decay": 0.78,
            "front_gain": 0.08,
            "front_slow_threshold": 0.84,
            "front_slow_throttle_cap": 0.88,
            "lateral_decay": 0.78,
            "lateral_gain": 0.12,
            "lateral_guard_threshold": 0.88,
        },
    },
    {
        "name": "nocicim_panic_front_only",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 4.5,
            "stop_distance_m": 1.8,
            "lateral_distance_threshold_m": 1.0,
            "slow_front_risk_threshold": 1.01,
            "stop_front_risk_threshold": 0.98,
            "slow_throttle_cap": 0.98,
            "lateral_steer_threshold": 1.01,
            "lateral_throttle_threshold": 1.01,
            "lateral_throttle_cap": 0.95,
            "lateral_steering_cap": 1.00,
            "front_decay": 0.50,
            "front_gain": 0.02,
            "front_slow_threshold": 1.01,
            "front_slow_throttle_cap": 0.98,
            "lateral_decay": 0.50,
            "lateral_gain": 0.00,
            "lateral_guard_threshold": 1.01,
        },
    },
    {
        "name": "nocicim_panic_soft_neutral",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 4.5,
            "stop_distance_m": 1.8,
            "lateral_distance_threshold_m": 1.0,
            "slow_front_risk_threshold": 1.01,
            "stop_front_risk_threshold": 0.98,
            "stop_brake_value": 0.0,
            "slow_throttle_cap": 0.98,
            "lateral_steer_threshold": 1.01,
            "lateral_throttle_threshold": 1.01,
            "lateral_throttle_cap": 0.95,
            "lateral_steering_cap": 1.00,
            "front_decay": 0.50,
            "front_gain": 0.02,
            "front_slow_threshold": 1.01,
            "front_slow_throttle_cap": 0.98,
            "lateral_decay": 0.50,
            "lateral_gain": 0.00,
            "lateral_guard_threshold": 1.01,
        },
    },
    {
        "name": "nocicim_panic_soft_front_only",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 4.5,
            "stop_distance_m": 1.8,
            "lateral_distance_threshold_m": 1.0,
            "slow_front_risk_threshold": 1.01,
            "stop_front_risk_threshold": 0.98,
            "stop_brake_value": 0.6,
            "slow_throttle_cap": 0.98,
            "lateral_steer_threshold": 1.01,
            "lateral_throttle_threshold": 1.01,
            "lateral_throttle_cap": 0.95,
            "lateral_steering_cap": 1.00,
            "front_decay": 0.50,
            "front_gain": 0.02,
            "front_slow_threshold": 1.01,
            "front_slow_throttle_cap": 0.98,
            "lateral_decay": 0.50,
            "lateral_gain": 0.00,
            "lateral_guard_threshold": 1.01,
        },
    },
    {
        "name": "nocicim_soft_front_trim",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 6.0,
            "stop_distance_m": 2.0,
            "lateral_distance_threshold_m": 1.0,
            "slow_front_risk_threshold": 0.70,
            "stop_front_risk_threshold": 0.98,
            "slow_throttle_cap": 0.85,
            "lateral_steer_threshold": 1.01,
            "lateral_throttle_threshold": 1.01,
            "lateral_throttle_cap": 0.95,
            "lateral_steering_cap": 1.00,
            "front_decay": 0.60,
            "front_gain": 0.03,
            "front_slow_threshold": 0.95,
            "front_slow_throttle_cap": 0.90,
            "lateral_decay": 0.50,
            "lateral_gain": 0.00,
            "lateral_guard_threshold": 1.01,
        },
    },
    {
        "name": "nocicim_route_preserve",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 6.0,
            "stop_distance_m": 2.0,
            "lateral_distance_threshold_m": 3.0,
            "slow_front_risk_threshold": 0.85,
            "stop_front_risk_threshold": 0.98,
            "slow_throttle_cap": 0.90,
            "lateral_steer_threshold": 0.85,
            "lateral_throttle_threshold": 0.90,
            "lateral_throttle_cap": 0.75,
            "lateral_steering_cap": 1.00,
            "front_decay": 0.65,
            "front_gain": 0.04,
            "front_slow_threshold": 0.92,
            "front_slow_throttle_cap": 0.92,
            "lateral_decay": 0.65,
            "lateral_gain": 0.04,
            "lateral_guard_threshold": 0.95,
        },
    },
    {
        "name": "nocicim_micro_guard",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 8.0,
            "stop_distance_m": 2.0,
            "lateral_distance_threshold_m": 5.0,
            "slow_front_risk_threshold": 0.99,
            "stop_front_risk_threshold": 0.99,
            "slow_throttle_cap": 0.95,
            "lateral_steer_threshold": 0.95,
            "lateral_throttle_threshold": 0.95,
            "lateral_throttle_cap": 0.85,
            "lateral_steering_cap": 1.00,
            "front_decay": 0.50,
            "front_gain": 0.02,
            "front_slow_threshold": 0.98,
            "front_slow_throttle_cap": 0.95,
            "lateral_decay": 0.50,
            "lateral_gain": 0.03,
            "lateral_guard_threshold": 0.98,
        },
    },
    {
        "name": "nocicim_risk_gated_micro_guard",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 8.0,
            "stop_distance_m": 2.0,
            "lateral_distance_threshold_m": 5.0,
            "slow_front_risk_threshold": 0.99,
            "stop_front_risk_threshold": 0.99,
            "slow_throttle_cap": 0.95,
            "lateral_steer_threshold": 0.95,
            "lateral_throttle_threshold": 0.95,
            "lateral_throttle_cap": 0.85,
            "lateral_steering_cap": 1.00,
            "front_decay": 0.50,
            "front_gain": 0.02,
            "front_slow_threshold": 0.98,
            "front_slow_throttle_cap": 0.95,
            "lateral_decay": 0.50,
            "lateral_gain": 0.03,
            "lateral_guard_threshold": 0.98,
            "allow_direct_stop": True,
            "require_memory_for_stop": False,
            "min_front_risk_for_stop": 0.95,
            "require_risk_for_lateral_guard": True,
            "min_lateral_risk_for_guard": 0.20,
        },
    },
    {
        "name": "nocicim_memory_confirmed_stop",
        "frontend": "nocicim_risk_field",
        "overrides": {
            "front_distance_threshold_m": 8.0,
            "stop_distance_m": 2.0,
            "lateral_distance_threshold_m": 5.0,
            "slow_front_risk_threshold": 0.99,
            "stop_front_risk_threshold": 0.99,
            "slow_throttle_cap": 0.95,
            "lateral_steer_threshold": 0.95,
            "lateral_throttle_threshold": 0.95,
            "lateral_throttle_cap": 0.85,
            "lateral_steering_cap": 1.00,
            "front_decay": 0.50,
            "front_gain": 0.02,
            "front_slow_threshold": 0.98,
            "front_slow_throttle_cap": 0.95,
            "lateral_decay": 0.50,
            "lateral_gain": 0.03,
            "lateral_guard_threshold": 0.98,
            "allow_direct_stop": False,
            "require_memory_for_stop": True,
            "min_front_risk_for_stop": 0.95,
            "require_risk_for_lateral_guard": True,
            "min_lateral_risk_for_guard": 0.20,
        },
    },
]


def candidate_by_name() -> Dict[str, Dict[str, Any]]:
    return {str(candidate["name"]): candidate for candidate in CANDIDATES}


def parse_candidate_names(value: str) -> List[str]:
    value = str(value).strip()
    names = list(candidate_by_name().keys())
    if not value or value.lower() == "all":
        return names
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in names]
    if unknown:
        raise ValueError("unknown candidate(s): {}; choices: {}".format(", ".join(unknown), ", ".join(names)))
    return selected


def build_eval_args(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    cfg = asdict(ReflexCfg())
    values: Dict[str, Any] = {
        "algo": args.algo,
        "checkpoint": args.checkpoint,
        "episodes": int(args.episodes),
        "start_seed": int(args.start_seed),
        "horizon": int(args.horizon),
        "traffic_density": float(args.traffic_density),
        "num_lasers": int(args.num_lasers),
        "write_trace": bool(args.write_trace),
        "outdir": args.outdir,
        "frontends": "",
    }
    for key, default in cfg.items():
        if key == "num_lasers":
            continue
        values[key] = overrides.get(key, default)
    return argparse.Namespace(**values)


def flatten_summary(candidate: str, frontend: str, summary: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"candidate": candidate, "frontend": frontend}
    row.update(summary)
    row["overrides_json"] = json.dumps(overrides, sort_keys=True)
    return row


def apply_deltas(row: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    row["success_delta"] = float(row["success_rate"]) - float(baseline["success_rate"])
    row["clean_success_delta"] = float(row["clean_success_rate"]) - float(baseline["clean_success_rate"])
    row["reward_delta"] = float(row["mean_reward"]) - float(baseline["mean_reward"])
    row["episode_cost_reduction"] = float(baseline["mean_episode_cost"]) - float(row["mean_episode_cost"])
    row["crash_rate_reduction"] = float(baseline["crash_rate"]) - float(row["crash_rate"])
    row["out_of_road_reduction"] = float(baseline["out_of_road_rate"]) - float(row["out_of_road_rate"])
    row["baseline_dominant_gain"] = int(
        row["success_delta"] >= 0.0
        and row["clean_success_delta"] >= 0.0
        and row["episode_cost_reduction"] > 0.0
    )
    row["safe_success_gain"] = int(row["success_delta"] >= 0.0 and row["episode_cost_reduction"] > 0.0)
    row["score"] = (
        100.0 * float(row["success_delta"])
        + 80.0 * float(row["clean_success_delta"])
        + 2.0 * float(row["episode_cost_reduction"])
        + 0.05 * float(row["reward_delta"])
        + 30.0 * float(row["crash_rate_reduction"])
        + 20.0 * float(row["out_of_road_reduction"])
    )


def run_candidate(args: argparse.Namespace, candidate: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    frontend = str(candidate["frontend"])
    name = str(candidate["name"])
    overrides = dict(candidate.get("overrides", {}))
    rows, _traces = run_mode(frontend, build_eval_args(args, overrides))
    renamed_rows: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["mode"] = "{}+{}".format(args.algo, name)
        copied["candidate"] = name
        copied["frontend"] = frontend
        renamed_rows.append(copied)
    summary_rows = summarize(renamed_rows)
    summary = summary_rows[0] if summary_rows else {}
    return renamed_rows, flatten_summary(name, frontend, summary, overrides)


def pick_best(rows: List[Dict[str, Any]], frontend: str = "nocicim_risk_field") -> Dict[str, Any]:
    rows = [row for row in rows if str(row.get("frontend")) == str(frontend)]
    if not rows:
        return {}
    dominant = [row for row in rows if int(row.get("baseline_dominant_gain", 0)) == 1]
    if dominant:
        rows = dominant
    else:
        safe_success = [row for row in rows if int(row.get("safe_success_gain", 0)) == 1]
        if safe_success:
            rows = safe_success
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("baseline_dominant_gain", 0)),
            int(row.get("safe_success_gain", 0)),
            float(row.get("score", 0.0)),
            float(row.get("success_delta", 0.0)),
            float(row.get("episode_cost_reduction", 0.0)),
            float(row.get("clean_success_delta", 0.0)),
        ),
        reverse=True,
    )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("logs/memristive_frontend_sweep"))
    parser.add_argument("--algo", choices=["td3", "epo", "rec", "qpsl", "lag", "fac"], default="epo")
    parser.add_argument("--checkpoint", type=str, default="models/paper_epo_1m_seed0_20260629")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--traffic-density", type=float, default=0.12)
    parser.add_argument("--num-lasers", type=int, default=30)
    parser.add_argument("--candidates", type=str, default="all")
    parser.add_argument("--include-risk-field", action="store_true")
    parser.add_argument("--write-trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    args.outdir.mkdir(parents=True, exist_ok=True)
    selected_names = parse_candidate_names(args.candidates)
    candidates = [candidate_by_name()[name] for name in selected_names]
    if bool(args.include_risk_field):
        candidates.insert(0, {"name": "risk_field_reference", "frontend": "risk_field", "overrides": {}})

    baseline_rows, _baseline_traces = run_mode("none", build_eval_args(args, {}))
    for row in baseline_rows:
        row["candidate"] = "none"
        row["frontend"] = "none"
    baseline_summary = summarize(baseline_rows)[0]
    flattened_baseline = flatten_summary("none", "none", baseline_summary, {})

    all_episode_rows = list(baseline_rows)
    summary_rows: List[Dict[str, Any]] = [flattened_baseline]
    for candidate in candidates:
        rows, summary_row = run_candidate(args, candidate)
        apply_deltas(summary_row, baseline_summary)
        all_episode_rows.extend(rows)
        summary_rows.append(summary_row)
        candidate_dir = args.outdir / str(candidate["name"])
        write_csv(candidate_dir / "per_episode.csv", rows)
        write_csv(candidate_dir / "aggregate.csv", [summary_row])

    scored_rows = [row for row in summary_rows if str(row["candidate"]) != "none"]
    best = pick_best(scored_rows)
    best_payload = {
        "algo": args.algo,
        "checkpoint": args.checkpoint,
        "episodes": int(args.episodes),
        "start_seed": int(args.start_seed),
        "best": best,
        "best_frontend": "nocicim_risk_field",
        "note": "baseline_dominant_gain means success>=baseline, clean_success>=baseline, and mean episode cost lower than baseline. risk_field_reference is reported only as a reference when included.",
    }
    (args.outdir / "sweep_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (args.outdir / "best_config.json").write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.outdir / "per_episode_matrix.csv", all_episode_rows)
    write_csv(args.outdir / "sweep_summary.csv", summary_rows)
    print(json.dumps(best_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
