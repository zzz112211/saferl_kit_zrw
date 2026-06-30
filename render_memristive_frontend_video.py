"""Render a saferl_kit policy with a memristive risk frontend."""

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np
import pygame

from eval_memristive_frontend import (
    EpisodeMetrics,
    MemristiveRiskReflex,
    ReflexCfg,
    build_policy,
    make_env,
    select_policy_action,
)


def surface_to_bgr(surface: Any) -> np.ndarray:
    rgb = pygame.surfarray.array3d(surface)
    rgb = np.transpose(rgb, (1, 0, 2))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def top_down_frame(env: Any, width: int, height: int) -> np.ndarray:
    surface = env.render(
        mode="top_down",
        film_size=(int(width), int(height)),
        screen_size=(int(width), int(height)),
        track_target_vehicle=True,
        num_stack=30,
        history_smooth=0,
        road_color=(35, 35, 35),
    )
    frame = surface_to_bgr(surface)
    if frame.shape[1] != int(width) or frame.shape[0] != int(height):
        frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return frame


def annotate(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    out = frame.copy()
    x, y = 14, 26
    for line in lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)
        y += 22
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, default=None)
    parser.add_argument("--algo", default="td3")
    parser.add_argument("--checkpoint", default="models/paper_td3_1m_seed0_20260629")
    parser.add_argument("--frontend", choices=["none", "risk_field", "nocicim_risk_field"], default="nocicim_risk_field")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--traffic-density", type=float, default=0.12)
    parser.add_argument("--num-lasers", type=int, default=30)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--frame-width", type=int, default=720)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_output = Path(args.trace_output) if args.trace_output else args.output.with_suffix(".trace.csv")

    env = make_env(int(args.seed), args)
    policy = build_policy(str(args.algo), str(args.checkpoint), env)
    reflex = MemristiveRiskReflex(str(args.frontend), ReflexCfg(num_lasers=int(args.num_lasers)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        float(args.fps),
        (int(args.frame_width), int(args.frame_height)),
    )
    if not writer.isOpened():
        env.close()
        raise RuntimeError("could not open video writer for {}".format(args.output))

    state = env.reset()
    done = False
    metrics = EpisodeMetrics()
    trace_rows: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}

    try:
        while not done and metrics.steps < int(args.horizon):
            nominal_action = select_policy_action(policy, str(args.algo), state)
            safe_action, _risk, debug = reflex.apply(nominal_action, state)
            next_state, reward, done, info = env.step(safe_action)
            metrics.update(float(reward), dict(info), int(debug["intervention"]), float(debug["action_distortion"]))

            frame = top_down_frame(env, int(args.frame_width), int(args.frame_height))
            if not bool(args.no_overlay):
                frame = annotate(
                    frame,
                    [
                        "{}+{} seed={} step={}".format(args.algo, args.frontend, args.seed, metrics.steps),
                        "cost={:.1f} interv={:.2f} distort={:.3f}".format(
                            metrics.episode_cost,
                            metrics.interventions / max(metrics.steps, 1),
                            metrics.action_distortion_sum / max(metrics.steps, 1),
                        ),
                        "front={:.2f} lateral={:.2f} xF={:.2f} xL={:.2f}".format(
                            float(debug.get("risk_front_risk", 0.0)),
                            float(debug.get("risk_lateral_risk", 0.0)),
                            float(debug.get("nocicim_x_front", 0.0)),
                            float(debug.get("nocicim_x_lateral", 0.0)),
                        ),
                        "nom=({:.2f},{:.2f}) safe=({:.2f},{:.2f})".format(
                            float(debug.get("nominal_steering", 0.0)),
                            float(debug.get("nominal_throttle_brake", 0.0)),
                            float(debug.get("safe_steering", 0.0)),
                            float(debug.get("safe_throttle_brake", 0.0)),
                        ),
                    ],
                )
            writer.write(frame)

            row: Dict[str, Any] = {
                "mode": "{}+{}".format(args.algo, args.frontend),
                "seed": int(args.seed),
                "step": int(metrics.steps),
                "reward": float(reward),
                "done": int(bool(done)),
                "cost": float(info.get("cost", 0.0)),
                "episode_cost": float(metrics.episode_cost),
                "arrive_dest": int(bool(info.get("arrive_dest", False))),
                "out_of_road": int(bool(info.get("out_of_road", False))),
                "crash": int(bool(info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False))),
            }
            row.update(debug)
            trace_rows.append(row)
            state = next_state
    finally:
        writer.release()
        env.close()

    write_csv(trace_output, trace_rows)
    row = metrics.to_row("{}+{}".format(args.algo, args.frontend), int(args.seed))
    print(
        "video_written",
        str(args.output),
        "trace",
        str(trace_output),
        "steps",
        row["steps"],
        "success",
        row["success"],
        "clean_success",
        row["clean_success"],
        "cost",
        row["episode_cost"],
        "crash",
        row["any_crash"],
        "out_of_road",
        row["out_of_road"],
        "intervention_ratio",
        round(float(row["intervention_ratio"]), 4),
    )


if __name__ == "__main__":
    main()
