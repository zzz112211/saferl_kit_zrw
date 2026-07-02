"""Evaluate saferl_kit MetaDrive checkpoints with a memristive risk reflex.

This script intentionally runs inside saferl_kit's original Python 3.7 runtime
and bundled MetaDrive v0.2.4 package. The SL4AD checkpoints were trained there;
evaluating them in a newer MetaDrive runtime changes the closed-loop behavior.
"""

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

import saferl_algos
from metadrive import SafeMetaDriveEnv


SUPPORTED_ALGOS = ("td3", "epo", "rec", "qpsl", "lag", "fac")


@dataclass
class ReflexCfg:
    front_distance_threshold_m: float = 8.0
    stop_distance_m: float = 3.0
    lateral_distance_threshold_m: float = 5.0
    lidar_range_m: float = 50.0
    num_lasers: int = 30
    front_cone_width: int = 7
    side_cone_width: int = 3
    slow_front_risk_threshold: float = 0.55
    stop_front_risk_threshold: float = 0.85
    stop_brake_value: float = -1.0
    slow_throttle_cap: float = 0.0
    lateral_steer_threshold: float = 0.55
    lateral_throttle_threshold: float = 0.65
    lateral_throttle_cap: float = 0.25
    lateral_steering_cap: float = 0.65
    front_decay: float = 0.82
    front_gain: float = 0.12
    front_risk_soft: float = 0.15
    front_risk_saturate: float = 0.55
    front_slow_threshold: float = 0.70
    front_slow_throttle_cap: float = 0.75
    lateral_decay: float = 0.82
    lateral_gain: float = 0.32
    lateral_risk_soft: float = 0.10
    lateral_risk_saturate: float = 0.55
    lateral_guard_threshold: float = 0.55
    allow_direct_stop: bool = True
    require_memory_for_stop: bool = False
    min_front_risk_for_stop: float = 0.0
    require_risk_for_lateral_guard: bool = False
    min_lateral_risk_for_guard: float = 0.0


@dataclass
class Risk:
    front_risk: float = 0.0
    lateral_risk: float = 0.0
    left_boundary_risk: float = 0.0
    right_boundary_risk: float = 0.0
    stop_required: bool = False
    safe_throttle_brake: float = 1.0


@dataclass
class EpisodeMetrics:
    reward: float = 0.0
    episode_cost: float = 0.0
    steps: int = 0
    arrive_dest: int = 0
    any_crash: int = 0
    out_of_road: int = 0
    out_of_time: int = 0
    interventions: int = 0
    action_distortion_sum: float = 0.0

    def update(self, reward: float, info: Dict[str, Any], intervention: int, distortion: float) -> None:
        self.steps += 1
        self.reward += float(reward)
        self.episode_cost += float(info.get("cost", 0.0))
        self.arrive_dest = max(self.arrive_dest, int(bool(info.get("arrive_dest", False))))
        self.any_crash = max(
            self.any_crash,
            int(bool(info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False))),
        )
        self.out_of_road = max(self.out_of_road, int(bool(info.get("out_of_road", False))))
        self.out_of_time = max(self.out_of_time, int(bool(info.get("max_step", False) or info.get("out_of_time", False))))
        self.interventions += int(intervention)
        self.action_distortion_sum += float(distortion)

    def to_row(self, mode: str, seed: int) -> Dict[str, Any]:
        clean_success = int(
            bool(self.arrive_dest)
            and float(self.episode_cost) == 0.0
            and int(self.any_crash) == 0
            and int(self.out_of_road) == 0
        )
        return {
            "mode": str(mode),
            "seed": int(seed),
            "steps": int(self.steps),
            "success": int(self.arrive_dest),
            "clean_success": int(clean_success),
            "reward": float(self.reward),
            "episode_cost": float(self.episode_cost),
            "cost_rate": float(self.episode_cost / max(self.steps, 1)),
            "any_crash": int(self.any_crash),
            "out_of_road": int(self.out_of_road),
            "out_of_time": int(self.out_of_time),
            "intervention_ratio": float(self.interventions / max(self.steps, 1)),
            "mean_action_distortion": float(self.action_distortion_sum / max(self.steps, 1)),
        }


def clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def clip_action(action: Any) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError("MetaDrive action must have at least two elements")
    return np.clip(arr[:2], -1.0, 1.0).astype(np.float32)


def front_lidar_indices(num_lasers: int, width: int) -> np.ndarray:
    width = int(np.clip(int(width), 1, int(num_lasers)))
    forward_count = (width + 1) // 2
    backward_count = width - forward_count
    return np.concatenate(
        [
            np.arange(forward_count, dtype=np.int64),
            np.arange(int(num_lasers) - backward_count, int(num_lasers), dtype=np.int64),
        ]
    )


def side_lidar_indices(num_lasers: int, center: int, width: int) -> np.ndarray:
    width = int(np.clip(int(width), 1, int(num_lasers)))
    half = width // 2
    offsets = np.arange(-half, width - half, dtype=np.int64)
    return (int(center) + offsets) % int(num_lasers)


def tail_lidar(obs: Any, num_lasers: int) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size < int(num_lasers):
        return np.ones(int(num_lasers), dtype=np.float32)
    return np.clip(arr[-int(num_lasers) :], 0.0, 1.0)


def estimate_risk(obs: Any, cfg: ReflexCfg) -> Risk:
    lidar = tail_lidar(obs, cfg.num_lasers)
    front = lidar[front_lidar_indices(cfg.num_lasers, cfg.front_cone_width)]
    left = lidar[side_lidar_indices(cfg.num_lasers, cfg.num_lasers // 4, cfg.side_cone_width)]
    right = lidar[side_lidar_indices(cfg.num_lasers, 3 * cfg.num_lasers // 4, cfg.side_cone_width)]
    front_distance = float(np.min(front) * cfg.lidar_range_m) if front.size else float("inf")
    left_distance = float(np.min(left) * cfg.lidar_range_m) if left.size else float("inf")
    right_distance = float(np.min(right) * cfg.lidar_range_m) if right.size else float("inf")
    front_risk = clip01((cfg.front_distance_threshold_m - front_distance) / max(cfg.front_distance_threshold_m, 1e-6))
    left_risk = clip01((cfg.lateral_distance_threshold_m - left_distance) / max(cfg.lateral_distance_threshold_m, 1e-6))
    right_risk = clip01((cfg.lateral_distance_threshold_m - right_distance) / max(cfg.lateral_distance_threshold_m, 1e-6))
    stop_required = bool(front_distance <= cfg.stop_distance_m)
    return Risk(
        front_risk=front_risk,
        lateral_risk=max(left_risk, right_risk),
        left_boundary_risk=left_risk,
        right_boundary_risk=right_risk,
        stop_required=stop_required,
        safe_throttle_brake=-1.0 if stop_required else 0.0,
    )


def normalized_stimulus(value: float, soft: float, saturate: float) -> float:
    if float(saturate) <= float(soft):
        return clip01(value)
    return clip01((float(value) - float(soft)) / (float(saturate) - float(soft)))


class MemristiveRiskReflex:
    def __init__(self, mode: str, cfg: Optional[ReflexCfg] = None) -> None:
        if mode not in {"none", "risk_field", "nocicim_risk_field"}:
            raise ValueError("unsupported frontend mode: {}".format(mode))
        self.mode = str(mode)
        self.cfg = cfg or ReflexCfg()
        self.x_front = 0.0
        self.x_lateral = 0.0

    def _front_stop_allowed(self, risk: Risk) -> bool:
        if not bool(self.cfg.allow_direct_stop):
            return False
        if float(risk.front_risk) < float(self.cfg.min_front_risk_for_stop):
            return False
        if bool(self.cfg.require_memory_for_stop) and self.x_front < float(self.cfg.front_slow_threshold):
            return False
        return True

    def _lateral_guard_allowed(self, risk: Risk) -> bool:
        if not bool(self.cfg.require_risk_for_lateral_guard):
            return True
        if float(risk.lateral_risk) >= float(self.cfg.min_lateral_risk_for_guard):
            return True
        return self.x_lateral >= float(self.cfg.lateral_guard_threshold)

    def apply(self, nominal_action: Any, obs: Any) -> Tuple[np.ndarray, Risk, Dict[str, Any]]:
        nominal = clip_action(nominal_action)
        risk = estimate_risk(obs, self.cfg) if self.mode != "none" else Risk()
        safe = np.array(nominal, dtype=np.float32)
        debug: Dict[str, Any] = {
            "risk_front_risk": float(risk.front_risk),
            "risk_lateral_risk": float(risk.lateral_risk),
            "risk_stop_required": int(bool(risk.stop_required)),
            "nocicim_x_front": float(self.x_front),
            "nocicim_x_lateral": float(self.x_lateral),
            "nocicim_front_slow_active": 0,
            "nocicim_lateral_guard_active": 0,
        }
        if self.mode == "none":
            return safe, risk, self._finish_debug(nominal, safe, debug)

        if (
            (bool(risk.stop_required) or float(risk.front_risk) >= float(self.cfg.stop_front_risk_threshold))
            and self._front_stop_allowed(risk)
        ):
            safe[1] = min(float(safe[1]), float(self.cfg.stop_brake_value), float(risk.safe_throttle_brake))
        elif float(risk.front_risk) >= float(self.cfg.slow_front_risk_threshold):
            safe[1] = min(float(safe[1]), float(self.cfg.slow_throttle_cap))

        high_speed_lateral = (
            abs(float(nominal[0])) >= float(self.cfg.lateral_steer_threshold)
            and float(nominal[1]) >= float(self.cfg.lateral_throttle_threshold)
        )
        if high_speed_lateral and self._lateral_guard_allowed(risk):
            safe[1] = min(float(safe[1]), float(self.cfg.lateral_throttle_cap))
            safe[0] = float(np.clip(float(safe[0]), -float(self.cfg.lateral_steering_cap), float(self.cfg.lateral_steering_cap)))

        if self.mode == "nocicim_risk_field":
            front_stimulus = normalized_stimulus(risk.front_risk, self.cfg.front_risk_soft, self.cfg.front_risk_saturate)
            lateral_stimulus = normalized_stimulus(
                risk.lateral_risk, self.cfg.lateral_risk_soft, self.cfg.lateral_risk_saturate
            )
            self.x_front = clip01(float(self.cfg.front_decay) * self.x_front + float(self.cfg.front_gain) * front_stimulus)
            self.x_lateral = clip01(
                float(self.cfg.lateral_decay) * self.x_lateral + float(self.cfg.lateral_gain) * lateral_stimulus
            )
            debug["nocicim_x_front"] = float(self.x_front)
            debug["nocicim_x_lateral"] = float(self.x_lateral)
            debug["nocicim_front_stimulus"] = float(front_stimulus)
            debug["nocicim_lateral_stimulus"] = float(lateral_stimulus)
            if self.x_front >= float(self.cfg.front_slow_threshold):
                safe[1] = min(float(safe[1]), float(self.cfg.front_slow_throttle_cap))
                debug["nocicim_front_slow_active"] = 1
            if self.x_lateral >= float(self.cfg.lateral_guard_threshold):
                safe[1] = min(float(safe[1]), float(self.cfg.lateral_throttle_cap))
                safe[0] = float(
                    np.clip(float(safe[0]), -float(self.cfg.lateral_steering_cap), float(self.cfg.lateral_steering_cap))
                )
                debug["nocicim_lateral_guard_active"] = 1

        safe = clip_action(safe)
        return safe, risk, self._finish_debug(nominal, safe, debug)

    def _finish_debug(self, nominal: np.ndarray, safe: np.ndarray, debug: Dict[str, Any]) -> Dict[str, Any]:
        distortion = float(np.linalg.norm(safe - nominal, ord=2))
        debug.update(
            {
                "nominal_steering": float(nominal[0]),
                "nominal_throttle_brake": float(nominal[1]),
                "safe_steering": float(safe[0]),
                "safe_throttle_brake": float(safe[1]),
                "intervention": int(distortion > 1e-6),
                "action_distortion": float(distortion),
            }
        )
        return debug


def required_suffixes(algo: str) -> Tuple[str, ...]:
    algo = str(algo).lower()
    if algo == "td3":
        return ("_actor", "_actor_optimizer", "_critic", "_critic_optimizer")
    if algo in {"epo", "lag", "qpsl"}:
        return ("_actor", "_actor_optimizer", "_critic", "_critic_optimizer", "_C_critic", "_C_critic_optimizer")
    if algo == "rec":
        return (
            "_actor",
            "_actor_optimizer",
            "_critic",
            "_critic_optimizer",
            "_C_critic",
            "_C_critic_optimizer",
            "_C_actor",
            "_C_actor_optimizer",
        )
    if algo == "fac":
        return (
            "_actor",
            "_actor_optimizer",
            "_critic",
            "_critic_optimizer",
            "_C_critic",
            "_C_critic_optimizer",
            "_lam_net",
            "_lam_net_optimizer",
        )
    raise ValueError("unsupported algo: {}".format(algo))


def assert_checkpoint_complete(prefix: str, algo: str) -> None:
    missing = [str(prefix) + suffix for suffix in required_suffixes(algo) if not Path(str(prefix) + suffix).exists()]
    if missing:
        raise FileNotFoundError("missing checkpoint files: " + ", ".join(missing))


def build_policy(algo: str, checkpoint_prefix: str, env: Any) -> Any:
    algo = str(algo).lower()
    assert_checkpoint_complete(checkpoint_prefix, algo)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "rew_discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2 * max_action,
        "noise_clip": 0.5 * max_action,
        "policy_freq": 2,
    }
    if algo == "td3":
        policy = saferl_algos.unconstrained.TD3(**kwargs)
    elif algo == "epo":
        policy = saferl_algos.exactpenalty.TD3epo(**dict(kwargs, delta=0.1, cost_discount=0.99, kappa=5))
    elif algo == "rec":
        policy = saferl_algos.recovery.TD3Recovery(**dict(kwargs, delta=0.1, cost_discount=0.99))
    elif algo == "qpsl":
        policy = saferl_algos.safetylayer.TD3Qpsl(**dict(kwargs, delta=0.1, cost_discount=0.99))
    elif algo == "lag":
        policy = saferl_algos.lagrangian.TD3Lag(**dict(kwargs, delta=0.1, lam_init=0.0, lam_lr=0.001, cost_discount=0.99))
    elif algo == "fac":
        policy = saferl_algos.fac.TD3Fac(**dict(kwargs, delta=0.1, cost_discount=0.99))
    else:
        raise ValueError("unsupported algo: {}".format(algo))
    policy.load(str(checkpoint_prefix))
    return policy


def select_policy_action(policy: Any, algo: str, state: Any) -> np.ndarray:
    arr = np.asarray(state)
    if str(algo).lower() == "qpsl":
        return clip_action(policy.select_action(arr, use_qpsl=True))
    if str(algo).lower() == "rec":
        action, _raw_action = policy.select_action(arr, recovery=True)
        return clip_action(action)
    return clip_action(policy.select_action(arr))


def make_env(seed: int, args: argparse.Namespace) -> Any:
    config = {
        "environment_num": 1,
        "start_seed": int(seed),
        "cost_to_reward": True,
        "traffic_density": float(args.traffic_density),
        "vehicle_config": {"lidar": {"num_lasers": int(args.num_lasers)}},
    }
    return SafeMetaDriveEnv(config=config)


def run_mode(frontend: str, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reflex_cfg = build_reflex_cfg(args)
    mode = "{}+{}".format(args.algo, frontend)
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    policy = None
    for seed in range(int(args.start_seed), int(args.start_seed) + int(args.episodes)):
        env = make_env(seed, args)
        try:
            if policy is None:
                policy = build_policy(args.algo, str(args.checkpoint), env)
            reflex = MemristiveRiskReflex(frontend, reflex_cfg)
            state = env.reset()
            done = False
            metrics = EpisodeMetrics()
            info: Dict[str, Any] = {}
            while not done and metrics.steps < int(args.horizon):
                nominal_action = select_policy_action(policy, args.algo, state)
                safe_action, _risk, debug = reflex.apply(nominal_action, state)
                next_state, reward, done, info = env.step(safe_action)
                metrics.update(float(reward), dict(info), int(debug["intervention"]), float(debug["action_distortion"]))
                if bool(args.write_trace):
                    trace_row = {
                        "mode": mode,
                        "seed": int(seed),
                        "step": int(metrics.steps),
                        "reward": float(reward),
                        "done": int(bool(done)),
                        "cost": float(info.get("cost", 0.0)),
                        "arrive_dest": int(bool(info.get("arrive_dest", False))),
                        "out_of_road": int(bool(info.get("out_of_road", False))),
                        "crash": int(bool(info.get("crash", False))),
                    }
                    trace_row.update(debug)
                    traces.append(trace_row)
                state = next_state
            rows.append(metrics.to_row(mode, seed))
        finally:
            env.close()
    return rows, traces


def mean(rows: List[Dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row.get(key, 0.0)) for row in rows])) if rows else float("nan")


def summarize(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(rows)
    out: List[Dict[str, Any]] = []
    for mode in sorted({str(row["mode"]) for row in rows}):
        subset = [row for row in rows if str(row["mode"]) == mode]
        out.append(
            {
                "mode": mode,
                "episodes": len(subset),
                "success_rate": mean(subset, "success"),
                "clean_success_rate": mean(subset, "clean_success"),
                "mean_reward": mean(subset, "reward"),
                "mean_episode_cost": mean(subset, "episode_cost"),
                "mean_cost_rate": mean(subset, "cost_rate"),
                "crash_rate": mean(subset, "any_crash"),
                "out_of_road_rate": mean(subset, "out_of_road"),
                "out_of_time_rate": mean(subset, "out_of_time"),
                "intervention_ratio": mean(subset, "intervention_ratio"),
                "mean_action_distortion": mean(subset, "mean_action_distortion"),
            }
        )
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


def parse_frontends(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected boolean value, got {}".format(value))


def add_reflex_cfg_args(parser: argparse.ArgumentParser) -> None:
    defaults = ReflexCfg()
    for field_name, default_value in asdict(defaults).items():
        if field_name == "num_lasers":
            continue
        value_type = parse_bool if isinstance(default_value, bool) else type(default_value)
        parser.add_argument("--{}".format(field_name.replace("_", "-")), type=value_type, default=default_value)


def build_reflex_cfg(args: argparse.Namespace) -> ReflexCfg:
    values = {}
    for field_name, default_value in asdict(ReflexCfg()).items():
        if field_name == "num_lasers":
            values[field_name] = int(getattr(args, "num_lasers", default_value))
        else:
            values[field_name] = getattr(args, field_name, default_value)
    return ReflexCfg(**values)


def run_self_test() -> None:
    cfg = ReflexCfg(num_lasers=30)
    clear_obs = np.ones(49, dtype=np.float32)
    risky_obs = np.ones(49, dtype=np.float32)
    risky_obs[-30] = 0.01
    assert estimate_risk(clear_obs, cfg).front_risk == 0.0
    assert estimate_risk(risky_obs, cfg).front_risk > 0.9
    reflex = MemristiveRiskReflex("nocicim_risk_field", cfg)
    safe, _risk, debug = reflex.apply(np.array([0.9, 0.9], dtype=np.float32), risky_obs)
    assert safe[1] <= 0.0
    assert debug["intervention"] == 1
    gated_cfg = ReflexCfg(
        num_lasers=30,
        require_risk_for_lateral_guard=True,
        min_lateral_risk_for_guard=0.2,
        allow_direct_stop=False,
        stop_distance_m=3.0,
    )
    gated_reflex = MemristiveRiskReflex("nocicim_risk_field", gated_cfg)
    clear_safe, _risk, clear_debug = gated_reflex.apply(np.array([0.99, 1.0], dtype=np.float32), clear_obs)
    assert clear_safe[1] == 1.0
    assert clear_debug["intervention"] == 0
    stop_obs = np.ones(49, dtype=np.float32)
    stop_obs[-30] = 0.01
    no_stop_reflex = MemristiveRiskReflex("nocicim_risk_field", gated_cfg)
    no_stop_safe, _risk, _debug = no_stop_reflex.apply(np.array([0.0, 1.0], dtype=np.float32), stop_obs)
    assert no_stop_safe[1] > -1.0
    front_only_cfg = ReflexCfg(
        num_lasers=30,
        allow_direct_stop=False,
        slow_front_risk_threshold=1.01,
        stop_front_risk_threshold=1.01,
        require_risk_for_lateral_guard=True,
        min_lateral_risk_for_guard=0.2,
    )
    front_only_reflex = MemristiveRiskReflex("nocicim_risk_field", front_only_cfg)
    front_only_safe, _risk, front_only_debug = front_only_reflex.apply(
        np.array([0.99, 1.0], dtype=np.float32), risky_obs
    )
    assert front_only_safe[1] == 1.0
    assert front_only_debug["intervention"] == 0
    assert required_suffixes("td3") == ("_actor", "_actor_optimizer", "_critic", "_critic_optimizer")
    out = Path("logs") / "_memristive_self_test.csv"
    write_csv(out, [{"a": 1}, {"a": 2, "b": 3}])
    assert out.read_text(encoding="utf-8").splitlines() == ["a,b", "1,", "2,3"]
    out.unlink()
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("logs/memristive_frontend_eval"))
    parser.add_argument("--algo", choices=list(SUPPORTED_ALGOS), default="td3")
    parser.add_argument("--checkpoint", type=str, default="models/paper_td3_1m_seed0_20260629")
    parser.add_argument("--frontends", type=str, default="none,risk_field,nocicim_risk_field")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--traffic-density", type=float, default=0.12)
    parser.add_argument("--num-lasers", type=int, default=30)
    parser.add_argument("--write-trace", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    add_reflex_cfg_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    torch.set_num_threads(1)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (args.outdir / "reflex_config.json").write_text(
        json.dumps(asdict(build_reflex_cfg(args)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    all_rows: List[Dict[str, Any]] = []
    all_traces: List[Dict[str, Any]] = []
    for frontend in parse_frontends(args.frontends):
        rows, traces = run_mode(frontend, args)
        all_rows.extend(rows)
        all_traces.extend(traces)
        write_csv(args.outdir / "{}__{}".format(args.algo, frontend) / "per_episode.csv", rows)
        write_csv(args.outdir / "{}__{}".format(args.algo, frontend) / "aggregate.csv", summarize(rows))
    write_csv(args.outdir / "per_episode_matrix.csv", all_rows)
    write_csv(args.outdir / "aggregate_matrix.csv", summarize(all_rows))
    if bool(args.write_trace):
        write_csv(args.outdir / "step_trace.csv", all_traces)


if __name__ == "__main__":
    main()
