import argparse
import csv
import os

import numpy as np
import torch

import saferl_algos
from metadrive import SafeMetaDriveEnv
from risk_reflex.configs import RiskReflexConfig
from risk_reflex.wrappers import RiskReflexController


def build_eval_env(seed):
    return SafeMetaDriveEnv(
        config=dict(
            environment_num=20,
            start_seed=seed + 100,
            cost_to_reward=True,
            traffic_density=0.12,
            vehicle_config=dict(lidar=dict(num_lasers=30)),
        )
    )


def build_policy(env, args):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy = saferl_algos.safetylayer.TD3Qpsl(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        rew_discount=args.rew_discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        cost_discount=args.cost_discount,
        delta=args.delta,
    )
    policy.load(os.path.join("./models", args.model_tag))
    return policy


def build_controller(args):
    if args.variant == "baseline":
        return None
    if args.variant not in {"longitudinal", "steering_limit"}:
        raise ValueError(f"unsupported variant: {args.variant}")

    config = RiskReflexConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        theta=args.theta,
        k=args.k,
        lambda_brake=args.lambda_brake,
        lambda_throttle=args.lambda_throttle,
        enable_steering_limit=args.variant == "steering_limit",
        steering_limit_gain=args.steering_limit_gain,
        min_steering_scale=args.min_steering_scale,
        trigger_threshold=args.trigger_threshold,
    )
    return RiskReflexController(config)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float_for_name(value):
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def run_signature(args):
    parts = [
        ("ep", args.eval_episodes),
        ("a", args.alpha),
        ("b", args.beta),
        ("g", args.gamma),
        ("t", args.theta),
        ("k", args.k),
        ("lb", args.lambda_brake),
        ("lt", args.lambda_throttle),
        ("sg", args.steering_limit_gain),
        ("ms", args.min_steering_scale),
        ("tt", args.trigger_threshold),
    ]
    return "_".join(f"{name}{format_float_for_name(value)}" for name, value in parts)


def evaluate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_eval_env(args.seed)
    policy = build_policy(env, args)

    episode_rows = []
    total_success = 0.0
    total_cost = 0.0
    total_cost_events = 0
    total_steps = 0
    total_pain = 0.0
    total_triggers = 0.0
    total_longitudinal_delta = 0.0
    total_steering_limited = 0.0

    try:
        for episode in range(args.eval_episodes):
            state = env.reset()
            done = False
            steps = 0
            episode_cost = 0.0
            episode_cost_events = 0
            episode_success = 0.0
            episode_pain = 0.0
            episode_triggers = 0.0
            episode_longitudinal_delta = 0.0
            episode_steering_limited = 0.0
            controller = build_controller(args)

            while not (done or steps >= env._max_episode_steps):
                action = policy.select_action(np.array(state), use_qpsl=True)
                if controller is not None:
                    action, diag = controller.apply(state, action)
                    episode_pain += float(diag["pain"])
                    episode_triggers += float(diag["reflex_active"])
                    episode_longitudinal_delta += float(diag["longitudinal_delta"])
                    episode_steering_limited += float(diag["steering_limited"])

                state, _reward, done, info = env.step(action)
                cost = float(info.get("cost", 0.0))
                if cost != 0.0:
                    episode_cost_events += 1
                episode_cost += cost
                episode_success = 1.0 if info.get("arrive_dest", False) else 0.0
                steps += 1

            total_success += episode_success
            total_cost += episode_cost
            total_cost_events += episode_cost_events
            total_steps += steps
            total_pain += episode_pain
            total_triggers += episode_triggers
            total_longitudinal_delta += episode_longitudinal_delta
            total_steering_limited += episode_steering_limited

            episode_rows.append(
                {
                    "episode": episode,
                    "variant": args.variant,
                    "seed": args.seed,
                    "model_tag": args.model_tag,
                    "steps": steps,
                    "arrive_dest": episode_success,
                    "EpCost": episode_cost,
                    "cost_events": episode_cost_events,
                    "CostRate": episode_cost_events / steps if steps else 0.0,
                    "mean_pain": episode_pain / steps if steps else 0.0,
                    "trigger_rate": episode_triggers / steps if steps else 0.0,
                    "mean_longitudinal_delta": episode_longitudinal_delta / steps if steps else 0.0,
                    "steering_limit_rate": episode_steering_limited / steps if steps else 0.0,
                }
            )
    finally:
        env.close()

    denominator = float(args.eval_episodes) if args.eval_episodes else 1.0
    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "model_tag": args.model_tag,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "theta": args.theta,
        "k": args.k,
        "lambda_brake": args.lambda_brake,
        "lambda_throttle": args.lambda_throttle,
        "steering_limit_gain": args.steering_limit_gain,
        "min_steering_scale": args.min_steering_scale,
        "trigger_threshold": args.trigger_threshold,
        "eval_episodes": args.eval_episodes,
        "SuccessRate": total_success / denominator,
        "EpCost": total_cost / denominator,
        "CostRate": total_cost_events / total_steps if total_steps else 0.0,
        "mean_pain": total_pain / total_steps if total_steps else 0.0,
        "trigger_rate": total_triggers / total_steps if total_steps else 0.0,
        "mean_longitudinal_delta": total_longitudinal_delta / total_steps if total_steps else 0.0,
        "steering_limit_rate": total_steering_limited / total_steps if total_steps else 0.0,
    }

    stem = f"{args.model_tag}_{args.variant}_seed{args.seed}_{run_signature(args)}"
    episode_path = os.path.join(args.out_dir, f"{stem}_episodes.csv")
    summary_path = os.path.join(args.out_dir, f"{stem}_summary.csv")
    episode_fields = [
        "episode",
        "variant",
        "seed",
        "model_tag",
        "steps",
        "arrive_dest",
        "EpCost",
        "cost_events",
        "CostRate",
        "mean_pain",
        "trigger_rate",
        "mean_longitudinal_delta",
        "steering_limit_rate",
    ]
    summary_fields = [
        "variant",
        "seed",
        "model_tag",
        "alpha",
        "beta",
        "gamma",
        "theta",
        "k",
        "lambda_brake",
        "lambda_throttle",
        "steering_limit_gain",
        "min_steering_scale",
        "trigger_threshold",
        "eval_episodes",
        "SuccessRate",
        "EpCost",
        "CostRate",
        "mean_pain",
        "trigger_rate",
        "mean_longitudinal_delta",
        "steering_limit_rate",
    ]
    write_csv(episode_path, episode_fields, episode_rows)
    write_csv(summary_path, summary_fields, [summary])
    print(summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--variant", choices=["baseline", "longitudinal", "steering_limit"], required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_episodes", default=20, type=int)
    parser.add_argument("--out_dir", default="runs/risk_reflex_eval_smoke")

    parser.add_argument("--alpha", default=0.92, type=float)
    parser.add_argument("--beta", default=0.45, type=float)
    parser.add_argument("--gamma", default=0.08, type=float)
    parser.add_argument("--theta", default=0.45, type=float)
    parser.add_argument("--k", default=12.0, type=float)
    parser.add_argument("--lambda_brake", default=0.35, type=float)
    parser.add_argument("--lambda_throttle", default=0.35, type=float)
    parser.add_argument("--steering_limit_gain", default=0.5, type=float)
    parser.add_argument("--min_steering_scale", default=0.35, type=float)
    parser.add_argument("--trigger_threshold", default=0.5, type=float)

    parser.add_argument("--delta", default=0.1, type=float)
    parser.add_argument("--cost_discount", default=0.99, type=float)
    parser.add_argument("--rew_discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
