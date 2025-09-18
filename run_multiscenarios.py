"""
Run 4 scenarios × (heuristics + TD3 + PPO) × N seeds, and aggregate mean ± 95% CI.

Fixes vs. previous:
  - Scheduling metric = Jain's fairness over per-user allocated bandwidth each step.
  - Bandwidth metric = mean per-user allocated bandwidth fraction (allocation / total_bandwidth).
  - Power metric     = mean linear power normalized to linear max (10^(P/10) / 10^(Pmax/10)).
  - Keeps episode return as Avg. Reward (↑).
"""

import os, sys, json, math, argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

# ---- your modules ----
from wireless_optim.environment import HetNetEnvironment              
from models.PPO import ppo_train                              
from models.TD3 import train_td3                              
from heuristics_baselines import (                           
    policy_pf_equal,
    policy_greedy_ofdma,
    policy_interference_pricing_power,
    _pf_update,
    _initial_pf_memory,
)

SCENARIOS = {
    "dense_urban":   dict(num_macro_bs=3, num_small_bs=10, num_users=50, max_steps=300),
    "sparse_suburb": dict(num_macro_bs=3, num_small_bs=0,  num_users=50, max_steps=300),
    "hotspot":       dict(num_macro_bs=3, num_small_bs=6,  num_users=50, max_steps=300),
    "mixed":         dict(num_macro_bs=3, num_small_bs=6,  num_users=50, max_steps=300),
}

HEURISTICS = {
    "g_ofdma": policy_greedy_ofdma,
    "ip_pc":   policy_interference_pricing_power,
    "pf_eq":   policy_pf_equal,
}

METHODS = ["g_ofdma","ip_pc","pf_eq","td3","ppo"]

DEFAULT_PPO = dict(
    num_steps=1024, num_epochs=150, lr=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, update_epochs=8,
    hidden_size=256, max_grad_norm=0.5,
)
DEFAULT_TD3 = dict(
    num_episodes=150, batch_size=256, replay_capacity=200_000,
    lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.005,
    policy_noise=0.2, noise_clip=0.5, policy_delay=2,
    warmup_steps=5000, action_noise_std=0.1,
    use_wandb=False, wandb_project="td3", wandb_name="td3",
)

def jain_fairness(x: jnp.ndarray, eps=1e-12) -> float:
    x = jnp.asarray(x)
    s1 = jnp.sum(x)
    s2 = jnp.sum(x * x) + eps
    n = x.size
    return float((s1 * s1) / (n * s2 + eps))

def make_env(scen):
    return HetNetEnvironment(**SCENARIOS[scen])

def seed_reset(env, seed):
    key = jax.random.PRNGKey(seed)
    ts = env.reset(key)
    return ts, key

def compute_step_metrics_from_alloc(env) -> dict:
    """Compute metrics purely from current allocations + power."""
    st = env._state
    # per-user total allocated bandwidth (sum across BS)
    bw_user_hz = jnp.sum(st["resource_allocations"], axis=0)  # [num_users]
    bw_frac_user = bw_user_hz / env.total_bandwidth           # fraction per user (can exceed 1 in aggregate if multi-BS)
    bandwidth_mean = float(jnp.mean(bw_frac_user))            # (↑)

    # fairness across users of allocation (Jain in [0,1]) (↑)
    fairness = jain_fairness(bw_user_hz)

    # power in linear, normalized to linear max (bits/Joule perspective) (↓)
    p_lin = 10.0 ** (st["power_levels"] / 10.0)               # mW
    p_lin_max = 10.0 ** (env.max_power / 10.0)
    power_norm = float(jnp.mean(p_lin / p_lin_max))

    return dict(bandwidth=bandwidth_mean, scheduling=fairness, power=power_norm)

def rollout_heuristic(env, mode: str, seed: int, steps: int):
    if   mode == "g_ofdma": pol = policy_greedy_ofdma
    elif mode == "ip_pc":   pol = policy_interference_pricing_power
    elif mode == "pf_eq":   pol = policy_pf_equal
    else: raise ValueError(mode)

    ts, _key = seed_reset(env, seed)
    state = env._state
    pf_mem = _initial_pf_memory(env.num_users)

    rewards, bw_hist, p_hist, sch_hist = [], [], [], []
    for _ in range(steps):
        if pol is policy_pf_equal:
            action, pf_score, sched_scores = pol(env, state, pf_mem)
        else:
            action, pf_score, sched_scores = pol(env, state)
        ts = env.step(action)
        metrics = compute_step_metrics_from_alloc(env)
        rewards.append(float(ts.reward))
        bw_hist.append(metrics["bandwidth"])
        p_hist.append(metrics["power"])
        sch_hist.append(metrics["scheduling"])
        pf_mem = _pf_update(pf_mem, jnp.ones((env.num_users,)))  # keep PF memory rolling
        state = env._state
        if ts.discount == 0.0:
            break

    return dict(
        reward=np.sum(rewards),
        bandwidth=np.mean(bw_hist),
        power=np.mean(p_hist),
        scheduling=np.mean(sch_hist),
    )

def rollout_ppo(env, ppo_net, params, steps: int, seed: int):
    ts, _key = seed_reset(env, seed)
    rewards, bw_hist, p_hist, sch_hist = [], [], [], []
    for _ in range(steps):
        obs = ts.observation
        mu, log_sigma, _ = ppo_net.apply(params, obs)  # PPO returns mu in [0,1]
        action = jnp.clip(mu, 0.0, 1.0)
        ts = env.step(action)
        m = compute_step_metrics_from_alloc(env)
        rewards.append(float(ts.reward))
        bw_hist.append(m["bandwidth"]); p_hist.append(m["power"]); sch_hist.append(m["scheduling"])
        if ts.discount == 0.0:
            break
    return dict(reward=np.sum(rewards), bandwidth=np.mean(bw_hist), power=np.mean(p_hist), scheduling=np.mean(sch_hist))

def rollout_td3(env, agent, steps: int, seed: int):
    ts, _key = seed_reset(env, seed)
    rewards, bw_hist, p_hist, sch_hist = [], [], [], []
    for _ in range(steps):
        obs = ts.observation
        action = jnp.clip(agent.actor.apply(agent.actor_params, obs), 0.0, 1.0)
        ts = env.step(action)
        m = compute_step_metrics_from_alloc(env)
        rewards.append(float(ts.reward))
        bw_hist.append(m["bandwidth"]); p_hist.append(m["power"]); sch_hist.append(m["scheduling"])
        if ts.discount == 0.0:
            break
    return dict(reward=np.sum(rewards), bandwidth=np.mean(bw_hist), power=np.mean(p_hist), scheduling=np.mean(sch_hist))

def mean_ci(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return (np.nan, np.nan)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    ci = 1.96 * s / max(1, np.sqrt(x.size))
    return m, ci

def run_one(scen, method, seed, ppo_cfg, td3_cfg, eval_steps, outdir):
    out_dir = Path(outdir) / scen / method / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(scen)

    if method in HEURISTICS:
        metrics = rollout_heuristic(env, method, seed, steps=eval_steps)
    elif method == "td3":
        agent, *_ = train_td3(env=env, **td3_cfg, seed=seed)
        metrics = rollout_td3(env, agent, steps=eval_steps, seed=seed)
    elif method == "ppo":
        params, ppo_net, *_ = ppo_train(env, ppo_cfg, seed=seed, use_wandb=False)
        metrics = rollout_ppo(env, ppo_net, params, steps=eval_steps, seed=seed)
    else:
        raise ValueError(method)

    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump({
            "avg_reward_norm": metrics["reward"],
            "bandwidth_mean_norm": metrics["bandwidth"],
            "power_mean_norm": metrics["power"],
            "scheduling_score_norm": metrics["scheduling"],
        }, f)
    print(f"[{scen}][{method}][seed={seed}] -> {out_json}")
    return out_json

def aggregate(outdir, out_csv, seeds=10):
    import pandas as pd
    rows = []
    for scen in SCENARIOS.keys():
        for m in METHODS:
            R,B,P,S = [],[],[],[]
            for sd in range(seeds):
                jf = Path(outdir) / scen / m / f"seed{sd}" / "metrics.json"
                if not jf.exists(): continue
                with open(jf) as f: d = json.load(f)
                R.append(d.get("avg_reward_norm", np.nan))
                B.append(d.get("bandwidth_mean_norm", np.nan))
                P.append(d.get("power_mean_norm", np.nan))
                S.append(d.get("scheduling_score_norm", np.nan))
            Rm,Rci = mean_ci(R); Bm,Bci = mean_ci(B); Pm,Pci = mean_ci(P); Sm,Sci = mean_ci(S)
            rows.append({
                "Scenario": scen,
                "Method": m,
                "Bandwidth (↑)": f"{Bm:.2f} ± {Bci:.2f}",
                "Power (↓)":     f"{Pm:.2f} ± {Pci:.2f}",
                "Scheduling (↑)":f"{Sm:.2f} ± {Sci:.2f}",
                "Avg. Reward (↑)": f"{Rm:.0f} ± {Rci:.0f}",
            })
    df = pd.DataFrame(rows)
    # Order
    scen_order = {s:i for i,s in enumerate(["dense_urban","sparse_suburb","hotspot","mixed"])}
    meth_order = {m:i for i,m in enumerate(["g_ofdma","ip_pc","pf_eq","td3","ppo"])}
    df = df.sort_values(["Scenario","Method"], key=lambda col: col.map(scen_order) if col.name=="Scenario" else col.map(meth_order))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[aggregate] wrote {out_csv}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()))
    ap.add_argument("--methods",   nargs="+", default=METHODS)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=300)
    ap.add_argument("--outdir", type=str, default="artifacts")
    # quick overrides
    ap.add_argument("--ppo_epochs", type=int, default=DEFAULT_PPO["num_epochs"])
    ap.add_argument("--ppo_steps",  type=int, default=DEFAULT_PPO["num_steps"])
    ap.add_argument("--td3_episodes", type=int, default=DEFAULT_TD3["num_episodes"])
    args = ap.parse_args()

    ppo_cfg = DEFAULT_PPO.copy(); ppo_cfg.update(num_epochs=args.ppo_epochs, num_steps=args.ppo_steps)
    td3_cfg = DEFAULT_TD3.copy(); td3_cfg.update(num_episodes=args.td3_episodes)

    for scen in args.scenarios:
        for m in args.methods:
            for sd in range(args.seeds):
                run_one(scen, m, sd, ppo_cfg, td3_cfg, args.eval_steps, args.outdir)

    aggregate(args.outdir, out_csv=str(Path(args.outdir) / "table_multiscenario.csv"), seeds=args.seeds)

if __name__ == "__main__":
    main()
