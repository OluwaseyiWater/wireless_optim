import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import os

# Create directory for plots
os.makedirs("plots", exist_ok=True)

# -------------------------
# Environment Import
# -------------------------
from wireless_optim.environment import HetNetEnvironment


def plot_terrain(env: HetNetEnvironment, 
                 states: List[jnp.ndarray], 
                 actions: List[jnp.ndarray],
                 episode: int = 0):
    """Visualize agent decisions on network terrain"""
    plt.figure(figsize=(12, 8))
    
    # Use environment's method to get macro BS positions.
    macro_positions = env._grid_positions(env.num_macro_bs, scale=1.0)
    plt.scatter(macro_positions[:, 0], macro_positions[:, 1], 
                c='red', s=100, label='Macro BS')
    # For small BS, generate dummy positions (or use attributes if available).
    key = jax.random.PRNGKey(1)
    small_bs = jax.random.uniform(key, (env.num_small_bs, 2), minval=0.0, maxval=1.0)
    plt.scatter(small_bs[:, 0], small_bs[:, 1], 
                c='blue', s=50, label='Small BS')
    
    # Extract user positions from the last state's observation.
    # This assumes the observation is structured as in env._get_observations().
    total_bs = env.num_macro_bs + env.num_small_bs
    obs = states[-1]
    # For illustration, assume the last 2*(total_bs+env.num_users) entries are positions.
    positions = obs[-2*(total_bs+env.num_users):].reshape(-1, 2)
    user_locs = positions[-env.num_users:]
    plt.scatter(user_locs[:, 0], user_locs[:, 1], 
                c='green', s=10, alpha=0.5, label='Users')
    
    # Plot actions as vectors for first 10 steps.
    for t, action in enumerate(actions[:10]):
        for bs_idx in range(action.shape[0]):
            action_type = jnp.argmax(action[bs_idx])
            color = 'purple' if action_type == 0 else 'orange' if action_type == 1 else 'cyan'
            if bs_idx < env.num_macro_bs:
                bs_loc = macro_positions[bs_idx]
            else:
                bs_loc = small_bs[bs_idx - env.num_macro_bs]
            plt.arrow(bs_loc[0], bs_loc[1], 
                      0.1 * (t+1), 0.1 * (t+1),
                      color=color, width=0.01, alpha=0.3)
    
    plt.title(f"Network State and Agent Actions (Episode {episode})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/terrain_episode_{episode}.png')
    plt.close()


def plot_action_distribution(ppo_actions: List[jnp.ndarray], 
                             d3qn_actions: List[jnp.ndarray]):
    """Compare action choices between agents"""
    plt.figure(figsize=(10, 6))
    
    ppo_flat = jnp.concatenate([jnp.argmax(a, axis=-1) for a in ppo_actions])
    d3qn_flat = jnp.concatenate([jnp.argmax(a, axis=-1) for a in d3qn_actions])
    
    for idx, (name, actions) in enumerate(zip(['PPO', 'D3QN'], [ppo_flat, d3qn_flat])):
        counts = jnp.bincount(actions, length=3)
        plt.bar(jnp.arange(3) + idx*0.3, counts, width=0.3, label=name)
    
    plt.title("Action Type Distribution")
    plt.xlabel("Action Type")
    plt.ylabel("Count")
    plt.xticks([0, 1, 2], ['Power Down', 'Maintain', 'Power Up'])
    plt.legend()
    plt.savefig('plots/action_distribution.png')
    plt.close()


def plot_metrics_comparison(ppo_metrics: Dict, d3qn_metrics: Dict):
    """Plot performance metrics comparison"""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    ppo_throughput = jnp.convolve(jnp.array(ppo_metrics['throughput']), jnp.ones(10)/10, mode='valid')
    d3qn_throughput = jnp.convolve(jnp.array(d3qn_metrics['throughput']), jnp.ones(10)/10, mode='valid')
    axs[0].plot(ppo_throughput, label='PPO')
    axs[0].plot(d3qn_throughput, label='D3QN')
    axs[0].set_title("Average Throughput (Moving Window)")
    axs[0].set_ylabel("Mbps")
    axs[0].legend()
    
    axs[1].plot(jnp.cumsum(jnp.array(ppo_metrics['handovers'])), label='PPO')
    axs[1].plot(jnp.cumsum(jnp.array(d3qn_metrics['handovers'])), label='D3QN')
    axs[1].set_title("Cumulative Handovers")
    axs[1].set_ylabel("Count")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('plots/metrics_comparison.png')
    plt.close()
