import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import os
from utils import unflatten_action
from jax import tree_util

from wireless_optim.environment import *


# Define a Transition dataclass to hold experience data
@dataclass
class Transition:
    obs: jnp.ndarray          # Current observation
    action_indices: jnp.ndarray  # Discrete action indices for each base station (BS)
    reward: float             # Reward received
    next_obs: jnp.ndarray     # Next observation
    done: float               # Done flag (1.0 if terminal, 0.0 otherwise)

tree_util.register_pytree_node(
    Transition,
    lambda t: ((t.obs, t.action_indices, t.reward, t.next_obs, t.done), None),
    lambda _, children: Transition(*children)
)

# Simple replay buffer to store transitions
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, key: jax.random.PRNGKey):
        indices = jax.random.choice(key, len(self.buffer), shape=(batch_size,), replace=False)
        batch = [self.buffer[int(idx)] for idx in indices]
        obs = jnp.stack([t.obs for t in batch])
        action_indices = jnp.stack([jnp.reshape(t.action_indices, (-1,)) for t in batch])
        rewards = jnp.array([t.reward for t in batch])
        next_obs = jnp.stack([t.next_obs for t in batch])
        dones = jnp.array([t.done for t in batch])
        return Transition(obs, action_indices, rewards, next_obs, dones)

class D3QN:
    def __init__(self, obs_dim, num_bs, action_dims_per_bs=3, num_bins_per_dimension=2, lr=3e-4, gamma=0.99):
        """
        Initialize the D3QN agent.

        Args:
            obs_dim (int): Dimension of the observation space.
            num_bs (int): Number of base stations (e.g., 13 for 3 macro + 10 small).
            action_dims_per_bs (int): Number of action dimensions per BS (e.g., 3 for power, bandwidth, scheduling).
            num_bins_per_dimension (int): Number of discrete bins per action dimension (e.g., 2 for 0.0 and 1.0).
            lr (float): Learning rate (default matches PPO at 3e-4).
            gamma (float): Discount factor.
        """
        self.num_bs = num_bs
        self.action_dims_per_bs = action_dims_per_bs
        self.num_bins_per_dimension = num_bins_per_dimension
        self.num_actions_per_bs = num_bins_per_dimension ** action_dims_per_bs  # e.g., 2^3 = 8
        self.total_action_dim = self.num_bs * self.num_actions_per_bs  # e.g., 13 * 8 = 104
        self.gamma = gamma
        self.obs_dim = obs_dim

        # Define the dueling Q-network architecture
        def q_network(x):
            hidden = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
            ])(x)
            value = hk.Linear(1)(hidden)  # State value: (batch_size, 1)
            advantages = hk.Linear(self.num_bs * self.num_actions_per_bs)(hidden)  # Advantages: (batch_size, 104)
            advantages = advantages.reshape(-1, self.num_bs, self.num_actions_per_bs)  # (batch_size, 13, 8)
            mean_advantages = jnp.mean(advantages, axis=-1, keepdims=True)  # (batch_size, 13, 1)
            q_values = value[:, None, None] + (advantages - mean_advantages)  # (batch_size, 13, 8)
            return q_values

        # Initialize the network and optimizer
        self.net = hk.without_apply_rng(hk.transform(q_network))
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  
            optax.adam(lr)
        )
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((self.obs_dim,))
        self.params = self.net.init(key, dummy_obs)
        self.target_params = self.params  # Initialize target network with same params
        self.opt_state = self.optimizer.init(self.params)

    def _d3qn_loss(self, params, target_params, transitions: Transition):
        """
        Compute the D3QN loss.

        Args:
            params: Current network parameters.
            target_params: Target network parameters.
            transitions: Batch of transitions.

        Returns:
            float: Mean squared error loss.
        """
        # (1) get the 3‑D Q: (batch, num_bs, num_actions_per_bs)
        q3  = self.net.apply(params,        transitions.obs)
        q3t = self.net.apply(target_params, transitions.next_obs)
    
        # (2) flatten the last two dims into one global action‑dim
        batch_size = q3.shape[0]
        q_flat  = q3.reshape(batch_size, -1)   # (batch, num_bs*num_actions_per_bs)
        qf_flat = q3t.reshape(batch_size, -1)  # same
    
        # (3) flatten the multi‑dim action_indices into a single integer per sample
        muls = (self.num_actions_per_bs ** jnp.arange(self.num_bs))[None, :]  # (1, num_bs)
        flat_idx = jnp.sum(transitions.action_indices * muls, axis=1)         # (batch,)
    
        # (4) now use the original 2‑D indexing
        batch_idx = jnp.arange(batch_size)
        q_selected = q_flat[batch_idx, flat_idx]
        next_best  = jnp.max(qf_flat, axis=1)
        target_q   = transitions.reward + (1-transitions.done)*self.gamma*next_best
    
        loss = jnp.mean(optax.huber_loss(q_selected, target_q, delta=1.0))
        return loss
        
    def update(self, batch: Transition):
        """
        Update the Q-network parameters using a batch of transitions.

        Args:
            batch: Batch of transitions.

        Returns:
            float: Loss value for the update step.
        """
        loss_value, grads = jax.value_and_grad(self._d3qn_loss)(self.params, self.target_params, batch)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss_value

    def update_target_network(self):
        """Update the target network parameters to match the current network."""
        self.target_params = self.params


def indices_to_actions(action_indices, num_bs, num_actions_per_bs=8):
    """
    Convert discrete action indices to continuous actions (e.g., 0.0 or 1.0).

    Args:
        action_indices: Array of discrete action indices for each BS (shape: (num_bs,)).
        num_bs: Number of base stations.
        num_actions_per_bs: Number of discrete actions per BS (e.g., 8 for 2^3).

    Returns:
        jnp.ndarray: Continuous actions (shape: (num_bs, action_dims_per_bs)).
    """
    actions = []
    for idx in action_indices:
        # Convert index to binary representation (assuming 3 dimensions per BS)
        binary = [(idx >> i) & 1 for i in range(3)]  # e.g., idx=5 -> [1, 0, 1]
        actions.append(jnp.array(binary, dtype=jnp.float32))
    return jnp.stack(actions)  # (num_bs, 3)

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from typing import Any, Tuple, List

def train_d3qn(
    env: Any,
    num_episodes: int = 100,
    batch_size: int = 256,
    replay_capacity: int = 10000,
    seed: int = 0,
    lr= 1e-7,
    wandb_project: str = "d3qn-training",
    wandb_name: str = None,
    use_wandb: bool = True,
) -> Tuple["D3QN", List[float], List[float]]:
    # ——— Setup agent & buffer ———
    obs_dim = int(np.prod(env.observation_spec().shape))
    num_bs  = env.action_spec().shape[0]
    agent   = D3QN(obs_dim, num_bs, lr= lr, gamma=0.99)
    buffer  = ReplayBuffer(capacity=replay_capacity)
    key     = jax.random.PRNGKey(seed)

    # JIT the update functions once
    def update_fn(params, opt_state, target_params, transitions):
        loss, grads = jax.value_and_grad(agent._d3qn_loss)(params, target_params, transitions)
        updates, new_opt_state = agent.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
        
    compiled_update = jax.jit(update_fn)
    compiled_update_target = jax.jit(agent.update_target_network)

    episode_rewards: List[float] = []
    episode_losses:  List[float] = []
    episode_powers = []
    episode_bandwidths = []
    episode_scheds = []

    min_replay  = 2000
    global_step = 0

    # ——— WandB Init ———
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "num_episodes":    num_episodes,
                "batch_size":      batch_size,
                "replay_capacity": replay_capacity,
                "min_replay_size": min_replay,
                "lr":              1e-5,
                "gamma":           0.99,
                "seed":            seed
            }
        )

    print("Starting D3QN training...", flush=True)

    # ——— Pre-fill replay buffer with random actions (warm-up) ———
    print(f"Warm-up: collecting {min_replay} random transitions...")
    while len(buffer.buffer) < min_replay:
        key, subkey = jax.random.split(key)
        ts = env.reset(subkey)
        done = False
        while not done and len(buffer.buffer) < min_replay:
            # random discrete indices per BS
            key, akey = jax.random.split(key)
            rand_idx = jax.random.randint(akey, (num_bs,), 0, agent.num_actions_per_bs)
            action = indices_to_actions(rand_idx, num_bs)
            ts2 = env.step(action)

            # store transition
            buffer.add(Transition(
                obs=ts.observation.flatten(),
                action_indices=rand_idx,
                reward=float(ts2.reward) / 1000.0,
                next_obs=ts2.observation.flatten(),
                done=1.0 if ts2.discount == 0 else 0.0
            ))
            done = bool(ts2.discount == 0)
            ts = ts2
    print("Warm-up complete.\n")

    # ——— Training Loop ———
    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        ts = env.reset(reset_key)
        obs = ts.observation
        done = False

        ep_reward = 0.0
        ep_loss   = 0.0
        updates   = 0

        power_means:     List[float] = []
        bandwidth_means: List[float] = []
        sched_means:     List[float] = []

        while not done:
            key, akey = jax.random.split(key)
            epsilon = max(0.01, 0.1 * (0.98 ** ep))

            # — pick action indices — 
            if jax.random.uniform(akey) < epsilon:
                action_idx = jax.random.randint(
                    akey, (num_bs,), 0, agent.num_actions_per_bs
                )
            else:
                qvals = agent.net.apply(agent.params, obs)
                # flatten + global argmax, then decode per-BS
                q_flat     = qvals.reshape((-1,))
                global_idx = int(jnp.argmax(q_flat))
                one_hot    = unflatten_action(global_idx, (num_bs, agent.num_actions_per_bs))
                action_idx = jnp.argmax(one_hot, axis=1)

            # — build continuous action — 
            action = indices_to_actions(action_idx, num_bs)

            # — record action stats — 
            pa = np.array(action[:, 0]); power_means.append(pa.mean())
            ba = np.array(action[:, 1]); bandwidth_means.append(ba.mean())
            ss = np.array(action[:, 2]); sched_means.append(ss.mean())

            # — step environment — 
            ts2 = env.step(action)
            next_obs = ts2.observation
            done     = bool(ts2.discount == 0)
            reward   = float(ts2.reward)

            # — W&B per-step logs — 
            if use_wandb:
                wandb.log({
                    "env/power_adjustments":     wandb.Histogram(pa),
                    "env/bandwidth_allocations": wandb.Histogram(ba),
                    "env/scheduling_scores":     wandb.Histogram(ss),
                    "env/pa_mean":               pa.mean(),
                    "env/ba_mean":               ba.mean(),
                    "env/ss_mean":               ss.mean(),
                    "global_step":               global_step,
                }, step=global_step)
            global_step += 1

            # — store transition — 
            buffer.add(Transition(
                obs=obs.flatten(),
                action_indices=action_idx,
                reward= jnp.clip((reward), -1e3, 1e3) / 1e3,
                next_obs=next_obs.flatten(),
                done=1.0 if done else 0.0
            ))
            obs = next_obs
            ep_reward += reward

            # — update via compiled JIT after warm-up — 
            key, skey = jax.random.split(key)
            batch = buffer.sample(batch_size, skey)
            agent.params, agent.opt_state, loss = compiled_update(
                agent.params, agent.opt_state, agent.target_params, batch
            )
            ep_loss += float(loss)
            updates += 1

            tau = 0.005
            agent.target_params = jax.tree.map(
                lambda t, s: t * (1 - tau) + s * tau,
                agent.target_params,
                agent.params,
            )

        # end of episode
        avg_loss = ep_loss / max(updates, 1)
        episode_rewards.append(ep_reward)
        episode_losses.append(avg_loss)
        episode_powers.append(np.mean(power_means))
        episode_bandwidths.append(np.mean(bandwidth_means))
        episode_scheds.append(np.mean(sched_means))

        if ep % 10 == 0:
            compiled_update_target()

        jax.clear_caches()

        print(f"Episode {ep}: Reward={ep_reward:.2f}, AvgLoss={avg_loss:.4f}")

        if use_wandb:
            wandb.log({
                "episode":                ep,
                "total_reward":           ep_reward,
                "avg_loss":               avg_loss,
                "epsilon":                epsilon,
                "updates":                updates,
                "mean_power_adjust":      np.mean(power_means),
                "mean_bandwidth_alloc":   np.mean(bandwidth_means),
                "mean_scheduling_score":  np.mean(sched_means),
            }, step=global_step)

    if use_wandb:
        wandb.finish()

    return agent, episode_rewards, episode_losses,  episode_powers, episode_bandwidths, episode_scheds


