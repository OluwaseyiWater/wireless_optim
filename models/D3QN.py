import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]   = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]  = "0.5"

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import matplotlib.pyplot as plt
from dataclasses import dataclass
from jax import tree_util
import wandb
from typing import Any, Tuple, List

# Assuming environment is available (HetNetEnvironment)
# from wireless_optim.environment import *

# Assuming utils.py contains unflatten_action if used
# from utils import unflatten_action

# Define a Transition dataclass to hold experience data
@dataclass(frozen=True)
class Transition:
    obs: jnp.ndarray
    action_indices: jnp.ndarray
    reward: float
    next_obs: jnp.ndarray
    done: float

tree_util.register_pytree_node(
    Transition,
    lambda t: ((t.obs, t.action_indices, t.reward, t.next_obs, t.done), None),
    lambda _, children: Transition(*children)
)

# Optimized Replay Buffer using NumPy arrays
class ReplayBuffer:
    # Replay buffer using pre-allocated NumPy arrays for efficient storage.
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_indices_shape: Tuple[int, ...]):
        self.capacity = capacity
        flat_obs_dim = int(np.prod(obs_shape))
        self.obs_buffer = np.empty((capacity, flat_obs_dim), dtype=np.float32)
        self.action_indices_buffer = np.empty((capacity, *action_indices_shape), dtype=np.int32)
        self.reward_buffer = np.empty(capacity, dtype=np.float32)
        self.next_obs_buffer = np.empty((capacity, flat_obs_dim), dtype=np.float32)
        self.done_buffer = np.empty(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def add(self, transition: Transition):
        self.obs_buffer[self.position] = np.asarray(transition.obs).flatten()
        self.action_indices_buffer[self.position] = np.asarray(transition.action_indices)
        self.reward_buffer[self.position] = np.asarray(transition.reward)
        self.next_obs_buffer[self.position] = np.asarray(transition.next_obs).flatten()
        self.done_buffer[self.position] = np.asarray(transition.done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, key: jax.random.PRNGKey):
        indices = jax.random.choice(key, self.size, shape=(batch_size,), replace=False)
        indices_np = np.asarray(indices)

        batch_obs = self.obs_buffer[indices_np]
        batch_action_indices = self.action_indices_buffer[indices_np]
        batch_rewards = self.reward_buffer[indices_np]
        batch_next_obs = self.next_obs_buffer[indices_np]
        batch_dones = self.done_buffer[indices_np]

        return Transition(
            obs=jax.device_put(batch_obs),
            action_indices=jax.device_put(batch_action_indices),
            reward=jax.device_put(batch_rewards),
            next_obs=jax.device_put(batch_next_obs),
            done=jax.device_put(batch_dones),
        )

    def __len__(self):
        return self.size

# D3QN Agent
class D3QN:
    # Dueling Deep Q-Network agent for discrete action spaces.
    def __init__(self, obs_dim: int, num_bs: int, action_dims_per_bs: int = 3, num_bins_per_dimension: int = 2, lr: float = 3e-4, gamma: float = 0.99):
        self.num_bs = num_bs
        self.action_dims_per_bs = action_dims_per_bs
        self.num_bins_per_dimension = num_bins_per_dimension
        self.num_actions_per_bs = num_bins_per_dimension ** action_dims_per_bs
        self.total_action_dim = self.num_bs * self.num_actions_per_bs
        self.gamma = gamma

        def q_network_fn(x):
            hidden = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
            ])(x)
            value = hk.Linear(1)(hidden)
            advantages = hk.Linear(self.num_bs * self.num_actions_per_bs)(hidden)
            advantages_reshaped = advantages.reshape(-1, self.num_bs, self.num_actions_per_bs)
            mean_advantages = jnp.mean(advantages_reshaped, axis=-1, keepdims=True)
            q_values = value[:, None, None] + (advantages_reshaped - mean_advantages)
            return q_values

        self.net = hk.without_apply_rng(hk.transform(q_network_fn))
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr)
        )

        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        self.params = self.net.init(key, dummy_obs)
        self.target_params = self.params
        self.opt_state = self.optimizer.init(self.params)

    @staticmethod
    def d3qn_loss(
        params, target_params, transitions: Transition,
        net_apply_fn, gamma_val: float, num_bs_val: int, num_actions_per_bs_val: int
    ):
        # Compute the D3QN loss.
        q3  = net_apply_fn(params, transitions.obs)
        q3t = net_apply_fn(target_params, transitions.next_obs)

        batch_size = q3.shape[0]
        q_flat  = q3.reshape(batch_size, -1)
        qf_flat = q3t.reshape(batch_size, -1)

        powers = jnp.arange(num_bs_val -1, -1, -1)
        muls = (num_actions_per_bs_val ** powers)
        flat_idx = jnp.sum(transitions.action_indices * muls, axis=1)

        batch_idx = jnp.arange(batch_size)
        q_selected = q_flat[batch_idx, flat_idx]
        next_best  = jnp.max(qf_flat, axis=1)
        target_q   = transitions.reward + (1.0 - transitions.done) * gamma_val * next_best

        loss = jnp.mean(optax.huber_loss(q_selected, jax.lax.stop_gradient(target_q), delta=1.0))
        return loss

    # CORRECTED: Define the training step as a static method inside the class
    @staticmethod
    def train_step(
        params, opt_state, target_params, transitions: Transition, key, global_step_jax,
        net_apply_fn, optimizer_update_fn, gamma_val: float, num_bs_val: int,
        num_actions_per_bs_val: int, tau_val: float, target_update_freq_val: int # Corrected: target_update_freq_val is int
    ):
        # Calculate loss
        loss, grads = jax.value_and_grad(D3QN.d3qn_loss)(
            params, target_params, transitions,
            net_apply_fn, gamma_val, num_bs_val, num_actions_per_bs_val
        )

        # Apply gradients
        updates, new_opt_state = optimizer_update_fn(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Soft/Hard target update logic
        # CORRECTED: Convert target_update_freq_val (int) to JAX array inside the JIT function
        target_update_condition = (global_step_jax % jnp.array(target_update_freq_val, dtype=jnp.int32)) == 0

        def soft_update(current_p, target_p, t):
             return jax.tree.map(lambda tp, p: t * p + (1.0 - t) * tp, target_p, current_p)

        def hard_update(current_p):
            return current_p

        updated_target_params = jax.lax.cond(
            tau_val < 1.0,
            lambda: soft_update(new_params, target_params, tau_val),
            lambda: jax.lax.cond(
                 target_update_condition,
                 lambda: hard_update(new_params),
                 lambda: target_params
            )
        )

        return new_params, new_opt_state, updated_target_params, loss, key


# Helper function for decoding discrete indices to continuous actions
def indices_to_actions(action_indices, num_bs, num_bins_per_dimension, action_dims_per_bs):
    # Convert discrete action indices (shape (num_bs,)) to continuous actions (shape (num_bs, action_dims_per_bs)).
    def decode_single_index(idx):
        decoded_actions = []
        current_idx = idx
        for dim in range(action_dims_per_bs):
            dim_idx = current_idx % num_bins_per_dimension
            continuous_val = dim_idx / (num_bins_per_dimension - 1.0) if num_bins_per_dimension > 1 else float(dim_idx)
            decoded_actions.append(continuous_val)
            current_idx = current_idx // num_bins_per_dimension
        return jnp.array(decoded_actions[::-1], dtype=jnp.float32)

    decoded_actions_vmap = jax.vmap(decode_single_index)(action_indices)
    return decoded_actions_vmap

# Main training function
def train_d3qn(
    env: Any,
    num_episodes: int = 100,
    batch_size: int = 256,
    replay_capacity: int = 100000,
    seed: int = 0,
    lr: float = 3e-4,
    gamma: float = 0.99,
    target_update_freq: int = 10,
    tau: float = 1.0,
    warmup_steps: int = 10000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_steps: int = 100000,
    wandb_project: str = "d3qn-training",
    wandb_name: str = None,
    use_wandb: bool = True,
) -> Tuple[D3QN, List[float], List[float], List[float], List[float], List[float]]:

    # --- Setup Environment & Agent ---
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()

    obs_dim = int(np.prod(obs_spec.shape))
    num_bs, action_dims_per_bs = action_spec.shape
    num_bins_per_dimension = 2
    num_actions_per_bs = num_bins_per_dimension ** action_dims_per_bs

    agent = D3QN(
        obs_dim=obs_dim,
        num_bs=num_bs,
        action_dims_per_bs=action_dims_per_bs,
        num_bins_per_dimension=num_bins_per_dimension,
        lr=lr,
        gamma=gamma
    )

    # --- Setup Replay Buffer ---
    buffer = ReplayBuffer(
        capacity=replay_capacity,
        obs_shape=obs_spec.shape,
        action_indices_shape=(num_bs,)
    )

    key = jax.random.PRNGKey(seed)
    global_step = 0

    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "algorithm": "D3QN",
                "num_episodes":    num_episodes,
                "batch_size":      batch_size,
                "replay_capacity": replay_capacity,
                "warmup_steps":    warmup_steps,
                "lr":              lr,
                "gamma":           gamma,
                "seed":            seed,
                "target_update_freq": target_update_freq,
                "tau":             tau,
                "epsilon_start":   epsilon_start,
                "epsilon_end":     epsilon_end,
                "epsilon_decay_steps": epsilon_decay_steps,
                "env_obs_dim": obs_dim,
                "env_num_bs": num_bs,
                "env_action_dims_per_bs": action_dims_per_bs,
                "agent_num_actions_per_bs": agent.num_actions_per_bs,
            }
        )
        if wandb_name is None:
             run.name = f"D3QN_seed{seed}"
             wandb.run.name = run.name

    print("Starting D3QN training...", flush=True)

    # --- JIT Training Step ---
    # CORRECTED: JIT the static method D3QN.train_step
    compiled_train_step = jax.jit(
        D3QN.train_step, # <-- Reference the static method
        static_argnames=[
             'net_apply_fn', 'optimizer_update_fn', 'gamma_val', 'num_bs_val',
             'num_actions_per_bs_val', 'tau_val', 'target_update_freq_val', # target_update_freq_val is Python int
        ]
    )

    # Pass these values as static args when calling compiled_train_step
    net_apply_fn_static = agent.net.apply
    optimizer_update_fn_static = agent.optimizer.update
    gamma_val_static = agent.gamma # Use agent's gamma
    num_bs_val_static = agent.num_bs
    num_actions_per_bs_val_static = agent.num_actions_per_bs
    tau_val_static = tau
    target_update_freq_val_static = target_update_freq # Use the Python int

    # --- Pre-fill replay buffer with random actions (warm-up) ---
    print(f"Warm-up: collecting {warmup_steps} random transitions...")
    current_warmup_steps = 0
    warmup_key = jax.random.fold_in(key, 1000)
    key, _ = jax.random.split(key)

    while current_warmup_steps < warmup_steps:
        warmup_key, reset_key = jax.random.split(warmup_key)
        ts = env.reset(reset_key)
        obs = ts.observation
        done = bool(ts.discount == 0.0)

        while not done and current_warmup_steps < warmup_steps:
            warmup_key, akey = jax.random.split(warmup_key)
            rand_idx = jax.random.randint(akey, (num_bs,), 0, agent.num_actions_per_bs)

            action = indices_to_actions(
                rand_idx, num_bs,
                num_bins_per_dimension=num_bins_per_dimension,
                action_dims_per_bs=action_dims_per_bs
            )

            ts2 = env.step(action)
            next_obs = ts2.observation
            done = bool(ts2.discount == 0.0)
            reward = float(ts2.reward)

            buffer.add(Transition(
                obs=obs.flatten(),
                action_indices=rand_idx,
                reward=jnp.clip(reward, -1e3, 1e3) / 1e3,
                next_obs=next_obs.flatten(),
                done=ts2.discount
            ))
            obs = next_obs
            current_warmup_steps += 1
            global_step += 1

    print(f"Warm-up complete. Buffer size: {len(buffer)}\n")

    episode_rewards: List[float] = []
    episode_losses:  List[float] = []
    episode_powers = []
    episode_bandwidths = []
    episode_scheds = []

    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        ts = env.reset(reset_key)
        obs = ts.observation
        done = bool(ts.discount == 0.0)

        ep_reward = 0.0
        ep_loss   = 0.0
        train_steps_in_episode = 0

        power_means:     List[float] = []
        bandwidth_means: List[float] = []
        sched_means:     List[float] = []

        while not done:
            key, akey, sample_key = jax.random.split(key, 3)

            # --- Action Selection (Epsilon-Greedy) ---
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (global_step / epsilon_decay_steps))

            if jax.random.uniform(akey) < epsilon:
                action_idx = jax.random.randint(
                    akey, (num_bs,), 0, agent.num_actions_per_bs
                )
            else:
                qvals = agent.net.apply(agent.params, obs.flatten()[None, :])
                q_flat = qvals.reshape((num_bs * agent.num_actions_per_bs,))
                global_flat_idx = jnp.argmax(q_flat)

                action_idx_list = []
                current_global_idx = global_flat_idx
                for i in range(num_bs):
                    bs_idx = current_global_idx % agent.num_actions_per_bs
                    action_idx_list.append(bs_idx)
                    current_global_idx = current_global_idx // agent.num_actions_per_bs
                action_idx = jnp.array(action_idx_list[::-1], dtype=jnp.int32)

            # — build continuous action for the environment —
            action = indices_to_actions(
                action_idx, num_bs,
                num_bins_per_dimension=num_bins_per_dimension,
                action_dims_per_bs=action_dims_per_bs
            )

            # — record action stats (using continuous actions for env) —
            pa = np.array(action[:, 0]); power_means.append(pa.mean())
            ba = np.array(action[:, 1]); bandwidth_means.append(ba.mean())
            ss = np.array(action[:, 2]); sched_means.append(ss.mean())

            # — step environment —
            ts2 = env.step(action)
            next_obs = ts2.observation
            done     = bool(ts2.discount == 0.0)
            reward   = float(ts2.reward)

            # — W&B per-step logs —
            if use_wandb:
                log_dict = {
                    "env/power_adjustments":     wandb.Histogram(pa),
                    "env/bandwidth_allocations": wandb.Histogram(ba),
                    "env/scheduling_scores":     wandb.Histogram(ss),
                    "env/pa_mean":               pa.mean(),
                    "env/ba_mean":               ba.mean(),
                    "env/ss_mean":               ss.mean(),
                    "global_step":               global_step,
                    "metrics/reward_per_step":   reward,
                    "agent/epsilon":             epsilon,
                    "buffer_size":               len(buffer),
                }
                wandb.log(log_dict, step=global_step)
            global_step += 1

            # — store transition —
            buffer.add(Transition(
                obs=obs.flatten(),
                action_indices=action_idx,
                reward=jnp.clip(reward, -1e3, 1e3) / 1e3,
                next_obs=next_obs.flatten(),
                done=ts2.discount
            ))

            obs = next_obs
            ep_reward += reward

            # --- Train the Agent (if buffer is large enough) ---
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size, sample_key)

                # CORRECTED: Call the compiled staticmethod with correct static args
                agent.params, agent.opt_state, agent.target_params, loss, key = compiled_train_step(
                    agent.params, agent.opt_state, agent.target_params, batch, key,
                    jnp.array(global_step, dtype=jnp.int32), # Dynamic global_step
                    # Static Arguments (matched by name in static_argnames)
                    net_apply_fn=net_apply_fn_static,
                    optimizer_update_fn=optimizer_update_fn_static,
                    gamma_val=gamma_val_static,
                    num_bs_val=num_bs_val_static,
                    num_actions_per_bs_val=num_actions_per_bs_val_static,
                    tau_val=tau_val_static,
                    target_update_freq_val=target_update_freq_val_static, # Pass the Python int static arg
                )

                ep_loss += float(loss)
                train_steps_in_episode += 1

                # W&B training logs (per training step)
                if use_wandb:
                    wandb.log({
                        "train/loss": float(loss),
                        "global_step": global_step,
                    }, step=global_step)

        # end of episode
        avg_loss = ep_loss / max(train_steps_in_episode, 1)

        episode_rewards.append(ep_reward)
        episode_losses.append(avg_loss)
        episode_powers.append(np.mean(power_means) if power_means else 0.0)
        episode_bandwidths.append(np.mean(bandwidth_means) if bandwidth_means else 0.0)
        episode_scheds.append(np.mean(sched_means) if sched_means else 0.0)

        jax.clear_caches()

        print(f"Episode {ep}: Reward={ep_reward:.2f}, AvgLoss={avg_loss:.4f}, Steps={global_step}")

        # W&B episode logs
        if use_wandb:
            wandb.log({
                "episode":                ep,
                "total_reward":           ep_reward,
                "episode_avg_loss":       avg_loss,
                "epsilon":                epsilon,
                "train_steps_in_episode": train_steps_in_episode,
                "mean_power_adjust":      np.mean(power_means) if power_means else 0.0,
                "mean_bandwidth_alloc":   np.mean(bandwidth_means) if bandwidth_means else 0.0,
                "mean_scheduling_score":  np.mean(sched_means) if sched_means else 0.0,
            }, step=global_step)

    if use_wandb:
        wandb.finish()

    return agent, episode_rewards, episode_losses, episode_powers, episode_bandwidths, episode_scheds
