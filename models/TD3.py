import os
# don’t grab all GPU/host RAM at startup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]   = "false"
# only use e.g. 50% of GPU or host RAM for JAX buffers
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]  = "0.5"

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Tuple, List
import wandb
from wireless_optim.environment import HetNetEnvironment

class SumTree:
    # SumTree implementation for efficient sampling and priority updates in PER.
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        return self.tree[0]

    def add(self, priority, data):
        tree_idx = self.data_ptr + self.capacity - 1
        self.data[self.data_ptr] = data
        self.update(tree_idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return (tree_idx, self.tree[tree_idx], self.data[data_idx])

class PrioritizedReplayBuffer:
    # Prioritized Replay Buffer storing transitions with associated priorities and providing IS weights.
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.sumtree = SumTree(capacity)
        self.epsilon = 1e-6
        self.max_priority = 1.0

        @dataclass(frozen=True)
        class PERTransition:
            obs: jnp.ndarray
            action: jnp.ndarray
            reward: float
            next_obs: jnp.ndarray
            done: float

        self.Transition = PERTransition

        jax.tree_util.register_pytree_node(
            self.Transition,
            lambda t: ((t.obs, t.action, t.reward, t.next_obs, t.done), None),
            lambda _, children: self.Transition(*children)
        )

    def add(self, transition: Any):
        priority = self.max_priority ** self.alpha
        self.sumtree.add(priority, transition)

    def sample(self, batch_size: int, key: jax.random.PRNGKey, beta: float):
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        batch_transitions = []
        batch_is_weights = np.zeros(batch_size, dtype=np.float32)

        total_priority = self.sumtree.total_priority()
        samples = np.random.uniform(0.0, total_priority, batch_size)

        for i in range(batch_size):
            s = samples[i]
            (tree_idx, priority, transition) = self.sumtree.get(s)
            batch_indices[i] = tree_idx
            batch_transitions.append(transition)

        batched_transitions = self.Transition(
            obs=jnp.stack([t.obs for t in batch_transitions]),
            action=jnp.stack([t.action for t in batch_transitions]),
            reward=jnp.stack([t.reward for t in batch_transitions]),
            next_obs=jnp.stack([t.next_obs for t in batch_transitions]),
            done=jnp.stack([t.done for t in batch_transitions]),
        )

        sampled_priorities = np.asarray([self.sumtree.tree[idx] for idx in batch_indices])
        p_i_sampled = sampled_priorities / total_priority
        p_min_sampled = np.min(p_i_sampled[p_i_sampled > 1e-9]) if np.any(p_i_sampled > 1e-9) else 1.0

        batch_is_weights = ((p_min_sampled / (p_i_sampled + 1e-9)) ** beta)

        max_is_weight = np.max(batch_is_weights) if len(batch_is_weights) > 0 else 1.0
        batch_is_weights = batch_is_weights / (max_is_weight + 1e-9)

        batch_indices_jax = jnp.array(batch_indices, dtype=jnp.int32)
        batch_is_weights_jax = jnp.array(batch_is_weights, dtype=jnp.float32)

        return batched_transitions, batch_indices_jax, batch_is_weights_jax

    def update_priorities(self, tree_indices, td_errors):
        td_errors_np = np.asarray(td_errors)
        for i in range(len(tree_indices)):
            priority = (abs(td_errors_np[i]) + self.epsilon) ** self.alpha
            self.sumtree.update(tree_indices[i], priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.sumtree.n_entries


# --- TD3 Agent Implementation ---
class TD3Agent:
    # TD3 Actor-Critic agent with Twin Critics, Delayed Policy Updates, and Target Smoothing.
    def __init__(self, obs_dim: int, num_bs: int, action_dims_per_bs: int,
                 action_scale: float, action_bias: float,
                 lr_actor: float = 1e-4, lr_critic: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_delay: int = 2):
        self.num_bs = num_bs
        self.action_dims_per_bs = action_dims_per_bs
        self.total_action_dim = num_bs * action_dims_per_bs
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        # Network Definitions (Haiku)
        def actor_fn(x):
            net = hk.Sequential([
                hk.Linear(256), jax.nn.relu,
                hk.Linear(256), jax.nn.relu,
                hk.Linear(self.total_action_dim),
                jax.nn.tanh,
            ])(x)
            return self.action_scale * net + self.action_bias

        def critic_fn(x, a):
            inputs = jnp.concatenate([x, a.reshape(a.shape[0], -1)], axis=-1)
            net = hk.Sequential([
                hk.Linear(256), jax.nn.relu,
                hk.Linear(256), jax.nn.relu,
                hk.Linear(1),
            ])(inputs)
            return net

        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        self.critic1 = hk.without_apply_rng(hk.transform(critic_fn))
        self.critic2 = hk.without_apply_rng(hk.transform(critic_fn))

        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, self.num_bs, self.action_dims_per_bs))

        key, init_key = jax.random.split(key)
        self.actor_params = self.actor.init(init_key, dummy_obs)
        self.target_actor_params = self.actor_params

        key, init_key = jax.random.split(key)
        self.critic1_params = self.critic1.init(init_key, dummy_obs, dummy_action)
        self.target_critic1_params = self.critic1_params

        key, init_key = jax.random.split(key)
        self.critic2_params = self.critic2.init(init_key, dummy_obs, dummy_action)
        self.target_critic2_params = self.critic2_params

        self.actor_optimizer = optax.adam(lr_actor)
        self.critic_optimizer = optax.adam(lr_critic)

        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic1_opt_state = self.critic_optimizer.init(self.critic1_params)
        self.critic2_opt_state = self.critic_optimizer.init(self.critic2_params)

    @staticmethod
    # JIT-compiled training step performing critic and (delayed) actor updates with PER and target smoothing.
    def train_step(
        actor_params, critic1_params, critic2_params,
        target_actor_params, target_critic1_params, target_critic2_params,
        actor_opt_state, critic1_opt_state, critic2_opt_state,
        batch: Any,
        batch_is_weights: jnp.ndarray,
        key: jax.random.PRNGKey,
        global_step: jnp.ndarray,
        actor_apply_fn, critic1_apply_fn, critic2_apply_fn,
        actor_optimizer_update_fn, critic_optimizer_update_fn,
        action_scale: float, action_bias: float,
        policy_noise: float, noise_clip: float,
        policy_delay: int, gamma: float, tau: float,
        num_bs: int, action_dims_per_bs: int
    ):
        obs, action, reward, next_obs, done = batch.obs, batch.action, batch.reward, batch.next_obs, batch.done
        batch_size = obs.shape[0]

        # --- Critic Update ---
        key, noise_key = jax.random.split(key)

        target_next_actions_flat = actor_apply_fn(target_actor_params, next_obs)

        noise = jax.random.normal(noise_key, shape=target_next_actions_flat.shape) * policy_noise
        noise = jnp.clip(noise, -noise_clip, noise_clip)

        noisy_target_next_actions_flat = target_next_actions_flat + noise
        clipped_target_next_actions_flat = jnp.clip(noisy_target_next_actions_flat, 0.0, 1.0)

        clipped_target_next_actions_reshaped = clipped_target_next_actions_flat.reshape(batch_size, num_bs, action_dims_per_bs)

        target_q1 = critic1_apply_fn(target_critic1_params, next_obs, clipped_target_next_actions_reshaped)
        target_q2 = critic2_apply_fn(target_critic2_params, next_obs, clipped_target_next_actions_reshaped)
        target_q = jnp.min(jnp.concatenate([target_q1, target_q2], axis=-1), axis=-1, keepdims=True)

        bellman_target = reward[:, None] + gamma * (1.0 - done[:, None]) * target_q

        # CORRECTED: Define separate loss functions for each critic
        def critic1_loss_fn(c1_params):
            q1 = critic1_apply_fn(c1_params, obs, action)
            is_weights_reshaped = batch_is_weights[:, None]
            td_error = jnp.abs(q1 - jax.lax.stop_gradient(bellman_target))
            loss = jnp.mean(is_weights_reshaped * optax.huber_loss(q1, jax.lax.stop_gradient(bellman_target), delta=1.0))
            return loss, td_error

        def critic2_loss_fn(c2_params):
            q2 = critic2_apply_fn(c2_params, obs, action)
            is_weights_reshaped = batch_is_weights[:, None]
            td_error = jnp.abs(q2 - jax.lax.stop_gradient(bellman_target))
            loss = jnp.mean(is_weights_reshaped * optax.huber_loss(q2, jax.lax.stop_gradient(bellman_target), delta=1.0))
            return loss, td_error

        (critic1_loss, td_error1), critic1_grads = jax.value_and_grad(critic1_loss_fn, has_aux=True)(critic1_params)
        (critic2_loss, td_error2), critic2_grads = jax.value_and_grad(critic2_loss_fn, has_aux=True)(critic2_params)

        critic1_updates, new_critic1_opt_state = critic_optimizer_update_fn(critic1_grads, critic1_opt_state)
        critic2_updates, new_critic2_opt_state = critic_optimizer_update_fn(critic2_grads, critic2_opt_state)

        new_critic1_params = optax.apply_updates(critic1_params, critic1_updates)
        new_critic2_params = optax.apply_updates(critic2_params, critic2_updates)

        mean_td_errors = jnp.mean(jnp.concatenate([td_error1, td_error2], axis=-1), axis=-1)

        # --- Actor Update (Delayed) ---
        actor_loss = jnp.array(0.0)
        new_actor_params = actor_params
        new_actor_opt_state = actor_opt_state
        new_target_actor_params = target_actor_params
        new_target_critic1_params = target_critic1_params
        new_target_critic2_params = target_critic2_params

        actor_update_condition = (global_step % policy_delay) == 0

        def actor_update_step():
            def actor_loss_fn(a_params):
                current_action_flat = actor_apply_fn(a_params, obs)
                current_action_reshaped = current_action_flat.reshape(batch_size, num_bs, action_dims_per_bs)
                q_value = critic1_apply_fn(jax.lax.stop_gradient(new_critic1_params), obs, current_action_reshaped)
                return -jnp.mean(q_value)

            current_actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params)
            actor_updates, current_new_actor_opt_state = actor_optimizer_update_fn(actor_grads, actor_opt_state)
            current_new_actor_params = optax.apply_updates(actor_params, actor_updates)

            current_new_target_actor_params = jax.tree.map(lambda t, p: t * (1 - tau) + p * tau, target_actor_params, current_new_actor_params)
            current_new_target_critic1_params = jax.tree.map(lambda t, p: t * (1 - tau) + p * tau, target_critic1_params, new_critic1_params)
            current_new_target_critic2_params = jax.tree.map(lambda t, p: t * (1 - tau) + p * tau, target_critic2_params, new_critic2_params)

            return (current_new_actor_params, current_new_actor_opt_state,
                    current_new_target_actor_params, current_new_target_critic1_params, current_new_target_critic2_params,
                    current_actor_loss)

        (new_actor_params, new_actor_opt_state,
         new_target_actor_params, new_target_critic1_params, new_target_critic2_params,
         actor_loss) = jax.lax.cond(
             actor_update_condition,
             actor_update_step,
             lambda: (new_actor_params, new_actor_opt_state,
                      new_target_actor_params, new_target_critic1_params, new_target_critic2_params,
                      actor_loss)
         )

        # CORRECTED: Return updated critic opt states
        return (new_actor_params, new_critic1_params, new_critic2_params,
                new_target_actor_params, new_target_critic1_params, new_target_critic2_params,
                new_actor_opt_state, new_critic1_opt_state, new_critic2_opt_state,
                actor_loss, critic1_loss, critic2_loss, mean_td_errors, key)


# --- Training Function ---
def train_td3(
    env: HetNetEnvironment,
    num_episodes: int = 100,
    batch_size: int = 256,
    replay_capacity: int = 100000,
    seed: int = 0,
    lr_actor: float = 1e-4,
    lr_critic: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 100000,
    warmup_steps: int = 10000,
    action_noise_std: float = 0.1,
    wandb_project: str = "td3-per-hetnet",
    wandb_name: str = None,
    use_wandb: bool = True,
) -> Tuple[TD3Agent, List[float]]:

    # Setup Agent
    obs_dim = int(np.prod(env.observation_spec().shape))
    action_spec = env.action_spec()
    num_bs, action_dims_per_bs = action_spec.shape
    action_min = float(action_spec.minimum)
    action_max = float(action_spec.maximum)
    action_scale = (action_max - action_min) / 2.0
    action_bias = (action_max + action_min) / 2.0

    agent = TD3Agent(
        obs_dim=obs_dim,
        num_bs=num_bs,
        action_dims_per_bs=action_dims_per_bs,
        action_scale=action_scale,
        action_bias=action_bias,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
    )

    # Setup Replay Buffer
    buffer = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=per_alpha)

    key = jax.random.PRNGKey(seed)
    global_step = 0
    per_beta_increase = (1.0 - per_beta_start) / per_beta_frames

    # CORRECTED: JIT the staticmethod and provide static args
    # batch_indices is not needed in JIT, removed from static_argnames
    compiled_train_step = jax.jit(
        TD3Agent.train_step,
        static_argnames=[
            'actor_apply_fn', 'critic1_apply_fn', 'critic2_apply_fn',
            'actor_optimizer_update_fn', 'critic_optimizer_update_fn',
            'action_scale', 'action_bias',
            'policy_noise', 'noise_clip',
            'policy_delay', 'gamma', 'tau',
            'num_bs', 'action_dims_per_bs',
        ]
    )


    episode_rewards: List[float] = []
    episode_actor_losses: List[float] = []
    episode_critic_losses: List[float] = []
    episode_powers = []
    episode_bandwidths = []
    episode_scheds = []

    # WandB Init
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "algorithm":       "TD3_PER",
                "num_episodes":    num_episodes,
                "batch_size":      batch_size,
                "replay_capacity": replay_capacity,
                "warmup_steps":    warmup_steps,
                "lr_actor":        lr_actor,
                "lr_critic":       lr_critic,
                "gamma":           gamma,
                "tau":             tau,
                "policy_noise":    policy_noise,
                "noise_clip":      noise_clip,
                "policy_delay":    policy_delay,
                "per_alpha":       per_alpha,
                "per_beta_start":  per_beta_start,
                "per_beta_frames": per_beta_frames,
                "action_noise_std": action_noise_std,
                "seed":            seed,
                "env_num_macro_bs": env.num_macro_bs,
                "env_num_small_bs": env.num_small_bs,
                "env_num_users": env.num_users,
                "env_max_steps": env.max_steps,
            }
        )
        if wandb_name is None:
             run.name = f"TD3_PER_seed{seed}"
             wandb.run.name = run.name


    print("Starting TD3+PER training...", flush=True)

    print(f"Warm-up: collecting {warmup_steps} random transitions...")
    current_warmup_steps = 0
    warmup_key = jax.random.fold_in(key, 1000)
    key, _ = jax.random.split(key)

    while current_warmup_steps < warmup_steps:
        warmup_key, reset_key = jax.random.split(warmup_key)
        ts = env.reset(reset_key)
        obs = ts.observation
        done = bool(ts.discount == 0.0) # Use discount for done check
        while not done and current_warmup_steps < warmup_steps:
            warmup_key, akey = jax.random.split(warmup_key)
            random_action = jax.random.uniform(
                akey,
                (agent.num_bs, agent.action_dims_per_bs),
                minval=action_spec.minimum,
                maxval=action_spec.maximum,
                dtype=action_spec.dtype
            )

            ts2 = env.step(random_action)
            next_obs = ts2.observation
            done = bool(ts2.discount == 0.0)
            reward = float(ts2.reward)

            buffer.add(buffer.Transition(
                obs=obs.flatten(),
                action=random_action,
                reward=jnp.clip(reward, -1e3, 1e3) / 1e3,
                next_obs=next_obs.flatten(),
                done=ts2.discount
            ))
            obs = next_obs
            current_warmup_steps += 1
            global_step += 1

    print(f"Warm-up complete. Buffer size: {len(buffer)}\n")

    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        ts = env.reset(reset_key)
        obs = ts.observation
        done = bool(ts.discount == 0.0)

        ep_reward = 0.0
        ep_actor_loss = 0.0
        ep_critic_loss = 0.0
        train_steps_in_episode = 0

        power_means:     List[float] = []
        bandwidth_means: List[float] = []
        sched_means:     List[float] = []

        while not done:
            key, akey, noise_key, sample_key = jax.random.split(key, 4)

            # --- Select Action with Exploration Noise ---
            action_flat = agent.actor.apply(agent.actor_params, obs.flatten())

            exploration_noise = jax.random.normal(noise_key, shape=action_flat.shape) * action_noise_std
            noisy_action_flat = action_flat + exploration_noise
            clipped_action_flat = jnp.clip(noisy_action_flat, action_spec.minimum, action_spec.maximum)
            action = clipped_action_flat.reshape(agent.num_bs, agent.action_dims_per_bs)

            # --- Record Action Stats ---
            pa = np.array(action[:, 0]); power_means.append(pa.mean())
            ba = np.array(action[:, 1]); bandwidth_means.append(ba.mean())
            ss = np.array(action[:, 2]); sched_means.append(ss.mean())

            # --- Step Environment ---
            ts2 = env.step(action)
            next_obs = ts2.observation
            done     = bool(ts2.discount == 0.0)
            reward   = float(ts2.reward)

            # --- Store Transition in PER Buffer ---
            buffer.add(buffer.Transition(
                obs=obs.flatten(),
                action=action,
                reward=jnp.clip(reward, -1e3, 1e3) / 1e3,
                next_obs=next_obs.flatten(),
                done=ts2.discount
            ))

            obs = next_obs
            ep_reward += reward

            # --- Train the Agent (if buffer is large enough) ---
            if len(buffer) >= batch_size:
                current_beta = min(1.0, per_beta_start + global_step * per_beta_increase)
                batch, batch_indices, batch_is_weights = buffer.sample(batch_size, sample_key, current_beta)

                (agent.actor_params, agent.critic1_params, agent.critic2_params,
                 agent.target_actor_params, agent.target_critic1_params, agent.target_critic2_params,
                 agent.actor_opt_state, agent.critic1_opt_state, agent.critic2_opt_state,
                 actor_loss, critic1_loss, critic2_loss, mean_td_errors, key) = compiled_train_step(
                    # Dynamic Arguments
                    agent.actor_params, agent.critic1_params, agent.critic2_params,
                    agent.target_actor_params, agent.target_critic1_params, agent.target_critic2_params,
                    agent.actor_opt_state, agent.critic1_opt_state, agent.critic2_opt_state,
                    batch, batch_is_weights,
                    key,
                    jnp.array(global_step, dtype=jnp.int32),
                    # Static Arguments (matched by name in static_argnames)
                    actor_apply_fn=agent.actor.apply,
                    critic1_apply_fn=agent.critic1.apply,
                    critic2_apply_fn=agent.critic2.apply,
                    actor_optimizer_update_fn=agent.actor_optimizer.update,
                    critic_optimizer_update_fn=agent.critic_optimizer.update,
                    action_scale=agent.action_scale,
                    action_bias=agent.action_bias,
                    policy_noise=agent.policy_noise,
                    noise_clip=agent.noise_clip,
                    policy_delay=agent.policy_delay,
                    gamma=agent.gamma,
                    tau=agent.tau,
                    num_bs=agent.num_bs,
                    action_dims_per_bs=agent.action_dims_per_bs,
                )

                buffer.update_priorities(np.asarray(batch_indices), np.asarray(mean_td_errors).squeeze())

                ep_actor_loss += float(actor_loss)
                ep_critic_loss += (float(critic1_loss) + float(critic2_loss)) / 2.0
                train_steps_in_episode += 1

                # W&B per-step logs
                if use_wandb:
                    log_dict = {
                        "env/power_adjustments":     wandb.Histogram(pa),
                        "env/bandwidth_allocations": wandb.Histogram(ba),
                        "env/scheduling_scores":     wandb.Histogram(ss),
                        "env/pa_mean":               pa.mean(),
                        "env/ba_mean":               ba.mean(),
                        "env/ss_mean":               ss.mean(),
                        "global_step":               global_step,
                        "train/per_beta":            current_beta,
                        "train/avg_td_error":        float(mean_td_errors.mean()),
                        "buffer_size":               len(buffer),
                        "metrics/reward_per_step":   reward,
                    }
                    if jnp.array(global_step, dtype=jnp.int32) % policy_delay == 0:
                        log_dict["train/actor_loss"] = float(actor_loss)

                    log_dict["train/critic_loss"] = (float(critic1_loss) + float(critic2_loss)) / 2.0
                    wandb.log(log_dict, step=global_step)

            global_step += 1

        # end of episode
        avg_actor_loss = ep_actor_loss / max(train_steps_in_episode // agent.policy_delay, 1)
        avg_critic_loss = ep_critic_loss / max(train_steps_in_episode, 1)

        episode_rewards.append(ep_reward)
        episode_actor_losses.append(avg_actor_loss)
        episode_critic_losses.append(avg_critic_loss)
        episode_powers.append(np.mean(power_means) if power_means else 0.0)
        episode_bandwidths.append(np.mean(bandwidth_means) if bandwidth_means else 0.0)
        episode_scheds.append(np.mean(sched_means) if sched_means else 0.0)

        jax.clear_caches()

        print(f"Episode {ep}: Reward={ep_reward:.2f}, AvgActorLoss={avg_actor_loss:.4f}, AvgCriticLoss={avg_critic_loss:.4f}, Steps={global_step}")

        if use_wandb:
            wandb.log({
                "episode":                ep,
                "total_reward":           ep_reward,
                "episode_avg_actor_loss": avg_actor_loss,
                "episode_avg_critic_loss": avg_critic_loss,
                "mean_power_adjust":      np.mean(power_means) if power_means else 0.0,
                "mean_bandwidth_alloc":   np.mean(bandwidth_means) if bandwidth_means else 0.0,
                "mean_scheduling_score":  np.mean(sched_means) if sched_means else 0.0,
            }, step=global_step)

    if use_wandb:
        wandb.finish()

    return agent, episode_rewards, episode_actor_losses, episode_critic_losses, episode_powers, episode_bandwidths, episode_scheds

