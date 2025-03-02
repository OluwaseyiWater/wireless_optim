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

# Create directory for plots
os.makedirs("plots", exist_ok=True)

# -------------------------
# Environment Import
# -------------------------
from wireless_optim.environment import HetNetEnvironment, Transition

# -------------------------
# Utility: Unflatten Action
# -------------------------
# def unflatten_action(index: int, shape):
#     total = int(np.prod(shape))
#     one_hot = jnp.zeros(total)
#     one_hot = one_hot.at[index].set(1.0)
#     return one_hot.reshape(shape)

# -------------------------ppo_train
# PPO Implementation (Training)
# -------------------------
def compute_gae(rewards, values, dones, gamma, lam):
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

def ppo_network_fn(obs, action_dim, hidden_size):
    mlp = hk.Sequential([
        hk.Linear(hidden_size), jax.nn.tanh,
        hk.Linear(hidden_size), jax.nn.tanh,
    ])
    hidden = mlp(obs)
    logits = hk.Linear(action_dim)(hidden)
    value = hk.Linear(1)(hidden)
    return logits, jnp.squeeze(value, axis=-1)

def ppo_train(env, config,seed=0):
    action_shape = env.action_spec().shape
    action_dim = int(np.prod(action_shape))
    obs_dim = env.observation_spec().shape[0]
    hidden_size = config['hidden_size']

    def forward_fn(obs):
        return ppo_network_fn(obs, action_dim, hidden_size)
    ppo_net = hk.without_apply_rng(hk.transform(forward_fn))
    key = jax.random.PRNGKey(seed)
    dummy_obs = jnp.zeros((obs_dim,))
    params = ppo_net.init(key, dummy_obs)
    optimizer = optax.adam(config['lr'])
    opt_state = optimizer.init(params)

    epoch_rewards = []
    epoch_losses = []

    for epoch in range(config['num_epochs']):
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        key, subkey = jax.random.split(key)
        state = env.reset(subkey)
        ep_reward = 0.0

        for step in range(config['num_steps']):
            obs = state.observation
            observations.append(obs)
            logits, value = ppo_net.apply(params, obs)
            values.append(np.array(value))
            probs = jax.nn.softmax(logits)
            key, subkey = jax.random.split(key)
            action = int(jax.random.categorical(subkey, logits))
            actions.append(action)
            log_prob = jnp.log(probs[action] + 1e-8)
            log_probs.append(float(log_prob))
            action_one_hot = unflatten_action(action, action_shape)
            state = env.step(action_one_hot)
            rewards.append(state.reward)
            dones.append(1.0 if state.discount == 0 else 0.0)
            ep_reward += state.reward
            if state.discount == 0:
                break

        _, last_value = ppo_net.apply(params, state.observation)
        values.append(np.array(last_value))

        observations = jnp.array(observations)
        actions = jnp.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        log_probs = jnp.array(log_probs)
        values = np.array(values)

        advantages, returns = compute_gae(rewards, values, dones, config['gamma'], config['gae_lambda'])
        advantages = jnp.array(advantages)
        returns = jnp.array(returns)

        def ppo_loss(params, observations, actions, old_log_probs, advantages, returns):
            logits, value_pred = jax.vmap(lambda obs: ppo_net.apply(params, obs))(observations)
            new_log_probs = jax.vmap(lambda logit, act: jnp.log(jax.nn.softmax(logit)[act] + 1e-8))(logits, actions)
            ratio = jnp.exp(new_log_probs - old_log_probs)
            clipped_ratio = jnp.clip(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
            policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
            value_loss = jnp.mean((returns - value_pred) ** 2)
            entropy = jnp.mean(-jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1))
            total_loss = policy_loss + config['vf_coef'] * value_loss - config['ent_coef'] * entropy
            return total_loss

        for _ in range(config['update_epochs']):
            loss, grads = jax.value_and_grad(ppo_loss)(params, observations, actions, log_probs, advantages, returns)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        epoch_rewards.append(ep_reward)
        epoch_losses.append(loss)
        print(f"PPO Epoch {epoch}: Reward = {ep_reward:.2f}, Loss = {loss:.4f}")

    return params, ppo_net, epoch_rewards, epoch_losses

def ppo_agent(config):
    env = HetNetEnvironment(**config)
    obs = env.reset().observation
    
    return agent
# -------------------------
# D3QN Implementation (Training)
# -------------------------
# @dataclass
# class Transition:
#     obs: jnp.ndarray
#     action: jnp.ndarray
#     reward: float
#     next_obs: jnp.ndarray
#     done: float

# class ReplayBuffer:
#     def __init__(self, capacity: int):
#         self.capacity = capacity
#         self.buffer: List[Transition] = []
    
#     def add(self, transition: Transition):
#         if len(self.buffer) >= self.capacity:
#             self.buffer.pop(0)
#         self.buffer.append(transition)
    
#     def sample(self, batch_size: int, key: jax.random.PRNGKey) -> Transition:
#         indices = jax.random.choice(key, len(self.buffer), shape=(batch_size,), replace=False)
#         batch = [self.buffer[int(idx)] for idx in indices]
#         obs = jnp.stack([t.obs for t in batch])
#         actions = jnp.stack([t.action for t in batch])
#         rewards = jnp.array([t.reward for t in batch])
#         next_obs = jnp.stack([t.next_obs for t in batch])
#         dones = jnp.array([t.done for t in batch])
#         return Transition(obs, actions, rewards, next_obs, dones)

# class D3QN:
#     def __init__(self, env, gamma=0.99, lr=1e-3):
#         self.env = env
#         self.obs_dim = env.observation_spec().shape[0]
#         self.num_bs = env.action_spec().shape[0]
#         self.action_shape = env.action_spec().shape
#         self.action_dim = int(np.prod(self.action_shape))
#         self.gamma = gamma
        
#         def q_network(x):
#             hidden = hk.Sequential([
#                 hk.Linear(64), jax.nn.relu,
#                 hk.Linear(64), jax.nn.relu,
#             ])(x)
#             value = hk.Linear(1)(hidden)
#             advantages = hk.Linear(self.action_dim)(hidden)
#             return value + (advantages - jnp.mean(advantages, axis=-1, keepdims=True))
        
#         self.net = hk.without_apply_rng(hk.transform(q_network))
#         self.optimizer = optax.adam(lr)
#         key = jax.random.PRNGKey(42)
#         dummy_obs = jnp.zeros((self.obs_dim,))
#         self.params = self.target_params = self.net.init(key, dummy_obs)
#         self.opt_state = self.optimizer.init(self.params)
    
#     def _d3qn_loss(self, params, target_params, transitions: Transition):
#         q_values = self.net.apply(params, transitions.obs)
#         next_q_values = self.net.apply(params, transitions.next_obs)
#         next_target_values = self.net.apply(target_params, transitions.next_obs)
#         selected_actions = jnp.argmax(next_q_values, axis=-1)
#         batch_indices = jnp.arange(transitions.obs.shape[0])
#         target_q = transitions.reward + (1 - transitions.done) * self.gamma * next_target_values[batch_indices, selected_actions]
#         taken_action = jnp.argmax(transitions.action.reshape(transitions.action.shape[0], -1), axis=-1)
#         loss = jnp.mean((q_values[batch_indices, taken_action] - target_q) ** 2)
#         return loss
    
#     def update(self, batch: Transition):
#         loss_value, grads = jax.value_and_grad(self._d3qn_loss)(self.params, self.target_params, batch)
#         updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
#         self.params = optax.apply_updates(self.params, updates)
#         return loss_value
    
#     def update_target_network(self):
#         self.target_params = self.params

# def train_d3qn(env, num_episodes=100, batch_size=64, replay_capacity=10000):
#     agent = D3QN(env)
#     replay_buffer = ReplayBuffer(capacity=replay_capacity)
#     key = jax.random.PRNGKey(0)
#     episode_rewards = []
    
#     for episode in range(num_episodes):
#         key, reset_key = jax.random.split(key)
#         state = env.reset(reset_key)
#         ep_reward = 0.0
        
#         while state.discount != 0:
#             key, action_key = jax.random.split(key)
#             epsilon = max(0.01, 0.1 * (0.98 ** episode))
#             if jax.random.uniform(action_key) < epsilon:
#                 action = env.action_spec().generate_value()
#             else:
#                 q_values = agent.net.apply(agent.params, state.observation)
#                 discrete_index = int(jnp.argmax(q_values))
#                 action = unflatten_action(discrete_index, env.action_spec().shape)
            
#             key, step_key = jax.random.split(key)
#             next_state = env.step(action)

# class Transition:
#     obs: jnp.ndarray
#     action: jnp.ndarray
#     reward: float
#     next_obs: jnp.ndarray
#     done: float
#             transition = Transition(
#                 obs=state.observation,
#                 action=action,
#                 reward=next_state.reward,
#                 next_obs=next_state.observation,
#                 done=1.0 if next_state.discount == 0 else 0.0
#             )
#             replay_buffer.add(transition)
#             ep_reward += next_state.reward
#             state = next_state
            
#             if len(replay_buffer.buffer) >= batch_size:
#                 key, sample_key = jax.random.split(key)
#                 batch = replay_buffer.sample(batch_size, sample_key)
#                 loss = agent.update(batch)
#                 agent.update_target_network()
#         episode_rewards.append(ep_reward)
#         print(f"D3QN Episode {episode}: Total Reward = {ep_reward:.2f}")
    
#     return agent, episode_rewards

# # -------------------------
# # Enhanced Evaluation with Trajectory Visualization
# # -------------------------
# @dataclass
# class EvaluationResult:
#     episode_rewards: List[float]
#     states: List[jnp.ndarray]
#     actions: List[jnp.ndarray]
#     metrics: Dict[str, List]

# def evaluate_agent(
#     env: HetNetEnvironment,
#     params: Dict,
#     num_episodes: int = 5,
#     is_ppo: bool = True,
#     max_steps: int = 100,
#     network=None
# ) -> EvaluationResult:
#     """Evaluate an agent with detailed trajectory tracking.
    
#     For PPO evaluation, if no network is provided, a new network is created based on ppo_network_fn.
#     For D3QN evaluation, you must pass in the same network (e.g., d3qn_agent.net) used during training.
#     """
#     result = EvaluationResult([], [], [], {'throughput': [], 'handovers': []})
    
#     if network is None:
#         if is_ppo:
#             def forward_fn(obs):
#                 logits, _ = ppo_network_fn(obs, int(np.prod(env.action_spec().shape)), 64)
#                 return logits
#             network = hk.without_apply_rng(hk.transform(forward_fn))
#         else:
#             raise ValueError("For D3QN evaluation, please provide the training network (e.g., d3qn_agent.net).")
    
#     for ep in range(num_episodes):
#         key = jax.random.PRNGKey(ep)
#         state = env.reset(key)
#         episode_states = []
#         episode_actions = []
#         episode_reward = 0.0
        
#         for step in range(max_steps):
#             logits_or_q = network.apply(params, state.observation)
#             # Reshape output to (num_bs, 3)
#             logits_or_q = logits_or_q.reshape(env.action_spec().shape)
#             # For each BS, choose the action with highest score.
#             action_idx = jnp.argmax(logits_or_q, axis=1)  # shape: (num_bs,)
#             action = jax.nn.one_hot(action_idx, env.action_spec().shape[1])  # shape: (num_bs, 3)
            
#             episode_states.append(state.observation)
#             episode_actions.append(action)
#             next_state = env.step(action)
#             episode_reward += next_state.reward
            
#             # Extract environment metrics (fallback if methods not defined).
#             throughput = env.get_current_throughput() if hasattr(env, 'get_current_throughput') else state.reward
#             handovers = env.get_handover_count() if hasattr(env, 'get_handover_count') else 0
#             result.metrics['throughput'].append(throughput)
#             result.metrics['handovers'].append(handovers)
            
#             if next_state.discount == 0:
#                 break
#             state = next_state
        
#         result.episode_rewards.append(episode_reward)
#         result.states.extend(episode_states)
#         result.actions.extend(episode_actions)
    
#     return result

# def plot_terrain(env: HetNetEnvironment, 
#                  states: List[jnp.ndarray], 
#                  actions: List[jnp.ndarray],
#                  episode: int = 0):
#     """Visualize agent decisions on network terrain"""
#     plt.figure(figsize=(12, 8))
    
#     # Use environment's method to get macro BS positions.
#     macro_positions = env._grid_positions(env.num_macro_bs, scale=1.0)
#     plt.scatter(macro_positions[:, 0], macro_positions[:, 1], 
#                 c='red', s=100, label='Macro BS')
#     # For small BS, generate dummy positions (or use attributes if available).
#     key = jax.random.PRNGKey(1)
#     small_bs = jax.random.uniform(key, (env.num_small_bs, 2), minval=0.0, maxval=1.0)
#     plt.scatter(small_bs[:, 0], small_bs[:, 1], 
#                 c='blue', s=50, label='Small BS')
    
#     # Extract user positions from the last state's observation.
#     # This assumes the observation is structured as in env._get_observations().
#     total_bs = env.num_macro_bs + env.num_small_bs
#     obs = states[-1]
#     # For illustration, assume the last 2*(total_bs+env.num_users) entries are positions.
#     positions = obs[-2*(total_bs+env.num_users):].reshape(-1, 2)
#     user_locs = positions[-env.num_users:]
#     plt.scatter(user_locs[:, 0], user_locs[:, 1], 
#                 c='green', s=10, alpha=0.5, label='Users')
    
#     # Plot actions as vectors for first 10 steps.
#     for t, action in enumerate(actions[:10]):
#         for bs_idx in range(action.shape[0]):
#             action_type = jnp.argmax(action[bs_idx])
#             color = 'purple' if action_type == 0 else 'orange' if action_type == 1 else 'cyan'
#             if bs_idx < env.num_macro_bs:
#                 bs_loc = macro_positions[bs_idx]
#             else:
#                 bs_loc = small_bs[bs_idx - env.num_macro_bs]
#             plt.arrow(bs_loc[0], bs_loc[1], 
#                       0.1 * (t+1), 0.1 * (t+1),
#                       color=color, width=0.01, alpha=0.3)
    
#     plt.title(f"Network State and Agent Actions (Episode {episode})")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f'terrain_episode_{episode}.png')
#     plt.close()

# def plot_action_distribution(ppo_actions: List[jnp.ndarray], 
#                              d3qn_actions: List[jnp.ndarray]):
#     """Compare action choices between agents"""
#     plt.figure(figsize=(10, 6))
    
#     ppo_flat = jnp.concatenate([jnp.argmax(a, axis=-1) for a in ppo_actions])
#     d3qn_flat = jnp.concatenate([jnp.argmax(a, axis=-1) for a in d3qn_actions])
    
#     for idx, (name, actions) in enumerate(zip(['PPO', 'D3QN'], [ppo_flat, d3qn_flat])):
#         counts = jnp.bincount(actions, length=3)
#         plt.bar(jnp.arange(3) + idx*0.3, counts, width=0.3, label=name)
    
#     plt.title("Action Type Distribution")
#     plt.xlabel("Action Type")
#     plt.ylabel("Count")
#     plt.xticks([0, 1, 2], ['Power Down', 'Maintain', 'Power Up'])
#     plt.legend()
#     plt.savefig('action_distribution.png')
#     plt.close()

# def plot_metrics_comparison(ppo_metrics: Dict, d3qn_metrics: Dict):
#     """Plot performance metrics comparison"""
#     fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
#     ppo_throughput = jnp.convolve(jnp.array(ppo_metrics['throughput']), jnp.ones(10)/10, mode='valid')
#     d3qn_throughput = jnp.convolve(jnp.array(d3qn_metrics['throughput']), jnp.ones(10)/10, mode='valid')
#     axs[0].plot(ppo_throughput, label='PPO')
#     axs[0].plot(d3qn_throughput, label='D3QN')
#     axs[0].set_title("Average Throughput (Moving Window)")
#     axs[0].set_ylabel("Mbps")
#     axs[0].legend()
    
#     axs[1].plot(jnp.cumsum(jnp.array(ppo_metrics['handovers'])), label='PPO')
#     axs[1].plot(jnp.cumsum(jnp.array(d3qn_metrics['handovers'])), label='D3QN')
#     axs[1].set_title("Cumulative Handovers")
#     axs[1].set_ylabel("Count")
#     axs[1].legend()
    
#     plt.tight_layout()
#     plt.savefig('metrics_comparison.png')
#     plt.close()

# -------------------------
# Updated Main Function
# -------------------------
# def main():
#     env = HetNetEnvironment(num_macro_bs=3, num_small_bs=10, num_users=50, max_steps=100)
    
#     # PPO Configuration.
#     ppo_config = {
#         'num_envs': 1,
#         'num_steps': 50,
#         'num_epochs': 10,
#         'lr': 3e-4,
#         'anneal_lr': True,
#         'gamma': 0.99,
#         'gae_lambda': 0.95,
#         'clip_coef': 0.2,
#         'ent_coef': 0.01,
#         'vf_coef': 0.5,
#         'max_grad_norm': 0.5,
#         'update_epochs': 4,
#         'hidden_size': 64,
#     }
    
#     print("Starting PPO training...")
#     ppo_params, ppo_net, ppo_rewards, ppo_losses = ppo_train(env, ppo_config)
#     print("PPO training completed.")
    
#     print("\nStarting D3QN training...")
#     d3qn_agent, d3qn_rewards = train_d3qn(env, num_episodes=10)
#     print("D3QN training completed.")
    
#     # Save training plots.
#     plt.figure(figsize=(10, 5))
#     plt.plot(ppo_rewards, label='PPO Rewards')
#     plt.plot(d3qn_rewards, label='D3QN Rewards')
#     plt.xlabel('Epoch/Episode')
#     plt.ylabel('Total Reward')
#     plt.title('Training Progress Comparison')
#     plt.legend()
#     plt.savefig("plots/training_progress_comparison.png")
#     plt.close()
    
#     if ppo_losses:
#         plt.figure(figsize=(10, 5))
#         plt.plot(ppo_losses, label='PPO Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('PPO Loss over Training')
#         plt.legend()
#         plt.savefig("plots/ppo_loss_over_training.png")
#         plt.close()
    
#     # Enhanced Evaluation.
#     print("\nEvaluating PPO...")
#     ppo_eval = evaluate_agent(env, ppo_params, is_ppo=True)
#     print(f"PPO Average Reward: {np.mean(ppo_eval.episode_rewards):.2f}")
    
#     print("\nEvaluating D3QN...")
#     # For D3QN, pass the agent's network to ensure matching architecture.
#     d3qn_eval = evaluate_agent(env, d3qn_agent.params, is_ppo=False, network=d3qn_agent.net)
#     print(f"D3QN Average Reward: {np.mean(d3qn_eval.episode_rewards):.2f}")
    
#     # Visualization.
#     plot_action_distribution(ppo_eval.actions, d3qn_eval.actions)
#     plot_metrics_comparison(ppo_eval.metrics, d3qn_eval.metrics)
    
#     # Plot terrain visualization for evaluation episodes.
#     plot_terrain(env, ppo_eval.states[:100], ppo_eval.actions[:100], episode=0)
#     plot_terrain(env, d3qn_eval.states[:100], d3qn_eval.actions[:100], episode=1)
    
#     # Plot cumulative evaluation rewards.
#     plt.figure(figsize=(10, 6))
#     plt.plot(np.cumsum(ppo_eval.episode_rewards), label='PPO')
#     plt.plot(np.cumsum(d3qn_eval.episode_rewards), label='D3QN')
#     plt.title("Cumulative Evaluation Rewards")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.legend()
#     plt.savefig('evaluation_rewards.png')
#     plt.close()

# if __name__ == "__main__":
#     main()