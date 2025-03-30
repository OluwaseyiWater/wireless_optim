# import jax
# import jax.numpy as jnp
# import numpy as np
# import haiku as hk
# import optax
# import matplotlib.pyplot as plt
# from dataclasses import dataclass
# from typing import List, Dict
# import os
# from utils import unflatten_action

# from wireless_optim.environment import *
# from .PPO import *

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
#     def __init__(self, env, gamma=0.96, lr=1e-5, seed=42):
#         self.env = env
#         self.obs_dim = env.observation_spec().shape[0]
#         self.num_bs = env.action_spec().shape[0]
#         self.action_shape = env.action_spec().shape
#         self.action_dim = int(np.prod(self.action_shape))
#         self.gamma = gamma
        
#         def q_network(x):
#             hidden = hk.Sequential([
#                 hk.Linear(128), jax.nn.relu,
#                 hk.Linear(128), jax.nn.relu,
#                 hk.Linear(128), jax.nn.relu,
#                 hk.Linear(128), jax.nn.relu,
#                 hk.Linear(128), jax.nn.relu,
#                 hk.Linear(128), jax.nn.relu,
#                 hk.Linear(128), jax.nn.relu,
#                 hk.Linear(64), jax.nn.relu,
#                 hk.Linear(64), jax.nn.relu,
#                 hk.Linear(64), jax.nn.relu,
#                 hk.Linear(64), jax.nn.relu,
#                 hk.Linear(64), jax.nn.relu,
#                 hk.Linear(32), jax.nn.relu,
#                 hk.Linear(32), jax.nn.relu,
#                 hk.Linear(32), jax.nn.relu,
#                 hk.Linear(32), jax.nn.relu,
#                 hk.Linear(32), jax.nn.relu,
#             ])(x)
#             value = hk.Linear(1)(hidden)
#             advantages = hk.Linear(self.action_dim)(hidden)
#             return value + (advantages - jnp.mean(advantages, axis=-1, keepdims=True))
        
#         self.net = hk.without_apply_rng(hk.transform(q_network))
#         self.optimizer = optax.adam(lr)
#         key = jax.random.PRNGKey(seed)
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

# def train_d3qn(env, num_episodes=100, batch_size=1000, replay_capacity=10000, seed=0):
#     agent = D3QN(env)
#     replay_buffer = ReplayBuffer(capacity=replay_capacity)
#     key = jax.random.PRNGKey(seed)
#     episode_rewards = []
#     episodes_losses = []
    
#     for episode in range(num_episodes):
#         key, reset_key = jax.random.split(key)
#         state = env.reset(reset_key)
#         ep_reward = 0.0
#         ep_loss = 0.0
        
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
#                 ep_loss += loss
#         episode_rewards.append(ep_reward)
#         episodes_losses.append(ep_loss)
#         print(f"D3QN Episode {episode}: Total Reward = {ep_reward:.2f}, Loss = {ep_loss:.4f}")
    
#     return agent, episode_rewards, episodes_losses


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


#%%writefile D3QN.py
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

from wireless_optim.environment import *


# Define a Transition dataclass to hold experience data
@dataclass
class Transition:
    obs: jnp.ndarray          # Current observation
    action_indices: jnp.ndarray  # Discrete action indices for each base station (BS)
    reward: float             # Reward received
    next_obs: jnp.ndarray     # Next observation
    done: float               # Done flag (1.0 if terminal, 0.0 otherwise)

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
        action_indices = jnp.stack([t.action_indices for t in batch])
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
            optax.clip_by_global_norm(0.3),  # Gradient clipping to stabilize training
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
        q_values = self.net.apply(params, transitions.obs)  # (batch_size, num_bs, num_actions_per_bs)
        next_target_values = self.net.apply(target_params, transitions.next_obs)  # (batch_size, num_bs, num_actions_per_bs)
        batch_indices = jnp.arange(transitions.obs.shape[0])

        # Q-values for selected actions, summed across BSs
        q_selected = jnp.sum(
            q_values[batch_indices, jnp.arange(self.num_bs), transitions.action_indices], 
            axis=1
        )  # (batch_size,)

        # Max Q-values for next state, summed across BSs
        max_next_q = jnp.sum(jnp.max(next_target_values, axis=-1), axis=1)  # (batch_size,)

        # Target Q-value using the Double DQN approach
        target_q = transitions.reward + (1 - transitions.done) * self.gamma * max_next_q  # (batch_size,)

        # Mean squared error loss
        loss = jnp.mean((q_selected - target_q) ** 2)
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

def train_d3qn(env, num_episodes=100, batch_size=256, replay_capacity=10000, seed=0):
    """
    Train the D3QN agent in the given environment.

    Args:
        env: Environment instance (e.g., HetNetEnvironment).
        num_episodes (int): Number of training episodes.
        batch_size (int): Size of the batch sampled from the replay buffer.
        replay_capacity (int): Capacity of the replay buffer.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (trained agent, list of episode rewards, list of episode losses).
    """
    # obs_dim = env.observation_spec().shape[0]
    obs_dim = int(np.prod(env.observation_spec().shape))  # Calculate flattened observation dimension
    num_bs = env.action_spec().shape[0]  # e.g., 13
    # agent = D3QN(obs_dim, num_bs, lr=3e-4, gamma=0.99)
    agent = D3QN(obs_dim, num_bs, lr=3e-4, gamma=0.99)
    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    key = jax.random.PRNGKey(seed)
    episode_rewards = []
    episode_losses = []
    
    print("Starting D3QN training...", flush=True)

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        ep_reward = 0.0
        ep_loss = 0.0
        done = False

        while not done:
            key, action_key = jax.random.split(key)
            # Epsilon-greedy exploration
            epsilon = max(0.01, 0.1 * (0.98 ** episode))
            if jax.random.uniform(action_key) < epsilon:
                # Random action indices for each BS
                action_indices = jax.random.randint(action_key, (num_bs,), 0, agent.num_actions_per_bs)
            else:
                # Greedy action selection
                q_values = agent.net.apply(agent.params, state.observation)  # (num_bs, num_actions_per_bs)
                action_indices = jnp.argmax(q_values, axis=-1)  # (num_bs,)

            # Convert discrete indices to continuous actions
            action = indices_to_actions(action_indices, num_bs)

            # Step the environment
            next_state = env.step(action)
            scaled_reward = next_state.reward / 100.0  # Scale reward to stabilize Q-values

            current_obs = state.observation.flatten()  # Flatten the observation
            next_obs = next_state.observation.flatten()

            transition = Transition(
                obs=current_obs,
                action_indices=action_indices,
                reward=scaled_reward,
                next_obs=next_obs,
                done=1.0 if next_state.discount == 0 else 0.0
            )
            
            replay_buffer.add(transition)
            ep_reward += next_state.reward
            state = next_state
            done = next_state.discount == 0

            # Update the agent if enough data is available
            if len(replay_buffer.buffer) >= batch_size:
                key, sample_key = jax.random.split(key)
                batch = replay_buffer.sample(batch_size, sample_key)
                loss = agent.update(batch)
                ep_loss += loss

        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        episode_rewards.append(ep_reward)
        episode_losses.append(ep_loss)
        print(f"D3QN Episode {episode}: Total Reward = {ep_reward:.2f}, Loss = {ep_loss:.4f}")

    return agent, episode_rewards, episode_losses

