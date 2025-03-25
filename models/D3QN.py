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
from .PPO import *

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
    
    def add(self, transition: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int, key: jax.random.PRNGKey) -> Transition:
        indices = jax.random.choice(key, len(self.buffer), shape=(batch_size,), replace=False)
        batch = [self.buffer[int(idx)] for idx in indices]
        obs = jnp.stack([t.obs for t in batch])
        actions = jnp.stack([t.action for t in batch])
        rewards = jnp.array([t.reward for t in batch])
        next_obs = jnp.stack([t.next_obs for t in batch])
        dones = jnp.array([t.done for t in batch])
        return Transition(obs, actions, rewards, next_obs, dones)

class D3QN:
    def __init__(self, env, gamma=0.99, lr=1e-3, seed=42):
        self.env = env
        self.obs_dim = env.observation_spec().shape[0]
        self.num_bs = env.action_spec().shape[0]
        self.action_shape = env.action_spec().shape
        self.action_dim = int(np.prod(self.action_shape))
        self.gamma = gamma
        
        def q_network(x):
            hidden = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
            ])(x)
            value = hk.Linear(1)(hidden)
            advantages = hk.Linear(self.action_dim)(hidden)
            return value + (advantages - jnp.mean(advantages, axis=-1, keepdims=True))
        
        self.net = hk.without_apply_rng(hk.transform(q_network))
        self.optimizer = optax.adam(lr)
        key = jax.random.PRNGKey(seed)
        dummy_obs = jnp.zeros((self.obs_dim,))
        self.params = self.target_params = self.net.init(key, dummy_obs)
        self.opt_state = self.optimizer.init(self.params)
    
    def _d3qn_loss(self, params, target_params, transitions: Transition):
        q_values = self.net.apply(params, transitions.obs)
        next_q_values = self.net.apply(params, transitions.next_obs)
        next_target_values = self.net.apply(target_params, transitions.next_obs)
        selected_actions = jnp.argmax(next_q_values, axis=-1)
        batch_indices = jnp.arange(transitions.obs.shape[0])
        target_q = transitions.reward + (1 - transitions.done) * self.gamma * next_target_values[batch_indices, selected_actions]
        taken_action = jnp.argmax(transitions.action.reshape(transitions.action.shape[0], -1), axis=-1)
        loss = jnp.mean((q_values[batch_indices, taken_action] - target_q) ** 2)
        return loss
    
    def update(self, batch: Transition):
        loss_value, grads = jax.value_and_grad(self._d3qn_loss)(self.params, self.target_params, batch)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss_value
    
    def update_target_network(self):
        self.target_params = self.params

def train_d3qn(env, num_episodes=100, batch_size=1000, replay_capacity=10000, seed=0):
    agent = D3QN(env)
    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    key = jax.random.PRNGKey(seed)
    episode_rewards = []
    episodes_losses = []
    
    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        ep_reward = 0.0
        ep_loss = 0.0
        
        while state.discount != 0:
            key, action_key = jax.random.split(key)
            epsilon = max(0.01, 0.1 * (0.98 ** episode))
            if jax.random.uniform(action_key) < epsilon:
                action = env.action_spec().generate_value()
            else:
                q_values = agent.net.apply(agent.params, state.observation)
                discrete_index = int(jnp.argmax(q_values))
                action = unflatten_action(discrete_index, env.action_spec().shape)
            
            key, step_key = jax.random.split(key)
            next_state = env.step(action)
            transition = Transition(
                obs=state.observation,
                action=action,
                reward=next_state.reward,
                next_obs=next_state.observation,
                done=1.0 if next_state.discount == 0 else 0.0
            )
            replay_buffer.add(transition)
            ep_reward += next_state.reward
            state = next_state
            
            if len(replay_buffer.buffer) >= batch_size:
                key, sample_key = jax.random.split(key)
                batch = replay_buffer.sample(batch_size, sample_key)
                loss = agent.update(batch)
                agent.update_target_network()
                ep_loss += loss
        episode_rewards.append(ep_reward)
        episodes_losses.append(ep_loss)
        print(f"D3QN Episode {episode}: Total Reward = {ep_reward:.2f}, Loss = {ep_loss:.4f}")
    
    return agent, episode_rewards, episodes_losses


# -------------------------
# Enhanced Evaluation with Trajectory Visualization
# -------------------------
@dataclass
class EvaluationResult:
    episode_rewards: List[float]
    states: List[jnp.ndarray]
    actions: List[jnp.ndarray]
    metrics: Dict[str, List]

def evaluate_agent(
    env: HetNetEnvironment,
    params: Dict,
    num_episodes: int = 5,
    is_ppo: bool = True,
    max_steps: int = 100,
    network=None
) -> EvaluationResult:
    """Evaluate an agent with detailed trajectory tracking.
    
    For PPO evaluation, if no network is provided, a new network is created based on ppo_network_fn.
    For D3QN evaluation, you must pass in the same network (e.g., d3qn_agent.net) used during training.
    """
    result = EvaluationResult([], [], [], {'throughput': [], 'handovers': []})
    
    if network is None:
        if is_ppo:
            def forward_fn(obs):
                logits, _ = ppo_network_fn(obs, int(np.prod(env.action_spec().shape)), 64)
                return logits
            network = hk.without_apply_rng(hk.transform(forward_fn))
        else:
            raise ValueError("For D3QN evaluation, please provide the training network (e.g., d3qn_agent.net).")
    
    for ep in range(num_episodes):
        key = jax.random.PRNGKey(ep)
        state = env.reset(key)
        episode_states = []
        episode_actions = []
        episode_reward = 0.0
        
        for step in range(max_steps):
            logits_or_q = network.apply(params, state.observation)
            # Reshape output to (num_bs, 3)
            logits_or_q = logits_or_q.reshape(env.action_spec().shape)
            # For each BS, choose the action with highest score.
            action_idx = jnp.argmax(logits_or_q, axis=1)  # shape: (num_bs,)
            action = jax.nn.one_hot(action_idx, env.action_spec().shape[1])  # shape: (num_bs, 3)
            
            episode_states.append(state.observation)
            episode_actions.append(action)
            next_state = env.step(action)
            episode_reward += next_state.reward
            
            # Extract environment metrics (fallback if methods not defined).
            throughput = env.get_current_throughput() if hasattr(env, 'get_current_throughput') else state.reward
            handovers = env.get_handover_count() if hasattr(env, 'get_handover_count') else 0
            result.metrics['throughput'].append(throughput)
            result.metrics['handovers'].append(handovers)
            
            if next_state.discount == 0:
                break
            state = next_state
        
        result.episode_rewards.append(episode_reward)
        result.states.extend(episode_states)
        result.actions.extend(episode_actions)
    
    return result
