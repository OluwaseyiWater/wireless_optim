import jax
import jax.numpy as jnp
import optax
import haiku as hk
import numpy as np
from dataclasses import dataclass
from typing import List

from jumanji.env import Environment
from purejaxrl.ppo import train as ppo_train
from RLax import dqn_loss 

from wireless_optim.environment import *


@dataclass
class Transition:
    obs: jnp.ndarray
    action: int
    reward: float
    next_obs: jnp.ndarray
    done: float

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
    
    def add(self, transition: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int, key: jax.random.PRNGKey) -> Transition:
        # Sample indices without replacement
        indices = jax.random.choice(key, len(self.buffer), shape=(batch_size,), replace=False)
        batch = [self.buffer[int(idx)] for idx in indices]
        obs = jnp.stack([t.obs for t in batch])
        actions = jnp.array([t.action for t in batch])
        rewards = jnp.array([t.reward for t in batch])
        next_obs = jnp.stack([t.next_obs for t in batch])
        dones = jnp.array([t.done for t in batch])
        return Transition(obs, actions, rewards, next_obs, dones)

class D3QN:
    """Dueling Double Deep Q-Network for HetNet"""
    def __init__(self, env: Environment, gamma=0.99, lr=1e-3):
        self.env = env
        self.obs_dim = env.observation_spec().shape[0]
        # Flatten the action space (number of BS x number of actions per BS)
        self.action_dim = env.action_spec().shape[0] * env.action_spec().shape[1]
        self.gamma = gamma
        
        def q_network(x):
            hidden = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(64), jax.nn.relu,
            ])(x)
            # Dueling architecture: compute value and advantage
            value = hk.Linear(1)(hidden)
            advantages = hk.Linear(self.action_dim)(hidden)
            return value + (advantages - jnp.mean(advantages, axis=-1, keepdims=True))
        
        self.net = hk.without_apply_rng(hk.transform(q_network))
        self.optimizer = optax.adam(lr)
        
        # Initialize network parameters
        key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros((self.obs_dim,))
        self.params = self.target_params = self.net.init(key, dummy_obs)
        self.opt_state = self.optimizer.init(self.params)
    
    def _d3qn_loss(self, params, target_params, transitions: Transition):
        q_values = self.net.apply(params, transitions.obs)
        next_q_values = self.net.apply(params, transitions.next_obs)
        next_target_values = self.net.apply(target_params, transitions.next_obs)
        
        # Double DQN: select best actions using current network, evaluate with target network
        selected_actions = jnp.argmax(next_q_values, axis=-1)
        batch_indices = jnp.arange(transitions.obs.shape[0])
        target_q = transitions.reward + (1 - transitions.done) * self.gamma * next_target_values[batch_indices, selected_actions]
        
        q_taken = q_values[batch_indices, transitions.action]
        loss = jnp.mean((q_taken - target_q) ** 2)
        return loss
    
    def update(self, batch: Transition):
        loss_value, grads = jax.value_and_grad(self._d3qn_loss)(self.params, self.target_params, batch)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss_value
    
    def update_target_network(self):
        self.target_params = self.params



def train_d3qn(env, num_episodes=1000, batch_size=64, replay_capacity=10000):
    agent = D3QN(env)
    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    
    key = jax.random.PRNGKey(0)
    best_reward = -jnp.inf
    
    for episode in range(num_episodes):
        # Reset environment with a new key split
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        episode_reward = 0.0
        
        while not state.done:
            # Epsilon-greedy exploration schedule
            epsilon = max(0.01, 0.1 * (0.98 ** episode))
            key, action_key = jax.random.split(key)
            if jax.random.uniform(action_key) < epsilon:
                # Random action generation
                action = env.action_spec().generate_value()
                # Assume action can be flattened to an integer index for simplicity
                action = int(action)
            else:
                # Greedy action selection based on Q-values
                q_values = agent.net.apply(agent.params, state.observation)
                action = int(jnp.argmax(q_values))
            
            key, step_key = jax.random.split(key)
            next_state = env.step(state, action)
            
            # Store transition in replay buffer
            transition = Transition(
                obs=state.observation,
                action=action,
                reward=next_state.reward,
                next_obs=next_state.observation,
                done=1.0 if next_state.done else 0.0
            )
            replay_buffer.add(transition)
            
            episode_reward += next_state.reward
            state = next_state
            
            # If sufficient samples are in buffer, perform an update
            if len(replay_buffer.buffer) >= batch_size:
                key, sample_key = jax.random.split(key)
                batch = replay_buffer.sample(batch_size, sample_key)
                loss = agent.update(batch)
                # Optionally, update the target network every update (or less frequently)
                agent.update_target_network()
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            # Optionally, save best model parameters
        
        print(f"Episode {episode}: Reward {episode_reward:.2f}, Loss {loss:.4f}")
    
    return agent
