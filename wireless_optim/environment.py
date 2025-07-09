import jax
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, transition, termination
from dataclasses import dataclass

class HetNetEnvironment(Environment):
    def __init__(self, num_macro_bs=3, num_small_bs=10, num_users=50, max_steps=100):
        super().__init__()
        self.num_macro_bs = num_macro_bs
        self.num_small_bs = num_small_bs
        self.num_users = num_users
        self.max_steps = max_steps
        self.num_bs = self.num_macro_bs + self.num_small_bs

        # Network parameters
        self.total_bandwidth = 100e6  # 100 MHz
        self.max_power = 40.0  # dBm
        self.carrier_freq = 2.4e9  # 2.4 GHz
        self.path_loss_exponent = 3.5
        self.shadow_std_db = 4.0
        self.noise_floor_dbm = -100.0

        action_size = self.num_bs + self.num_bs + (self.num_bs * self.num_users)
        self._action_spec = specs.BoundedArray(
            shape=(action_size,),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0, 
            name="action"
        )
        self._state = None

    def reset(self, key: jax.random.PRNGKey) -> TimeStep:
        """Reset the environment to initial state with improved BS and user placement."""
        keys = jax.random.split(key, 3)

        macro_positions = self._grid_positions(self.num_macro_bs, scale=1.0)
        small_positions = jax.random.uniform(keys[0], (self.num_small_bs, 2), minval=0.0, maxval=1.0)
        bs_positions = jnp.concatenate([macro_positions, small_positions], axis=0)
        user_positions = jax.random.uniform(keys[1], (self.num_users, 2), minval=0.0, maxval=1.0)
        self._state = {
            'bs_positions': bs_positions,
            'user_positions': user_positions,
            'resource_allocations': jnp.zeros((self.num_bs, self.num_users)), 
            'power_levels': jnp.full((self.num_bs,), 30.0),
            'interference': jnp.zeros((self.num_users,)),
            'step_count': 0,
            'key': keys[2] 
        }

        obs = self._get_observations(self._state)
        return restart(observation=obs)

    def step(self, action: jnp.ndarray) -> TimeStep:
        power_action = action[:self.num_bs]
        bw_action = action[self.num_bs : 2 * self.num_bs]
        sched_action = action[2 * self.num_bs:].reshape(self.num_bs, self.num_users)
        power_change_db = (power_action - 0.5) * 10.0
        new_power = jnp.clip(self._state['power_levels'] + power_change_db, 0.0, self.max_power)
        step_key, new_key = jax.random.split(self._state['key'])
        distances = jnp.linalg.norm(
            self._state['bs_positions'][:, None] - self._state['user_positions'][None, :], axis=-1
        )
        path_loss_db = self._calculate_path_loss_db(distances, key=step_key)
        interference_linear = self._calculate_interference(self._state, new_power, path_loss_db)
        sinr_db, serving_bs_indices = self._calculate_sinr(self._state, new_power, interference_linear, path_loss_db)
        new_allocations = self._update_allocations(
            self._state['resource_allocations'],
            bw_action, 
            sched_action
        )

        user_throughputs = self._calculate_throughput(new_allocations, sinr_db, serving_bs_indices)
        reward = self._calculate_reward(user_throughputs, new_power)
        self._state = {
            'bs_positions': self._state['bs_positions'],
            'user_positions': self._state['user_positions'],
            'power_levels': new_power,
            'resource_allocations': new_allocations,
            'interference': interference_linear, 
            'step_count': self._state['step_count'] + 1,
            'key': new_key
        }

        obs = self._get_observations(self._state)
        done = self._state['step_count'] >= self.max_steps
        
        if done:
            return termination(reward=reward, observation=obs)
        else:
            return transition(reward=reward, observation=obs)

    def _grid_positions(self, num_points, scale=1.0):
        n = int(jnp.ceil(jnp.sqrt(num_points)))
        lin = jnp.linspace(0.1, 0.9, n)
        grid = jnp.array(jnp.meshgrid(lin, lin)).T.reshape(-1, 2)
        return grid[:num_points] * scale

    def _update_allocations(self, current_allocations, bandwidth_allocations, scheduling_scores):
        user_proportions = jax.nn.softmax(scheduling_scores, axis=-1)
        new_allocations_hz = (self.total_bandwidth * bandwidth_allocations)[:, None] * user_proportions
        return new_allocations_hz

    def _get_observations(self, state):
        interference_dbm = 10 * jnp.log10(state['interference'] + 1e-9) + 30
        return jnp.concatenate([
            state['power_levels'],
            interference_dbm,
            state['resource_allocations'].flatten(),
            state['bs_positions'].flatten(),
            state['user_positions'].flatten()
        ])

    def _calculate_interference(self, state, power, path_loss_db):
        power_linear_mw = 10 ** (power / 10.0)
        path_loss_linear = 10 ** (path_loss_db / 10.0)
        
        received_power_linear_matrix = power_linear_mw[:, None] / path_loss_linear
        total_received_power_linear_at_users = jnp.sum(received_power_linear_matrix, axis=0)
        return total_received_power_linear_at_users

    def _calculate_path_loss_db(self, distance, key):
        pl_log_dist = 10.0 * self.path_loss_exponent * jnp.log10(distance + 1e-3)
        shadowing = jax.random.normal(key, distance.shape) * self.shadow_std_db
        return pl_log_dist + shadowing

    def _calculate_sinr(self, state, power, total_received_power_linear, path_loss_db):
        power_linear_mw = 10 ** (power / 10.0)
        path_loss_linear = 10 ** (path_loss_db / 10.0)
        
        received_power_linear_matrix = power_linear_mw[:, None] / path_loss_linear
        
        serving_bs_indices = jnp.argmax(received_power_linear_matrix, axis=0)
        serving_signal_linear = jnp.max(received_power_linear_matrix, axis=0)

        interference_linear = total_received_power_linear - serving_signal_linear
        noise_linear = 10 ** (self.noise_floor_dbm / 10.0)
        
        sinr_linear = serving_signal_linear / (interference_linear + noise_linear + 1e-9)
        sinr_db = 10.0 * jnp.log10(sinr_linear + 1e-9)

        return sinr_db, serving_bs_indices

    def _calculate_throughput(self, allocations, sinr_db, serving_bs_indices):
        sinr_linear = 10 ** (sinr_db / 10.0)
        capacity_per_hz = jnp.log2(1 + sinr_linear)
        user_one_hot = jax.nn.one_hot(serving_bs_indices, self.num_bs, dtype=allocations.dtype)
        bw_for_each_user = jnp.sum(allocations.T * user_one_hot, axis=1)
        user_throughputs = bw_for_each_user * capacity_per_hz
        return user_throughputs 

    def _calculate_reward(self, user_throughputs, power):
        fairness_reward = jnp.sum(jnp.log1p(user_throughputs / 1e6)) 
        total_power_watt = jnp.sum(10 ** (power / 10.0)) / 1000.0
        power_penalty_factor = 0.5 
        power_penalty = total_power_watt * power_penalty_factor
        reward_scaling_factor = 10.0
        reward = (fairness_reward - power_penalty) / reward_scaling_factor
        
        return reward
        
    def observation_spec(self):
        obs_len = self.num_bs + self.num_users + (self.num_bs * self.num_users) + \
                  (self.num_bs * 2) + (self.num_users * 2)
        return specs.Array(shape=(obs_len,), dtype=jnp.float32, name="observation")

    @property
    def discount_spec(self):
        return specs.BoundedArray(
            shape=(), dtype=jnp.float32, minimum=0.0, maximum=1.0, name="discount"
        )

    def action_spec(self):
        return self._action_spec
