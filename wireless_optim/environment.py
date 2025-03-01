import jax
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, transition, termination
from dataclasses import dataclass

class HetNetEnvironment(Environment):
    """Enhanced Jumanji environment for HetNet Resource Allocation with improved realism."""
    
    def __init__(self, num_macro_bs=3, num_small_bs=10, num_users=50, max_steps=100):
        super().__init__()
        self.num_macro_bs = num_macro_bs
        self.num_small_bs = num_small_bs
        self.num_users = num_users
        self.max_steps = max_steps
        
        # Network parameters
        self.total_bandwidth = 100e6  # 100 MHz
        self.max_power = 40.0  # dBm
        self.carrier_freq = 2.4e9  # 2.4 GHz
        self.path_loss_exponent = 3.5  # realistic urban path loss exponent
        self.shadow_std_db = 4.0  # standard deviation for shadow fading (dB)
        self.noise_floor_dbm = -100.0  # effective noise floor in dBm for the total bandwidth
        
        # Define action spec: each BS (macro + small) outputs [power_adjust, bandwidth_alloc, scheduling_score]
        self._action_spec = specs.BoundedArray(
            shape=(self.num_macro_bs + self.num_small_bs, 3),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="action"
        )
        # Internal state storage
        self.state = None
        
    def reset(self, key: jax.random.PRNGKey) -> TimeStep:
        """Reset the environment to initial state with improved BS and user placement."""
        keys = jax.random.split(key, 4)
        
        # Place macro base stations on a grid (e.g., equally spaced)
        macro_positions = self._grid_positions(self.num_macro_bs, scale=1.0, key=keys[0])
        
        # Scatter small BS around the area, slightly clustered around macro BS
        small_positions = jax.random.uniform(keys[1], (self.num_small_bs, 2), minval=0.0, maxval=1.0)
        
        bs_positions = jnp.concatenate([macro_positions, small_positions], axis=0)
        
        # Randomly place users in the area
        user_positions = jax.random.uniform(keys[2], (self.num_users, 2), minval=0.0, maxval=1.0)
        
        self.state = {
            'bs_positions': bs_positions,             # shape: (num_bs, 2)
            'user_positions': user_positions,         # shape: (num_users, 2)
            'resource_allocations': jnp.zeros((self.num_macro_bs + self.num_small_bs, self.num_users)),
            'power_levels': jnp.full((self.num_macro_bs + self.num_small_bs,), 30.0),  # starting power in dBm
            'interference': jnp.zeros((self.num_users,)),
            'step_count': 0
        }
        
        obs = self._get_observations(self.state)
        return restart(observation=obs)
    
    def step(self, action: jnp.ndarray) -> TimeStep:
        """Perform one timestep in the environment using the provided action."""
        # Decode actions
        power_adjustments = action[:, 0]  # [0,1] scale
        bandwidth_allocations = action[:, 1]  # fraction of available bandwidth per BS
        scheduling_scores = action[:, 2]  # used to decide which users are prioritized
        
        # Update power levels (adjust power by a factor, then clip)
        new_power = jnp.clip(self.state['power_levels'] + (power_adjustments - 0.5) * 5.0, 0.0, self.max_power)
        
        # Update resource allocations based on bandwidth allocation and scheduling decisions
        new_allocations = self._update_allocations(self.state['resource_allocations'], 
                                                   bandwidth_allocations, 
                                                   scheduling_scores)
        
        # Calculate interference based on updated power and positions
        interference = self._calculate_interference(self.state, new_power)
        
        # Calculate SINR with path loss, shadowing, and interference
        sinr = self._calculate_sinr(self.state, new_power, interference)
        
        # Calculate throughput and reward
        throughput = self._calculate_throughput(new_allocations, sinr)
        reward = self._calculate_reward(throughput, new_power)
        
        # Update internal state
        self.state = {
            **self.state,
            'power_levels': new_power,
            'resource_allocations': new_allocations,
            'interference': interference,
            'step_count': self.state['step_count'] + 1
        }
        
        obs = self._get_observations(self.state)
        done = self.state['step_count'] >= self.max_steps
        
        return termination(observation=obs, reward=reward) if done else transition(observation=obs, reward=reward)
    
    def _grid_positions(self, num_points, scale=1.0, key=None):
        """Generate positions for macro BS on a grid."""
        n = int(jnp.ceil(jnp.sqrt(num_points)))
        lin = jnp.linspace(0.1, 0.9, n)
        grid = jnp.array(np.meshgrid(lin, lin)).T.reshape(-1, 2)
        return grid[:num_points] * scale

    def _update_allocations(self, current_allocations, bandwidth_allocations, scheduling_scores):
        """Update resource allocations based on BS bandwidth fractions and scheduling decisions."""
        bs_weight = jnp.expand_dims(bandwidth_allocations * (scheduling_scores + 0.1), axis=-1)
        new_allocations = bs_weight * jnp.ones_like(current_allocations) / self.num_users
        return new_allocations

    def _get_observations(self, state):
        """Create an observation vector from the current state."""
        return jnp.concatenate([
            state['power_levels'],
            state['interference'],
            state['resource_allocations'].flatten(),
            state['bs_positions'].flatten(),
            state['user_positions'].flatten()
        ])
    
    def _calculate_interference(self, state, power):
        """Calculate interference at each user using a distance-based model."""
        distances = jnp.linalg.norm(
            state['bs_positions'][:, None] - state['user_positions'][None, :],
            axis=-1
        )
        interference_matrix = power[:, None] / (distances + 1e-3)**self.path_loss_exponent
        return jnp.sum(interference_matrix, axis=0)
    
    def _calculate_path_loss(self, distance):
        """Compute free-space path loss (in dB) with a path loss exponent and log-normal shadowing."""
        pl = 20 * jnp.log10(distance + 1e-3) * self.path_loss_exponent / 2.0
        shadow = jax.random.normal(jax.random.PRNGKey(0), distance.shape) * self.shadow_std_db
        return pl + shadow

    def _calculate_sinr(self, state, power, interference):
        """Calculate SINR at each user."""
        distances = jnp.linalg.norm(
            state['bs_positions'][:, None] - state['user_positions'][None, :],
            axis=-1
        )
        path_loss = self._calculate_path_loss(distances)
        received_signals = power[:, None] - path_loss
        best_signal = jnp.max(received_signals, axis=0)
        best_signal_linear = 10 ** (best_signal / 10)
        noise_linear = 10 ** (self.noise_floor_dbm / 10)
        sinr_linear = best_signal_linear / (interference + noise_linear)
        return 10 * jnp.log10(sinr_linear + 1e-6)
    
    def _calculate_throughput(self, allocations, sinr):
        """Calculate throughput using the Shannon-Hartley theorem."""
        sinr_linear = 10 ** (sinr / 10)
        throughput = allocations * jnp.log2(1 + sinr_linear)
        return jnp.sum(throughput)
    
    def _calculate_reward(self, throughput, power):
        """Multi-objective reward that encourages high throughput, penalizes high power, and rewards fairness."""
        throughput_reward = throughput
        power_penalty = jnp.sum(power) * 0.1
        fairness = throughput / (jnp.sum(power) + 1e-3)
        return throughput_reward - power_penalty + fairness
  
    def observation_spec(self):
        total_length = (self.num_macro_bs + self.num_small_bs) + self.num_users + \
                       (self.num_macro_bs + self.num_small_bs) * self.num_users + \
                       2 * ((self.num_macro_bs + self.num_small_bs) + self.num_users)
        return specs.Array(
            shape=(total_length,),
            dtype=jnp.float32,
            name="observation"
        )

    @property
    def discount_spec(self):
        return specs.BoundedArray(
            shape=(),
            dtype=jnp.float32,
            minimum=jnp.array(0.0, dtype=jnp.float32),  # Explicit JAX float32
            maximum=jnp.array(1.0, dtype=jnp.float32),  # Explicit JAX float32
            name="discount"
        )

    def action_spec(self):
        return self._action_spec


@dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: float
    next_obs: jnp.ndarray
    done: float


# Example usage
# if __name__ == "__main__":
#     env = HetNetEnvironment()
#     key = jax.random.PRNGKey(0)
#     timestep = env.reset(key)
    
#     # Run for a few steps using random actions from the action spec
#     for step in range(10):
#         action = env.action_spec().generate_value()
#         timestep = env.step(action)
#         print(f"Step {step}: Reward {timestep.reward:.2f}")
