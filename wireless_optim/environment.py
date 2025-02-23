import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition

class HetNetEnvironment(Environment):
    """Custom Jumanji environment for HetNet Resource Allocation"""
    
    def __init__(self, num_macro_bs=3, num_small_bs=10, num_users=50, max_steps=100):
        super().__init__()
        self.num_macro_bs = num_macro_bs
        self.num_small_bs = num_small_bs
        self.num_users = num_users
        self.max_steps = max_steps
        
        # Network parameters
        self.total_bandwidth = 100e6  # 100 MHz
        self.max_power = 40  # dBm
        self.carrier_freq = 2.4e9  # 2.4 GHz
        
        # Define action and observation specs
        self._action_spec = specs.BoundedArray(
            shape=(num_macro_bs + num_small_bs, 3),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="action"
        )
        
    def reset(self, key: jax.random.PRNGKey) -> TimeStep:
        """Reset the environment to initial state"""
        # Initialize base stations and users
        state = {
            'bs_positions': jax.random.uniform(key, (self.num_macro_bs + self.num_small_bs, 2)),
            'user_positions': jax.random.uniform(key, (self.num_users, 2)),
            'resource_allocations': jnp.zeros((self.num_macro_bs + self.num_small_bs, self.num_users)),
            'power_levels': jnp.full((self.num_macro_bs + self.num_small_bs,), 30.0),
            'interference': jnp.zeros((self.num_users,)),
            'step_count': 0
        }
        
        # Calculate initial observations
        obs = self._get_observations(state)
        return restart(obs)

    def step(self, state: dict, action: jnp.ndarray) -> TimeStep:
        """Perform one timestep in the environment"""
        # Decode actions
        power_adjustments = action[:, 0]
        bandwidth_allocations = action[:, 1]
        user_scheduling = action[:, 2]
        
        # Update power levels (clipped to valid range)
        new_power = jnp.clip(state['power_levels'] + power_adjustments * 5, 0, self.max_power)
        
        # Update resource allocations
        new_allocations = self._update_allocations(
            state['resource_allocations'], 
            bandwidth_allocations,
            user_scheduling
        )
        
        # Calculate interference and SINR
        interference = self._calculate_interference(state)
        sinr = self._calculate_sinr(state, new_power, interference)
        
        # Calculate throughput and rewards
        throughput = self._calculate_throughput(new_allocations, sinr)
        reward = self._calculate_reward(throughput, new_power)
        
        # Update state
        new_state = {
            **state,
            'power_levels': new_power,
            'resource_allocations': new_allocations,
            'interference': interference,
            'step_count': state['step_count'] + 1
        }
        
        # Check termination
        done = new_state['step_count'] >= self.max_steps
        obs = self._get_observations(new_state)
        
        return transition(obs, reward, done) if done else termination(obs, reward)

    def _get_observations(self, state):
        """Create observation space with network metrics"""
        return jnp.concatenate([
            state['power_levels'],
            state['interference'],
            state['resource_allocations'].flatten(),
            state['bs_positions'].flatten(),
            state['user_positions'].flatten()
        ])

    def _calculate_interference(self, state):
        """Calculate interference using distance-based model"""
        # Simplified interference model (distance-based)
        distances = jnp.linalg.norm(
            state['bs_positions'][:, None] - state['user_positions'][None, :],
            axis=-1
        )
        return jnp.sum(state['power_levels'][:, None] / (distances + 1e-6), axis=0)

    def _calculate_sinr(self, state, power, interference):
        """Calculate Signal-to-Interference-plus-Noise Ratio"""
        noise_floor = -174  # dBm/Hz
        signal = power[:, None] - 20 * jnp.log10(
            jnp.linalg.norm(state['bs_positions'][:, None] - state['user_positions'][None, :], axis=-1)
        )
        return signal - 10 * jnp.log10(interference + 10**(noise_floor/10))

    def _calculate_throughput(self, allocations, sinr):
        """Calculate throughput using Shannon-Hartley theorem"""
        return allocations * jnp.log2(1 + 10**(sinr/10))

    def _calculate_reward(self, throughput, power):
        """Multi-objective reward function"""
        throughput_reward = jnp.sum(throughput)
        power_penalty = jnp.sum(power) * 0.1
        fairness = (jnp.sum(throughput)**2) / (self.num_users * jnp.sum(throughput**2))
        return throughput_reward - power_penalty + fairness

    def observation_spec(self):
        return specs.Array(
            shape=(self.num_macro_bs + self.num_small_bs + 
                  self.num_users + 
                  (self.num_macro_bs + self.num_small_bs)*self.num_users +
                  2*(self.num_macro_bs + self.num_small_bs + self.num_users),),
            dtype=jnp.float32,
            name="observation"
        )

    def action_spec(self):
        return self._action_spec

# Example usage
# if __name__ == "__main__":
#     env = HetNetEnvironment()
#     key = jax.random.PRNGKey(0)
#     state = env.reset(key)
    
#     for _ in range(10):
#         action = env.action_spec().generate_value()
#         timestep = env.step(state, action)
#         state = timestep.state
#         print(f"Step {_}: Reward {timestep.reward:.2f}")