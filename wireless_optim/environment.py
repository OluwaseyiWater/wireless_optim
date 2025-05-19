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
        # Ensure actions are clipped to [0, 1] just in case
        action = jnp.clip(action, self._action_spec.minimum, self._action_spec.maximum)

        power_adjustments = action[:, 0]  # [0,1] scale
        bandwidth_allocations = action[:, 1]  # fraction of available bandwidth per BS
        scheduling_scores = action[:, 2]  # used to decide which users are prioritized

        # Update power levels (adjust power by a factor, then clip)
        # Assuming action[:, 0] = 0.5 means no change, > 0.5 means increase, < 0.5 means decrease
        # Scale (action - 0.5) from [-0.5, 0.5] to [-5.0, 5.0] dB change
        power_change_db = (power_adjustments - 0.5) * 10.0 # Let's use 10 dB range for adjustment
        new_power = jnp.clip(self.state['power_levels'] + power_change_db, 0.0, self.max_power)

        # Update resource allocations based on bandwidth allocation and scheduling decisions
        # Use softplus to make bandwidth_allocations > 0
        new_allocations = self._update_allocations(
            self.state['resource_allocations'],
            jax.nn.softplus(bandwidth_allocations), # Ensure positive and varying
            scheduling_scores
        )

        # Calculate interference based on updated power and positions
        interference = self._calculate_interference(self.state, new_power)

        # Calculate SINR with path loss, shadowing, and interference
        # Pass current step count or derive key deterministically for shadowing
        sinr_key = jax.random.PRNGKey(self.state['step_count']) # Using step count for deterministic shadowing
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

        # Jumanji convention: discount is 1.0 for non-terminal, 0.0 for terminal
        discount = jnp.array(0.0 if done else 1.0, dtype=jnp.float32)

        return termination(observation=obs, reward=reward) if done else transition(observation=obs, reward=reward)

    def _grid_positions(self, num_points, scale=1.0, key=None):
        """Generate positions for macro BS on a grid."""
        n = int(jnp.ceil(jnp.sqrt(num_points)))
        lin = jnp.linspace(0.1, 0.9, n)
        grid = jnp.array(np.meshgrid(lin, lin)).T.reshape(-1, 2)
        return grid[:num_points] * scale

    def _update_allocations(self, current_allocations, bandwidth_allocations, scheduling_scores):
        """Update resource allocations based on BS bandwidth fractions and scheduling decisions."""
        # Ensure bandwidth_allocations and scheduling_scores influence allocation
        # Simple approach: distribute total BS bandwidth according to scheduling scores
        # Normalize scheduling scores per BS
        normalized_scores = scheduling_scores / (jnp.sum(scheduling_scores, axis=-1, keepdims=True) + 1e-6)

        # Total bandwidth per BS scaled by action
        bs_bandwidth = self.total_bandwidth * bandwidth_allocations
        bs_effectiveness = jnp.expand_dims(bandwidth_allocations * scheduling_scores, axis=-1) # Shape (num_bs, 1)
        effective_bw_alloc = jax.nn.softplus(bandwidth_allocations) # > 0
        effective_sched_scores = jax.nn.softplus(scheduling_scores) # > 0
        bs_weight = jnp.expand_dims(effective_bw_alloc * effective_sched_scores, axis=-1) # Shape (num_bs, 1)
        # Distribute this "weight" equally among all users from this BS's perspective
        new_allocations = bs_weight * jnp.ones_like(current_allocations) 
        bs_contribution_weight = jnp.expand_dims(bandwidth_allocations * (scheduling_scores + 0.1), axis=-1) # Shape (num_bs, 1)
        new_allocations = bs_contribution_weight * jnp.ones_like(current_allocations) # Shape (num_bs, num_users)
        # Remove the / self.num_users. The scaling can be implicit in reward.

        return new_allocations


    def _get_observations(self, state):
        """Create an observation vector from the current state."""
        return jnp.concatenate([
            state['power_levels'],                             # (num_bs,)
            state['interference'],                             # (num_users,)
            state['resource_allocations'].flatten(),           # (num_bs * num_users,)
            state['bs_positions'].flatten(),                   # (num_bs * 2,)
            state['user_positions'].flatten()                  # (num_users * 2,)
        ])

    def _calculate_interference(self, state, power):
        """Calculate interference at each user using a distance-based model."""
        distances = jnp.linalg.norm(
            state['bs_positions'][:, None] - state['user_positions'][None, :],
            axis=-1
        ) # Shape (num_bs, num_users)

        # Power in dBm needs conversion to linear mW for interference summation
        power_linear_mw = 10 ** (power / 10.0) # Shape (num_bs,)
        path_loss_linear = 10 ** (self._calculate_path_loss_db(distances) / 10.0) # Shape (num_bs, num_users)

        # Received power linear at user j from BS i = power_linear_mw[i] / path_loss_linear[i, j]
        received_power_linear_matrix = power_linear_mw[:, None] / path_loss_linear # Shape (num_bs, num_users)

        # Calculate total received power linear at each user from ALL BSs
        total_received_power_linear_at_users = jnp.sum(received_power_linear_matrix, axis=0) # Shape (num_users,)

        # Return this total received power. The SINR function will handle desired signal vs interference.
        return total_received_power_linear_at_users # This was named 'interference' before, let's stick to that for observation consistency


    def _calculate_path_loss_db(self, distance, key=None):
        """Compute free-space path loss (in dB) with a path loss exponent and log-normal shadowing."""
        # Use the provided key for shadowing if available, otherwise derive it (e.g., from step count)
        if key is None:
             key = jax.random.PRNGKey(0)
        
        pl_log_dist = 10.0 * self.path_loss_exponent * jnp.log10(distance + 1e-3)

        # Add shadow fading
        keys = jax.random.split(key, distance.shape[-1]) # Split key for each user/distance pair
        shadow = jax.random.normal(jax.random.fold_in(key, jnp.sum(distance)), distance.shape) * self.shadow_std_db
        # Using fold_in for slightly better key usage across different distances/calls

        return pl_log_dist + shadow


    def _calculate_sinr(self, state, power, total_received_power_linear):
        """Calculate SINR at each user."""
        distances = jnp.linalg.norm(
            state['bs_positions'][:, None] - state['user_positions'][None, :], axis=-1) # Shape (num_bs, num_users)

        # Assuming user connects to BS with strongest signal
        # Calculate received signal strength (linear mW) for each BS-user pair
        power_linear_mw = 10 ** (power / 10.0) # Shape (num_bs,)

        # Need path loss in linear for received power calculation
        # Call _calculate_path_loss_db and convert to linear
        # Pass a key to path loss calculation!
        path_loss_key = jax.random.fold_in(jax.random.PRNGKey(self.state['step_count']), 1) # Use step count for key
        path_loss_db = self._calculate_path_loss_db(distances, key=path_loss_key) # Shape (num_bs, num_users)
        path_loss_linear = 10 ** (path_loss_db / 10.0) # Shape (num_bs, num_users)

        received_power_linear_matrix = power_linear_mw[:, None] / path_loss_linear # Shape (num_bs, num_users)

        # Find the serving BS for each user (BS with max received power)
        serving_bs_indices = jnp.argmax(received_power_linear_matrix, axis=0) # Shape (num_users,)

        # Get the serving signal power for each user
        serving_signal_linear = jnp.max(received_power_linear_matrix, axis=0) # Shape (num_users,)

        # Total received power at each user was calculated in _calculate_interference.
        # Interference = Total Received Power - Serving Signal Power
        interference_linear = total_received_power_linear - serving_signal_linear # Shape (num_users,)

        # Add noise floor (linear)
        noise_linear = 10 ** (self.noise_floor_dbm / 10.0)

        # SINR = Serving Signal / (Interference + Noise)
        # Ensure denominator is non-zero and non-negative
        sinr_linear = serving_signal_linear / (interference_linear + noise_linear + 1e-9) # Add epsilon for stability
        sinr_linear = jnp.clip(sinr_linear, 1e-9, None) # Clip for stability

        # Convert SINR back to dB for return, clipping to avoid log(0)
        return 10.0 * jnp.log10(sinr_linear)


    def _calculate_throughput(self, allocations, sinr_db):
        """Calculate throughput using the Shannon-Hartley theorem."""
        # allocations shape: (num_bs, num_users)
        # sinr_db shape: (num_users,) - SINR experienced by each user (from their best BS)

        # Convert SINR from dB to linear
        sinr_linear = 10 ** (sinr_db / 10.0)
        sinr_linear = jnp.clip(sinr_linear, 0.0, None) # Ensure non-negative

        # Calculate theoretical capacity per user (bits/sec/Hz) based on their SINR
        capacity_per_hz_per_user = jnp.log2(1 + sinr_linear + 1e-6) # Add epsilon for stability, shape (num_users,)
        capacity_term = jnp.log2(1 + sinr_linear + 1e-6) # Shape (num_users,)
        throughput = jnp.sum(allocations * capacity_term[None, :]) # Sum over both axes
        return throughput * self.total_bandwidth # Scale by total bandwidth? Or is allocations already proportional to total BW?

    def _calculate_throughput(self, allocations, sinr_db):
         """Calculate throughput using the Shannon-Hartley theorem based on allocations and user SINR."""
         sinr_linear = 10 ** (sinr_db / 10.0)
         sinr_linear = jnp.clip(sinr_linear, 0.0, None)
         capacity_per_hz_per_user = jnp.log2(1 + sinr_linear + 1e-6) # Shape (num_users,)

         # Sum of allocation weights directed towards each user from all BSs
         total_allocation_weight_per_user = jnp.sum(allocations, axis=0) # Shape (num_users,)
         total_throughput = jnp.sum(total_allocation_weight_per_user * capacity_per_hz_per_user)

         return total_throughput


    def _calculate_reward(self, throughput, power):
        """Multi-objective reward that encourages high throughput, penalizes high power, and rewards fairness."""

        # Re-using the user's logic directly:
        throughput_reward = jnp.clip(throughput, 0.0, 1000.0)  # Lower cap based on realistic throughput (dimensionless?)
        total_power_dbm = jnp.sum(power)
        total_power_linear_mw = jnp.sum(10 ** (power / 10.0)) # Sum of linear power

        
        total_power_linear_mw = jnp.sum(10 ** (power / 10.0)) # Shape ()
        total_power_linear_mw = jnp.clip(total_power_linear_mw, 1e-9, None) # Avoid div by zero in fairness

        # Scale penalty factor - needs tuning.
        power_penalty = total_power_linear_mw * 1e-4 # Example scaling

        # Scale fairness term - needs tuning. Throughput might be around 1000 (dimensionless).
        # Fairness = throughput / total_power_linear_mw * fairness_factor
        fairness = throughput_reward / (total_power_linear_mw + 1e-9) * 100.0 # Example scaling

        reward = throughput_reward - power_penalty + fairness
        reward = jnp.where(jnp.isnan(reward) | jnp.isinf(reward), 0.0, reward) # Handle NaNs/Infs

        return reward


    def observation_spec(self):
        num_bs = self.num_macro_bs + self.num_small_bs
        num_users = self.num_users
        total_length = num_bs + num_users + \
                       num_bs * num_users + \
                       num_bs * 2 + \
                       num_users * 2
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
            minimum=jnp.array(0.0, dtype=jnp.float32),
            maximum=jnp.array(1.0, dtype=jnp.float32),
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
