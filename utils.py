import jax.numpy as jnp
import numpy as np

# -------------------------
# Utility: Unflatten Action
# -------------------------
def unflatten_action(index: int, shape):
    total = int(np.prod(shape))
    one_hot = jnp.zeros(total)
    one_hot = one_hot.at[index].set(1.0)
    return one_hot.reshape(shape)


# =============================================================================
# Offline RL Pretraining
# =============================================================================
def offline_pretrain_policy(dataset, agent, pretrain_epochs=1000, batch_size=64):
    """
    Populate the agent's replay buffer with dataset transitions and pretrain the agent.
    """
    for transition in dataset:
        agent.replay_buffer.add(Transition(
            obs=jnp.array(transition["state"]),
            action=jnp.array(transition["action"]),
            reward=transition["reward"],
            next_obs=jnp.array(transition["next_state"]),
            done=1.0 if transition["done"] else 0.0
        ))
    key = jax.random.PRNGKey(42)
    for epoch in range(pretrain_epochs):
        if len(agent.replay_buffer.buffer) >= batch_size:
            key, sample_key = jax.random.split(key)
            batch = agent.replay_buffer.sample(batch_size, sample_key)
            loss = agent.update(batch)
            if epoch % 100 == 0:
                print(f"Pretraining Epoch {epoch}: Loss = {loss:.4f}")
    agent.update_target_network()
    print("Offline pretraining completed.")


def train_d3qn_online(env, agent, num_episodes=10, batch_size=64):
    """
    Continue training the pretrained D3QN agent on the online environment.
    """
    key = jax.random.PRNGKey(0)
    episode_rewards = []
    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        ep_reward = 0.0
        while state.discount != 0:
            key, action_key = jax.random.split(key)
            epsilon = max(0.01, 0.1 * (0.98 ** episode))
            if jax.random.uniform(action_key) < epsilon:
                action = env.action_spec().generate_value()
            else:
                q_values = agent.net.apply(agent.params, state.observation)
                # Reshape q_values to (num_bs, 3)
                q_values = q_values.reshape(env.action_spec().shape)
                # Get per-BS actions: for each base station, select the action with highest Q-value.
                action_indices = jnp.argmax(q_values, axis=1)
                # Convert to one-hot representation: shape (num_bs, 3)
                action = jax.nn.one_hot(action_indices, env.action_spec().shape[1])
            key, step_key = jax.random.split(key)
            next_state = env.step(action)
            agent.replay_buffer.add(Transition(
                obs=state.observation,
                action=action,
                reward=next_state.reward,
                next_obs=next_state.observation,
                done=1.0 if next_state.discount == 0 else 0.0
            ))
            ep_reward += next_state.reward
            state = next_state
            if len(agent.replay_buffer.buffer) >= batch_size:
                key, sample_key = jax.random.split(key)
                batch = agent.replay_buffer.sample(batch_size, sample_key)
                loss = agent.update(batch)
                agent.update_target_network()
        episode_rewards.append(ep_reward)
        print(f"Online D3QN Episode {episode}: Total Reward = {ep_reward:.2f}")
    return agent, episode_rewards