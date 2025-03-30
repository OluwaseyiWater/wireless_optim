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
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
        hk.Linear(hidden_size), jax.nn.relu,
    ])
    hidden = mlp(obs)
    #logits = hk.Linear(action_dim)(hidden)
    mu = hk.Linear(action_dim, w_init=hk.initializers.TruncatedNormal(stddev=0.1))(hidden)  # Mean of action distribution
    mu = jax.nn.sigmoid(mu)  # Ensure mu in [0, 1]
    log_sigma = hk.Linear(action_dim)(hidden) # Log standard deviation
    value = hk.Linear(1)(hidden)
    return mu, log_sigma, jnp.squeeze(value, axis=-1)

# def ppo_train(env, config,seed=0):
#     action_shape = env.action_spec().shape
#     action_dim = int(np.prod(action_shape))
#     obs_dim = env.observation_spec().shape[0]
#     hidden_size = config['hidden_size']

#     def forward_fn(obs):
#         return ppo_network_fn(obs, action_dim, hidden_size)
#     ppo_net = hk.without_apply_rng(hk.transform(forward_fn))
#     key = jax.random.PRNGKey(seed)
#     dummy_obs = jnp.zeros((obs_dim,))
#     params = ppo_net.init(key, dummy_obs)
#     optimizer = optax.adam(config['lr'])
#     opt_state = optimizer.init(params)

#     epoch_rewards = []
#     epoch_losses = []

#     for epoch in range(config['num_epochs']):
#         observations = []
#         actions = []
#         rewards = []
#         dones = []
#         log_probs = []
#         values = []

#         key, subkey = jax.random.split(key)
#         state = env.reset(subkey)
#         ep_reward = 0.0

#         for step in range(config['num_steps']):
#             obs = state.observation
#             observations.append(obs)
#             logits, value = ppo_net.apply(params, obs)
#             values.append(np.array(value))
#             probs = jax.nn.softmax(logits)
#             key, subkey = jax.random.split(key)
#             action = int(jax.random.categorical(subkey, logits))
#             actions.append(action)
#             log_prob = jnp.log(probs[action] + 1e-8)
#             log_probs.append(float(log_prob))
#             action_one_hot = unflatten_action(action, action_shape)
#             state = env.step(action_one_hot)
#             rewards.append(state.reward)
#             dones.append(1.0 if state.discount == 0 else 0.0)
#             ep_reward += state.reward
#             if state.discount == 0:
#                 break

#         _, last_value = ppo_net.apply(params, state.observation)
#         values.append(np.array(last_value))

#         observations = jnp.array(observations)
#         actions = jnp.array(actions)
#         rewards = np.array(rewards)
#         dones = np.array(dones)
#         log_probs = jnp.array(log_probs)
#         values = np.array(values)

#         advantages, returns = compute_gae(rewards, values, dones, config['gamma'], config['gae_lambda'])
#         advantages = jnp.array(advantages)
#         returns = jnp.array(returns)

#         def ppo_loss(params, observations, actions, old_log_probs, advantages, returns):
#             mu, log_sigma, value_pred = jax.vmap(lambda obs: ppo_net.apply(params, obs))(observations)
#             sigma = jnp.exp(log_sigma)
#             new_log_probs = jax.vmap(lambda mu, sigma, a: jax.scipy.stats.norm.logpdf(a, mu, sigma).sum())(
#                 mu, sigma, actions)
#             ratio = jnp.exp(new_log_probs - old_log_probs)
#             clipped_ratio = jnp.clip(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
#             policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
#             value_loss = jnp.mean((returns - value_pred) ** 2)
#             entropy = jnp.mean(0.5 * (jnp.log(2 * jnp.pi * sigma**2) + 1))  # Gaussian entropy
#             total_loss = policy_loss + config['vf_coef'] * value_loss + config['ent_coef'] * entropy
#             return total_loss

#         for _ in range(config['update_epochs']):
#             loss, grads = jax.value_and_grad(ppo_loss)(params, observations, actions, log_probs, advantages, returns)
#             updates, opt_state = optimizer.update(grads, opt_state)
#             params = optax.apply_updates(params, updates)

#         epoch_rewards.append(ep_reward)
#         epoch_losses.append(loss)
#         print(f"PPO Epoch {epoch}: Reward = {ep_reward:.2f}, Loss = {loss:.4f}")

#     return params, ppo_net, epoch_rewards, epoch_losses

def ppo_train(env, config, seed=0):
    action_shape = env.action_spec().shape
    action_dim = int(np.prod(action_shape))
    num_bs = action_shape[0]
    obs_dim = env.observation_spec().shape[0]
    hidden_size = config['hidden_size']

    def forward_fn(obs):
        return ppo_network_fn(obs, action_dim, hidden_size)
    ppo_net = hk.without_apply_rng(hk.transform(forward_fn))
    key = jax.random.PRNGKey(seed)
    dummy_obs = jnp.zeros((obs_dim,))
    params = ppo_net.init(key, dummy_obs)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config['lr']))
    opt_state = optimizer.init(params)

    reward_mean = 0.0
    reward_std = 1.0
    reward_m2 = 0.0
    reward_count = 0
    obs_mean = jnp.zeros(obs_dim)
    obs_std = jnp.ones(obs_dim)
    obs_m2 = jnp.zeros(obs_dim)
    obs_count = 0

    def ppo_loss(params, observations, actions, old_log_probs, advantages, returns):
        mu, log_sigma, value_pred = jax.vmap(lambda obs: ppo_net.apply(params, obs))(observations)
        value_pred = jnp.clip(value_pred, -1000, 1000)  # Clip value predictions
        log_sigma = jnp.clip(log_sigma, -20, 2)
        sigma = jnp.exp(log_sigma)
        batch_size = actions.shape[0]
        num_bs = actions.shape[1]
        mu = mu.reshape(batch_size, num_bs, 3)
        sigma = sigma.reshape(batch_size, num_bs, 3)
        new_log_probs = jax.vmap(lambda mu, sigma, a: jax.scipy.stats.norm.logpdf(a, mu, sigma).sum())(
            mu, sigma, actions)
        ratio = jnp.exp(jnp.clip(new_log_probs - old_log_probs, -20, 20))
        clipped_ratio = jnp.clip(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
        value_loss = jnp.mean((returns - value_pred) ** 2)
        entropy = jnp.mean(0.5 * (jnp.log(2 * jnp.pi * sigma**2 + 1e-6) + 1))
        total_loss = policy_loss + config['vf_coef'] * value_loss + config['ent_coef'] * entropy
        return jnp.where(jnp.isnan(total_loss), 0.0, total_loss) # Prevent nan loss

    def update_reward_stats(rewards):
        nonlocal reward_mean, reward_std, reward_m2, reward_count
        for r in rewards:
            reward_count += 1
            delta = r - reward_mean
            reward_mean += delta / reward_count
            delta2 = r - reward_mean
            reward_m2 += delta * delta2
            reward_std = jnp.sqrt(reward_m2 / reward_count) if reward_count > 1 else 1.0
        reward_std = jnp.maximum(reward_std, 1e-2)

    def update_obs_stats(obs):
        nonlocal obs_mean, obs_std, obs_m2, obs_count
        obs_count += 1
        delta = obs - obs_mean
        obs_mean += delta / obs_count
        delta2 = obs - obs_mean
        obs_m2 += delta * delta2
        obs_std = jnp.sqrt(obs_m2 / obs_count) if obs_count > 1 else 1.0
        obs_std = jnp.maximum(obs_std, 1e-2)

    epoch_rewards = []
    epoch_losses = []

    for epoch in range(config['num_epochs']):
        observations, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        key, subkey = jax.random.split(key)
        state = env.reset(subkey)
        ep_reward = 0.0

        for step in range(config['num_steps']):
            obs = state.observation
            update_obs_stats(obs)
            norm_obs = (obs - obs_mean) / (obs_std + 1e-8)
            observations.append(obs)
            mu, log_sigma, value = ppo_net.apply(params, norm_obs)
            sigma = jnp.exp(jnp.clip(log_sigma, -20, 2))
            key, subkey = jax.random.split(key)
            action_flat = mu + sigma * jax.random.normal(subkey, shape=mu.shape)
            action_flat = jnp.clip(action_flat, 0.0, 1.0)
            action = action_flat.reshape(num_bs, 3)
            actions.append(action)
            log_prob = jax.scipy.stats.norm.logpdf(action_flat, mu, sigma).sum()
            log_probs.append(float(log_prob))
            values.append(np.array(value))
            state = env.step(action)
            rewards.append(state.reward)
            dones.append(1.0 if state.discount == 0 else 0.0)
            ep_reward += state.reward
            if state.discount == 0:
                break

        update_reward_stats(rewards)
        norm_rewards = (np.array(rewards) - reward_mean) / (reward_std + 1e-8)
        _, _, last_value = ppo_net.apply(params, (state.observation - obs_mean) / (obs_std + 1e-8))
        values.append(np.array(last_value))
        advantages, returns = compute_gae(norm_rewards, np.array(values), np.array(dones), 
                                         config['gamma'], config['gae_lambda'])

        for _ in range(config['update_epochs']):
            loss, grads = jax.value_and_grad(ppo_loss)(params, jnp.array(observations), 
                                                       jnp.array(actions), jnp.array(log_probs), 
                                                       advantages, returns)
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
