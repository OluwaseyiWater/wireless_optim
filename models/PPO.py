import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import os
import wandb

os.makedirs("plots", exist_ok=True)


from wireless_optim.environment import HetNetEnvironment


def unflatten_action(flat_action, num_bs, num_users):
    
    pa_end = num_bs
    ba_end = 2 * num_bs
    pa = flat_action[:pa_end]
    ba = flat_action[pa_end:ba_end]
    ss = flat_action[ba_end:].reshape((num_bs, num_users))
    return pa, ba, ss


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
    ])
    hidden = mlp(obs)
    mu = hk.Linear(action_dim, w_init=hk.initializers.TruncatedNormal(stddev=0.1))(hidden)
    mu = jax.nn.sigmoid(mu)
    log_sigma = hk.Linear(action_dim)(hidden)
    value = hk.Linear(1)(hidden)
    return mu, log_sigma, jnp.squeeze(value, axis=-1)

def ppo_train(
       env,
       config,
       seed: int = 0,
       use_wandb: bool = True,
       wandb_project: str = "ppo-training",
       wandb_name: str = None,
    ):
    
    
    action_spec = env.action_spec()
    action_dim = int(np.prod(action_spec.shape))
    num_bs = env.num_bs
    num_users = env.num_users
    obs_dim = env.observation_spec().shape[0]
    hidden_size = config['hidden_size']

    def forward_fn(obs):
        return ppo_network_fn(obs, action_dim, hidden_size)
    ppo_net = hk.without_apply_rng(hk.transform(forward_fn))
    key = jax.random.PRNGKey(seed)
    global_step = 0

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={**config, "seed": seed})
    
    dummy_obs = jnp.zeros((obs_dim,))
    params = ppo_net.init(key, dummy_obs)
    optimizer = optax.chain(optax.clip_by_global_norm(config['max_grad_norm']), optax.adam(config['lr']))
    opt_state = optimizer.init(params)

    
    reward_mean, reward_std, reward_m2, reward_count = 0.0, 1.0, 0.0, 0
    obs_mean, obs_std, obs_m2, obs_count = jnp.zeros(obs_dim), jnp.ones(obs_dim), jnp.zeros(obs_dim), 0

    def ppo_loss(params, observations, actions, old_log_probs, advantages, returns):
        mu, log_sigma, value_pred = jax.vmap(lambda obs: ppo_net.apply(params, obs))(observations)
        value_pred = jnp.clip(value_pred, -1000, 1000)
        
        log_sigma = jnp.clip(log_sigma, -20, 2)
        sigma = jnp.exp(log_sigma)
        
        new_log_probs = jax.scipy.stats.norm.logpdf(actions, mu, sigma).sum(axis=-1)
        
        ratio = jnp.exp(jnp.clip(new_log_probs - old_log_probs, -20, 20))
        clipped_ratio = jnp.clip(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
        
        value_loss = jnp.mean((returns - value_pred) ** 2)
        
        entropy = jnp.mean(0.5 * (jnp.log(2 * jnp.pi * sigma**2 + 1e-6) + 1).sum(axis=-1))
        
        total_loss = policy_loss + config['vf_coef'] * value_loss - config['ent_coef'] * entropy
        return jnp.where(jnp.isnan(total_loss), 0.0, total_loss)

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

    epoch_rewards, epoch_losses = [], []
    epoch_powers, epoch_bandwidths, epoch_schedules = [], [], []

    for epoch in range(config['num_epochs']):
        observations, norm_observations, actions, rewards, dones, log_probs, values = [], [], [], [], [], [], []
        
        key, subkey = jax.random.split(key)
        state = env.reset(subkey)
        ep_reward = 0.0

        for step in range(config['num_steps']):
            obs = state.observation
            update_obs_stats(obs)
            norm_obs = (obs - obs_mean) / (obs_std + 1e-8)
            observations.append(obs) 
            norm_observations.append(norm_obs)
            
            mu, log_sigma, value = ppo_net.apply(params, norm_obs)
            sigma = jnp.exp(jnp.clip(log_sigma, -20, 2))
            
            key, subkey = jax.random.split(key)
            
            action_flat = mu + sigma * jax.random.normal(subkey, shape=mu.shape)
            action_flat = jnp.clip(action_flat, 0.0, 1.0)
            
            if use_wandb:
                
                pa, ba, ss = unflatten_action(action_flat, num_bs, num_users)
                wandb.log({
                    "ppo/power_adjustments":     wandb.Histogram(np.array(pa)),
                    "ppo/bandwidth_allocations": wandb.Histogram(np.array(ba)),
                    "ppo/scheduling_scores_mean": np.mean(np.array(ss)), 
                    "ppo/pa_mean":               np.mean(np.array(pa)),
                    "ppo/ba_mean":               np.mean(np.array(ba)),
                    "global_step":               global_step,
                }, step=global_step)
            global_step += 1
            
            
            actions.append(action_flat)
            log_prob = jax.scipy.stats.norm.logpdf(action_flat, mu, sigma).sum()
            log_probs.append(float(log_prob))
            values.append(np.array(value))
            
            state = env.step(action_flat)
            rewards.append(state.reward)
            dones.append(1.0 if state.discount == 0 else 0.0)
            ep_reward += state.reward
            
            if state.discount == 0:
                break
        
        unpacked_actions = [unflatten_action(a, num_bs, num_users) for a in actions]
        epoch_powers.append(np.array([a[0] for a in unpacked_actions]))
        epoch_bandwidths.append(np.array([a[1] for a in unpacked_actions]))
        epoch_schedules.append(np.array([a[2] for a in unpacked_actions]))

        update_reward_stats(rewards)
        norm_rewards = (np.array(rewards) - reward_mean) / (reward_std + 1e-8)
        _, _, last_value = ppo_net.apply(params, (state.observation - obs_mean) / (obs_std + 1e-8))
        values.append(np.array(last_value))
        advantages, returns = compute_gae(norm_rewards, np.array(values), np.array(dones), 
                                         config['gamma'], config['gae_lambda'])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(config['update_epochs']):
            loss, grads = jax.value_and_grad(ppo_loss)(params, jnp.array(norm_observations), 
                                                       jnp.array(actions), jnp.array(log_probs), 
                                                       advantages, returns)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        jax.clear_caches()
        
        epoch_rewards.append(ep_reward)
        epoch_losses.append(loss)
        
        print(f"PPO Epoch {epoch}: Reward = {ep_reward:.2f}, Loss = {loss:.4f}")

        if use_wandb:
            wandb.log({
                "ppo/epoch":         epoch,
                "ppo/total_reward":  ep_reward,
                "ppo/epoch_loss":    float(loss),
            }, step=global_step)

    if use_wandb:
        best_reward = max(epoch_rewards) if epoch_rewards else 0
        avg_reward  = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        best_loss   = min(epoch_losses) if epoch_losses else 0
        avg_loss    = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        wandb.run.summary["best_reward"] = best_reward
        wandb.run.summary["avg_reward"]  = avg_reward
        wandb.run.summary["best_loss"]   = best_loss
        wandb.run.summary["avg_loss"]    = avg_loss

        wandb.finish()

    return params, ppo_net, epoch_rewards, epoch_losses, epoch_powers, epoch_bandwidths, epoch_schedules
