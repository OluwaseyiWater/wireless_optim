import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from dataclasses import dataclass
from typing import Tuple, Dict
import wandb
from wireless_optim.environment import HetNetEnvironment

def unflatten_action(flat_action, num_bs, num_users):
    pa_end, ba_end = num_bs, 2 * num_bs
    pa, ba = flat_action[:pa_end], flat_action[pa_end:ba_end]
    ss = flat_action[ba_end:].reshape((num_bs, num_users))
    return pa, ba, ss

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity):
        self.capacity, self.ptr, self.size = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr], self.action[self.ptr], self.reward[self.ptr], self.next_obs[self.ptr], self.done[self.ptr] = obs, action, reward, next_obs, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {k: getattr(self, k)[indices] for k in ["obs", "action", "reward", "next_obs", "done"]}

class TD3Agent:
    def __init__(self, obs_dim, action_dim, max_action, lr_actor, lr_critic, gamma, tau, policy_noise, noise_clip, policy_delay):
        self.action_dim, self.max_action = action_dim, max_action
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip, self.policy_delay = policy_noise, noise_clip, policy_delay
        self.action_scale = max_action / 2.0
        self.action_bias = max_action / 2.0
        
        def actor_fn(x):
            net = hk.Sequential([
                hk.Linear(256), jax.nn.relu,
                hk.Linear(256), jax.nn.relu,
                hk.Linear(self.action_dim, w_init=hk.initializers.RandomUniform(-3e-3, 3e-3)),
                jax.nn.tanh,
            ])(x)
            return self.action_scale * net + self.action_bias 
       

        def critic_fn(x, a):
            
            norm_a = (a - self.action_bias) / self.action_scale
            return hk.Sequential([
                hk.Linear(256), jax.nn.relu,
                hk.Linear(256), jax.nn.relu,
                hk.Linear(1),
            ])(jnp.concatenate([x, norm_a], axis=-1))

        self.actor, self.critic = map(hk.without_apply_rng, (hk.transform(actor_fn), hk.transform(critic_fn)))
        key = jax.random.PRNGKey(0)
        dummy_obs, dummy_action = jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim))
        key, a_key, c1_key, c2_key = jax.random.split(key, 4)
        self.actor_params, self.critic1_params, self.critic2_params = self.actor.init(a_key, dummy_obs), self.critic.init(c1_key, dummy_obs, dummy_action), self.critic.init(c2_key, dummy_obs, dummy_action)
        self.actor_optimizer = optax.adam(lr_actor)
        
        
        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), 
            optax.adam(lr_critic)
        )
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic1_opt_state = self.critic_optimizer.init(self.critic1_params)
        self.critic2_opt_state = self.critic_optimizer.init(self.critic2_params)

def train_td3(
    env: HetNetEnvironment, num_episodes: int, batch_size: int, replay_capacity: int,
    seed: int, lr_actor: float, lr_critic: float, gamma: float, tau: float,
    policy_noise: float, noise_clip: float, policy_delay: int,
    warmup_steps: int, action_noise_std: float, wandb_project: str,
    wandb_name: str, use_wandb: bool, **kwargs
):
    obs_dim, action_dim = env.observation_spec().shape[0], env.action_spec().shape[0]
    max_action = float(env.action_spec().maximum)
    agent = TD3Agent(obs_dim, action_dim, max_action, lr_actor, lr_critic, gamma, tau, policy_noise, noise_clip, policy_delay)
    
    key = jax.random.PRNGKey(seed)
    buffer = ReplayBuffer(obs_dim, action_dim, replay_capacity)
    
    @jax.jit
    def update_fn(params, opt_states, batch, key, global_step):
        actor_params, critic1_params, critic2_params, target_actor_params, target_critic1_params, target_critic2_params = params
        actor_opt_state, critic1_opt_state, critic2_opt_state = opt_states
        obs_b, action_b, reward_b, next_obs_b, done_b = batch['obs'], batch['action'], batch['reward'], batch['next_obs'], batch['done']
        
        noise = jnp.clip(jax.random.normal(key, action_b.shape) * policy_noise, -noise_clip, noise_clip)
        
        next_action = agent.actor.apply(target_actor_params, next_obs_b)
        next_action = jnp.clip(next_action + noise, 0.0, max_action)
        
        target_q1 = agent.critic.apply(target_critic1_params, next_obs_b, next_action)
        target_q2 = agent.critic.apply(target_critic2_params, next_obs_b, next_action)
        target_q = jnp.min(jnp.concatenate([target_q1, target_q2], axis=-1), axis=-1)
        bellman_target = reward_b + gamma * (1.0 - done_b) * target_q
        
        def critic_loss_fn(c_params):
            q_pred = agent.critic.apply(c_params, obs_b, action_b)
            return jnp.mean((q_pred.squeeze() - jax.lax.stop_gradient(bellman_target)) ** 2)
        
        critic1_loss, c1_grads = jax.value_and_grad(critic_loss_fn)(critic1_params)
        critic2_loss, c2_grads = jax.value_and_grad(critic_loss_fn)(critic2_params)
        c1_updates, new_c1_opt_state = agent.critic_optimizer.update(c1_grads, critic1_opt_state)
        c2_updates, new_c2_opt_state = agent.critic_optimizer.update(c2_grads, critic2_opt_state)
        new_c1_params = optax.apply_updates(critic1_params, c1_updates)
        new_c2_params = optax.apply_updates(critic2_params, c2_updates)
        
        def actor_update_body(inputs):
            a_p, a_opt_s, c1_p, t_a_p, t_c1_p, t_c2_p = inputs
            def actor_loss_fn(p): return -jnp.mean(agent.critic.apply(c1_p, obs_b, agent.actor.apply(p, obs_b)))
            a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(a_p)
            a_updates, new_a_opt_s = agent.actor_optimizer.update(a_grads, a_opt_s)
            new_a_p = optax.apply_updates(a_p, a_updates)
            new_t_a_p = jax.tree.map(lambda t, p: t * (1 - tau) + p * tau, t_a_p, new_a_p)
            new_t_c1_p = jax.tree.map(lambda t, p: t * (1 - tau) + p * tau, t_c1_p, new_c1_params)
            new_t_c2_p = jax.tree.map(lambda t, p: t * (1 - tau) + p * tau, t_c2_p, new_c2_params)
            return new_a_p, new_a_opt_s, new_t_a_p, new_t_c1_p, new_t_c2_p, a_loss
        
        def no_update(inputs):
            a_p, a_opt_s, _, t_a_p, t_c1_p, t_c2_p = inputs
            return a_p, a_opt_s, t_a_p, t_c1_p, t_c2_p, 0.0

        new_actor_params, new_actor_opt_state, new_target_actor_params, new_target_critic1_params, new_target_critic2_params, actor_loss = jax.lax.cond(
            global_step % policy_delay == 0, actor_update_body, no_update,
            (params[0], opt_states[0], new_c1_params, params[3], params[4], params[5]))
        
        updated_params = (new_actor_params, new_c1_params, new_c2_params, new_target_actor_params, new_target_critic1_params, new_target_critic2_params)
        updated_opt_states = (new_actor_opt_state, new_c1_opt_state, new_c2_opt_state)
        losses = {"actor_loss": actor_loss, "critic_loss": (critic1_loss + critic2_loss) / 2}
        return updated_params, updated_opt_states, losses
    
    params = (agent.actor_params, agent.critic1_params, agent.critic2_params, agent.actor_params, agent.critic1_params, agent.critic2_params)
    opt_states = (agent.actor_opt_state, agent.critic1_opt_state, agent.critic2_opt_state)
    
    print(f"Starting TD3 training... Warming up for {warmup_steps} steps.")
    key, reset_key = jax.random.split(key)
    ts = env.reset(reset_key)
    obs = ts.observation
    for _ in range(warmup_steps):
        key, action_key, reset_key = jax.random.split(key, 3)
        action = jax.random.uniform(action_key, (action_dim,), minval=0.0, maxval=max_action)
        ts = env.step(action)
        buffer.add(np.array(obs), np.array(action), ts.reward, np.array(ts.observation), 1.0 - ts.discount)
        obs = ts.observation if ts.discount != 0.0 else env.reset(reset_key).observation
    print("Warm-up complete.")

    global_step = warmup_steps
    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        ts = env.reset(reset_key)
        obs = ts.observation
        ep_reward, ep_len = 0.0, 0
        done = False
        
        while not done:
            key, action_key, update_key = jax.random.split(key, 3)
            actor_action = agent.actor.apply(params[0], obs)
            noise = jax.random.normal(action_key, (action_dim,)) * action_noise_std
            action = jnp.clip(actor_action + noise, 0.0, max_action)
            ts = env.step(action)
            next_obs, reward, discount = ts.observation, ts.reward, ts.discount
            buffer.add(np.array(obs), np.array(action), reward, np.array(next_obs), 1.0 - discount)
            obs = next_obs
            done = discount == 0.0
            batch = jax.tree.map(jnp.asarray, buffer.sample(batch_size))
            params, opt_states, losses = update_fn(params, opt_states, batch, update_key, global_step)
            
            ep_reward += float(reward)
            ep_len += 1
            global_step += 1

           
            if use_wandb:
                
                wandb.log({f"train/{k}": v for k, v in losses.items()}, step=global_step)
                pa, ba, ss = unflatten_action(action, env.num_bs, env.num_users)
                wandb.log({
                    "metrics/reward_per_step": float(reward),
                    "env/pa_mean": np.mean(pa),
                    "env/ba_mean": np.mean(ba),
                    "env/ss_mean": np.mean(ss)
                }, step=global_step)
           
            
            if done: 
                break
        
        print(f"Episode {ep}: Reward={ep_reward:.2f}, Length={ep_len}, Global Step={global_step}")
        if use_wandb: 
            wandb.log({"episode": ep, "total_reward": ep_reward, "ep_length": ep_len}, step=global_step)
    
    agent.actor_params, agent.critic1_params, agent.critic2_params, \
    agent.target_actor_params, agent.target_critic1_params, agent.target_critic2_params = params
    
    return agent, [], [], [], [], [], []
