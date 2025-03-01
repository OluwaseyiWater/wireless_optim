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


def ppo_training_setup():
    """Configure PPO for HetNet resource allocation"""
    env = HetNetEnvironment(num_macro_bs=3, num_small_bs=10, num_users=50)
    
    # Hyperparameters
    config = {
        'num_envs': 2048,          # Parallel environments
        'num_steps': 100,           # Steps per environment
        'num_epochs': 10,           # Training epochs
        'lr': 3e-4,
        'anneal_lr': True,
        'gae': True,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_coef': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'update_epochs': 4,
        'activation': 'tanh',
        'hidden_size': 64,
    }

    # Network architecture: mapping observation to actions
    def network_fn(obs):
        mlp = hk.Sequential([
            hk.Linear(config['hidden_size']), jax.nn.tanh,
            hk.Linear(config['hidden_size']), jax.nn.tanh,
            hk.Linear(env.action_spec().shape[0] * env.action_spec().shape[1])
        ])
        return mlp(obs)

    # Train the agent using the provided PPO training function
    trained_params = ppo_train(
        env,
        network_fn,
        num_envs=config['num_envs'],
        num_steps=config['num_steps'],
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        anneal_lr=config['anneal_lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_coef=config['clip_coef'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        update_epochs=config['update_epochs']
    )
    
    return trained_params