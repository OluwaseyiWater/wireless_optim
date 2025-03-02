from wireless_optim.environment import *
from models.PPO import ppo_train
import argparse
import jax.numpy as jnp
import pickle
import os
from flax import serialization
import orbax.checkpoint as ocp


# parse from terminal input
parser = argparse.ArgumentParser()
parser.add_argument('--num_macro_bs', type=int, default=3)
parser.add_argument('--num_small_bs', type=int, default=10)
parser.add_argument('--num_users', type=int, default=50)
parser.add_argument('--max_steps', type=int, default=100)
parser.add_argument('--train_seed', type=int, default=42)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--num_steps_per_episode', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--num_steps', type=int, default=50)
parser.add_argument('--gae_lambda', type=float, default=0.95)
parser.add_argument('--update_epochs', type=int, default=4)


args = parser.parse_args()
num_macro_bs = args.num_macro_bs
num_small_bs = args.num_small_bs
num_users = args.num_users
max_steps = args.max_steps
training_seed = args.train_seed
num_episodes = args.num_episodes
num_steps_per_episode = args.num_steps_per_episode
gamma = args.gamma
lr = args.lr
hidden_size = args.hidden_size
num_epochs = args.num_epochs
num_steps = args.num_steps
gae_lambda = args.gae_lambda
update_epochs = args.update_epochs


def main():
    env = HetNetEnvironment(num_macro_bs=num_macro_bs,
                 num_small_bs=num_small_bs, num_users=num_users, 
                 max_steps=max_steps)
    
    # PPO Configuration.
    ppo_config = {
        'num_envs': 1,
        'num_steps': num_steps,
        'num_epochs': num_epochs,
        'lr': lr,
        'anneal_lr': True,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_coef': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'update_epochs': update_epochs,
        'hidden_size': hidden_size,
       
    }
    
    print("Starting PPO training...")
    ppo_params, ppo_net, ppo_rewards, ppo_losses = ppo_train(env, ppo_config, seed=training_seed)
    print("PPO training completed.")

    if not os.path.exists("train_models"):
        os.makedirs("train_models")
    
    # Save PPO parameters and network to .pkl files
    with open('train_models/ppo_params.pkl', 'wb') as f:
       f.write(serialization.to_bytes(ppo_params))
    with open('train_models/ppo_rewards.pkl', 'wb') as f:
        pickle.dump(ppo_rewards, f)
    with open('train_models/ppo_losses.pkl', 'wb') as f:
        pickle.dump(ppo_losses, f)

    print("PPO parameters and network saved to .pkl files.")
    
    #save config parameters to .pkl file
    with open('train_models/ppo_config.pkl', 'wb') as f:
        pickle.dump(ppo_config, f)
    print("PPO config parameters saved to .pkl file.")

    #save environmnet arguments to .pkl file
    with open('train_models/env_ppo.pkl', 'wb') as f:
        ppo_config = {'num_macro_bs': num_macro_bs, 'num_small_bs': num_small_bs,
         'num_users': num_users, 'max_steps': max_steps}
        pickle.dump(ppo_config, f)

    print("PPO environment arguments saved to .pkl file.")


if __name__ == "__main__":
    main()