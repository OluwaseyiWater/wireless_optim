from wireless_optim.environment import HetNetEnvironment
from models.PPO import ppo_train
import jax.numpy as jnp
import pickle
import os
from flax import serialization
import hydra
from omegaconf import DictConfig, OmegaConf

# Hydra decorator to load config from a YAML file
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Print the resolved configuration (useful for debugging)
    print(OmegaConf.to_yaml(cfg))

    # Create the environment using config values
    env = HetNetEnvironment(
        num_macro_bs=cfg.env.num_macro_bs,
        num_small_bs=cfg.env.num_small_bs,
        num_users=cfg.env.num_users,
        max_steps=cfg.env.max_steps
    )

    # PPO Configuration from the config file
    ppo_config = {
        'num_envs': cfg.ppo.num_envs,
        'num_steps': cfg.ppo.num_steps,
        'num_epochs': cfg.ppo.num_epochs,
        'lr': cfg.ppo.lr,
        'anneal_lr': cfg.ppo.anneal_lr,
        'gamma': cfg.ppo.gamma,
        'gae_lambda': cfg.ppo.gae_lambda,
        'clip_coef': cfg.ppo.clip_coef,
        'ent_coef': cfg.ppo.ent_coef,
        'vf_coef': cfg.ppo.vf_coef,
        'max_grad_norm': cfg.ppo.max_grad_norm,
        'update_epochs': cfg.ppo.update_epochs,
        'hidden_size': cfg.ppo.hidden_size,
    }

    # Start PPO training
    print("Starting PPO training...")
    ppo_params, ppo_net, ppo_rewards, ppo_losses, episode_powers, episode_bandwidths, episode_scheds = ppo_train(
        env, ppo_config, seed=cfg.training.seed
    )
    print("PPO training completed.")

    # Ensure the output directory exists (Hydra manages output dirs automatically)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train_models_dir = os.path.join(output_dir, "train_models")
    os.makedirs(train_models_dir, exist_ok=True)

    # Save PPO parameters and network to .pkl files
    with open(os.path.join(train_models_dir, "ppo_params.pkl"), "wb") as f:
        f.write(serialization.to_bytes(ppo_params))
    with open(os.path.join(train_models_dir, "ppo_rewards.pkl"), "wb") as f:
        pickle.dump(ppo_rewards, f)
    with open(os.path.join(train_models_dir, "ppo_losses.pkl"), "wb") as f:
        pickle.dump(ppo_losses, f)
    print("PPO parameters and network saved to .pkl files.")

    # Save PPO config parameters
    with open(os.path.join(train_models_dir, "ppo_config.pkl"), "wb") as f:
        pickle.dump(ppo_config, f)
    print("PPO config parameters saved to .pkl file.")

    # Save environment arguments
    with open(os.path.join(train_models_dir, "env_ppo.pkl"), "wb") as f:
        env_config = {
            "num_macro_bs": cfg.env.num_macro_bs,
            "num_small_bs": cfg.env.num_small_bs,
            "num_users": cfg.env.num_users,
            "max_steps": cfg.env.max_steps
        }
        pickle.dump(env_config, f)
        
    with open(os.path.join(train_models_dir, "episode_powers.pkl"), "wb") as f:
        pickle.dump(episode_powers, f)
    with open(os.path.join(train_models_dir, "episode_bandwidths.pkl"), "wb") as f:
        pickle.dump(episode_bandwidths, f)
    with open(os.path.join(train_models_dir, "episode_scheds.pkl"), "wb") as f:
        pickle.dump(episode_scheds, f)
        
    print("PPO environment arguments saved to .pkl file.")


if __name__ == "__main__":
    main()

# from wireless_optim.environment import *
# from models.PPO import ppo_train
# import argparse
# import jax.numpy as jnp
# import pickle
# import os
# from flax import serialization
# import orbax.checkpoint as ocp


# # parse from terminal input
# parser = argparse.ArgumentParser()
# parser.add_argument('--num_macro_bs', type=int, default=3)
# parser.add_argument('--num_small_bs', type=int, default=10)
# parser.add_argument('--num_users', type=int, default=50)
# parser.add_argument('--max_steps', type=int, default=1000)
# parser.add_argument('--train_seed', type=int, default=42)
# parser.add_argument('--num_episodes', type=int, default=1000)
# parser.add_argument('--num_steps_per_episode', type=int, default=100)
# parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--lr', type=float, default=3e-4)
# parser.add_argument('--hidden_size', type=int, default=64)
# parser.add_argument('--num_epochs', type=int, default=10)
# parser.add_argument('--num_steps', type=int, default=4096)
# parser.add_argument('--gae_lambda', type=float, default=0.95)
# parser.add_argument('--update_epochs', type=int, default=4)
# parser.add_argument('--clip_coef', type=float, default=0.2)
# parser.add_argument('--ent_coef', type=float, default=0.05)
# parser.add_argument('--vf_coef', type=float, default=0.1)
# parser.add_argument('--max_grad_norm', type=float, default=0.3)


# args = parser.parse_args()
# num_macro_bs = args.num_macro_bs
# num_small_bs = args.num_small_bs
# num_users = args.num_users
# max_steps = args.max_steps
# training_seed = args.train_seed
# num_episodes = args.num_episodes
# num_steps_per_episode = args.num_steps_per_episode
# gamma = args.gamma
# lr = args.lr
# hidden_size = args.hidden_size
# num_epochs = args.num_epochs
# num_steps = args.num_steps
# gae_lambda = args.gae_lambda
# update_epochs = args.update_epochs
# clip_coef = args.clip_coef
# ent_coef = args.ent_coef
# vf_coef = args.vf_coef
# max_grad_norm = args.max_grad_norm


# def main():
#     env = HetNetEnvironment(num_macro_bs=num_macro_bs,
#                  num_small_bs=num_small_bs, num_users=num_users, 
#                  max_steps=max_steps)
    
#     # PPO Configuration.
#     ppo_config = {
#         'num_envs': 1,
#         'num_steps': num_steps,
#         'num_epochs': num_epochs,
#         'lr': lr,
#         'anneal_lr': True,
#         'gamma': gamma,
#         'gae_lambda': gae_lambda,
#         'clip_coef': clip_coef,
#         'ent_coef': ent_coef,
#         'vf_coef': vf_coef,
#         'max_grad_norm': max_grad_norm,
#         'update_epochs': update_epochs,
#         'hidden_size': hidden_size,
       
#     }
    
#     print("Starting PPO training...")
#     ppo_params, ppo_net, ppo_rewards, ppo_losses = ppo_train(env, ppo_config, seed=training_seed)
#     print("PPO training completed.")

#     if not os.path.exists("train_models"):
#         os.makedirs("train_models")
    
#     # Save PPO parameters and network to .pkl files
#     with open('train_models/ppo_params.pkl', 'wb') as f:
#        f.write(serialization.to_bytes(ppo_params))
#     with open('train_models/ppo_rewards.pkl', 'wb') as f:
#         pickle.dump(ppo_rewards, f)
#     with open('train_models/ppo_losses.pkl', 'wb') as f:
#         pickle.dump(ppo_losses, f)

#     print("PPO parameters and network saved to .pkl files.")
    
#     #save config parameters to .pkl file
#     with open('train_models/ppo_config.pkl', 'wb') as f:
#         pickle.dump(ppo_config, f)
#     print("PPO config parameters saved to .pkl file.")

#     #save environmnet arguments to .pkl file
#     with open('train_models/env_ppo.pkl', 'wb') as f:
#         ppo_config = {'num_macro_bs': num_macro_bs, 'num_small_bs': num_small_bs,
#          'num_users': num_users, 'max_steps': max_steps}
#         pickle.dump(ppo_config, f)

#     print("PPO environment arguments saved to .pkl file.")


# if __name__ == "__main__":
#     main()