import os
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from flax import serialization

from wireless_optim.environment import HetNetEnvironment
from models.PPO import ppo_train 


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    
    env = HetNetEnvironment(
        num_macro_bs=cfg.env.num_macro_bs,
        num_small_bs=cfg.env.num_small_bs,
        num_users=cfg.env.num_users,
        max_steps=cfg.env.max_steps
    )

    
    ppo_config = {
        'num_steps': cfg.ppo.num_steps,
        'num_epochs': cfg.ppo.num_epochs,
        'lr': cfg.ppo.lr,
        'gamma': cfg.ppo.gamma,
        'gae_lambda': cfg.ppo.gae_lambda,
        'clip_coef': cfg.ppo.clip_coef,
        'ent_coef': cfg.ppo.ent_coef,
        'vf_coef': cfg.ppo.vf_coef,
        'update_epochs': cfg.ppo.update_epochs,
        'hidden_size': cfg.ppo.hidden_size,
        'max_grad_norm': cfg.ppo.max_grad_norm,
    }


    print("Starting PPO training...")
    
    ppo_params, ppo_net, ppo_rewards, ppo_losses, episode_powers, episode_bandwidths, episode_scheds = ppo_train(
        env,
        ppo_config,
        seed=cfg.training.seed,
        use_wandb=cfg.wandb.use, 
        wandb_project=cfg.wandb.project,
        wandb_name=cfg.wandb.name
    )
    print("PPO training completed.")

    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train_models_dir = os.path.join(output_dir, "train_models")
    os.makedirs(train_models_dir, exist_ok=True)
    print(f"Saving artifacts to: {train_models_dir}")

    
    with open(os.path.join(train_models_dir, "ppo_params.pkl"), "wb") as f:
        f.write(serialization.to_bytes(ppo_params))

    
    with open(os.path.join(train_models_dir, "ppo_rewards.pkl"), "wb") as f:
        pickle.dump(ppo_rewards, f)
    with open(os.path.join(train_models_dir, "ppo_losses.pkl"), "wb") as f:
        pickle.dump(ppo_losses, f)
        
    
    with open(os.path.join(train_models_dir, "episode_powers.pkl"), "wb") as f:
        pickle.dump(episode_powers, f)
    with open(os.path.join(train_models_dir, "episode_bandwidths.pkl"), "wb") as f:
        pickle.dump(episode_bandwidths, f)
    with open(os.path.join(train_models_dir, "episode_scheds.pkl"), "wb") as f:
        pickle.dump(episode_scheds, f)
        
    print("PPO parameters and training metrics saved.")


    with open(os.path.join(train_models_dir, "run_config.pkl"), "wb") as f:
        pickle.dump(OmegaConf.to_container(cfg, resolve=True), f)
        
    print("Run configuration saved.")


if __name__ == "__main__":
    main()
