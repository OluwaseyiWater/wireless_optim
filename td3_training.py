import os
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from flax import serialization
import chex
import yaml
import wandb
from models.TD3 import train_td3
from wireless_optim.environment import HetNetEnvironment

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

    
    if cfg.wandb.use:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=config_dict
        )

    print("\nStarting TD3+PER training...")
    agent, rewards, actor_losses, critic_losses, powers, bandwidths, scheds = train_td3(
        env=env,
        num_episodes=cfg.td3.num_episodes,
        batch_size=cfg.td3.batch_size,
        lr_actor=cfg.td3.agent.lr_actor,
        lr_critic=cfg.td3.agent.lr_critic,
        gamma=cfg.td3.agent.gamma,
        tau=cfg.td3.agent.tau,
        policy_noise=cfg.td3.agent.policy_noise,
        noise_clip=cfg.td3.agent.noise_clip,
        policy_delay=cfg.td3.agent.policy_delay,
        action_noise_std=cfg.td3.agent.action_noise_std,
        seed=cfg.training.seed,
        wandb_project=cfg.wandb.project,
        wandb_name=cfg.wandb.name,
        use_wandb=cfg.wandb.use,
        **cfg.td3.buffer
    )
    print("TD3+PER training completed.")

    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train_models_dir = os.path.join(output_dir, "train_models")
    os.makedirs(train_models_dir, exist_ok=True)
    print(f"Saving artifacts to: {train_models_dir}")

    
    with open(os.path.join(train_models_dir, "td3_agent_actor_params.pkl"), "wb") as f:
        f.write(serialization.to_bytes(agent.actor_params))
    with open(os.path.join(train_models_dir, "td3_agent_critic1_params.pkl"), "wb") as f:
        f.write(serialization.to_bytes(agent.critic1_params))
    with open(os.path.join(train_models_dir, "td3_agent_critic2_params.pkl"), "wb") as f:
        f.write(serialization.to_bytes(agent.critic2_params))
    print("Agent parameters saved reliably.")

    
    with open(os.path.join(train_models_dir, "episode_rewards.pkl"), "wb") as f:
        pickle.dump(rewards, f)
    with open(os.path.join(train_models_dir, "episode_actor_losses.pkl"), "wb") as f:
        pickle.dump(actor_losses, f)
    with open(os.path.join(train_models_dir, "episode_critic_losses.pkl"), "wb") as f:
        pickle.dump(critic_losses, f)
    with open(os.path.join(train_models_dir, "episode_powers.pkl"), "wb") as f:
        pickle.dump(powers, f)
    with open(os.path.join(train_models_dir, "episode_bandwidths.pkl"), "wb") as f:
        pickle.dump(bandwidths, f)
    with open(os.path.join(train_models_dir, "episode_scheds.pkl"), "wb") as f:
        pickle.dump(scheds, f)
    print("Training metrics saved.")

    
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)
    print("Run configuration saved.")

    if cfg.wandb.use:
        wandb.finish()

if __name__ == "__main__":
    main()
