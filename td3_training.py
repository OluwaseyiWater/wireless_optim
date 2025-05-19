import hydra
from omegaconf import DictConfig
import os #
import pickle 
from flax import serialization 
from models.TD3 import train_td3
from wireless_optim.environment import HetNetEnvironment
# --- Hydra Main Function ---
@hydra.main(version_base="1.2", config_path="conf", config_name="config") # Specify version_base and config path/name
def main(cfg: DictConfig):
    """
    Main training function using Hydra configuration.
    """

    # Create environment using parameters from the config
    env = HetNetEnvironment(
        num_macro_bs=cfg.env.num_macro_bs,
        num_small_bs=cfg.env.num_small_bs,
        num_users=cfg.env.num_users,
        max_steps=cfg.env.max_steps
    )

    print(f"\nEnvironment Specs:")
    print(f"  Observation Spec: {env.observation_spec()}")
    print(f"  Action Spec: {env.action_spec()}")


    print("\nStarting TD3+PER training...")
    # Train agent using parameters from the config
    agent, rewards, actor_losses, critic_losses, powers, bandwidths, scheds = train_td3(
        env=env, # Pass the environment instance
        num_episodes=cfg.training.num_episodes,
        batch_size=cfg.training.batch_size,
        replay_capacity=cfg.td3.buffer.replay_capacity,
        seed=cfg.training.seed,
        # Agent parameters
        lr_actor=cfg.td3.agent.lr_actor,
        lr_critic=cfg.td3.agent.lr_critic,
        gamma=cfg.td3.agent.gamma,
        tau=cfg.td3.agent.tau,
        policy_noise=cfg.td3.agent.policy_noise,
        noise_clip=cfg.td3.agent.noise_clip,
        policy_delay=cfg.td3.agent.policy_delay,
        action_noise_std=cfg.td3.agent.action_noise_std,
        # Buffer parameters (related to training)
        per_alpha=cfg.td3.buffer.per_alpha,
        per_beta_start=cfg.td3.buffer.per_beta_start,
        per_beta_frames=cfg.td3.buffer.per_beta_frames,
        warmup_steps=cfg.td3.buffer.warmup_steps,
        # WandB parameters
        wandb_project=cfg.td3.wandb.project,
        wandb_name=cfg.td3.wandb.name,
        use_wandb=cfg.td3.wandb.use,
    )
    print("TD3+PER training completed.")

    run_dir_name = hydra.core.hydra_config.HydraConfig.get().run.dir.split('/')[-1]
    out_dir = os.path.join("train_results", run_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving results to: {os.path.abspath(out_dir)}") 

    # Save model parameters 
    try:
        with open(os.path.join(out_dir, "td3_agent_actor_params.pkl"), "wb") as f:
            pickle.dump(agent.actor_params, f) # Save actor params
        with open(os.path.join(out_dir, "td3_agent_critic1_params.pkl"), "wb") as f:
            pickle.dump(agent.critic1_params, f) # Save critic1 params
        with open(os.path.join(out_dir, "td3_agent_critic2_params.pkl"), "wb") as f:
            pickle.dump(agent.critic2_params, f) # Save critic2 params
        print("Agent parameters saved.")
    except Exception as e:
        print(f"Warning: Could not save Haiku params directly with pickle. Error: {e}")
        print("Consider using JAX/Flax serialization methods if needed for reliable reloading.")


    # Save rewards & losses
    with open(os.path.join(out_dir, "episode_rewards.pkl"), "wb") as f:
        pickle.dump(rewards, f)
    with open(os.path.join(out_dir, "episode_actor_losses.pkl"), "wb") as f:
        pickle.dump(actor_losses, f)
    with open(os.path.join(out_dir, "episode_critic_losses.pkl"), "wb") as f:
        pickle.dump(critic_losses, f)
    with open(os.path.join(out_dir, "episode_powers.pkl"), "wb") as f:
        pickle.dump(powers, f)
    with open(os.path.join(out_dir, "episode_bandwidths.pkl"), "wb") as f:
        pickle.dump(bandwidths, f)
    with open(os.path.join(out_dir, "episode_scheds.pkl"), "wb") as f:
        pickle.dump(scheds, f)
    print("Training logs saved.")

    # Save the full Hydra config for this run for reproducibility
    try:
        import yaml
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print("Run configuration saved.")
    except ImportError:
        print("Warning: PyYAML not installed. Skipping saving the full config to YAML.")


# --- Script Entry Point ---
if __name__ == "__main__":
    
    main()
