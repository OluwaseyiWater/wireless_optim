import os
import pickle
import hydra
from omegaconf import DictConfig
from flax import serialization
from models.D3QN import train_d3qn
from wireless_optim.environment import HetNetEnvironment


# --- Hydra Main Function ---
@hydra.main(version_base="1.2", config_path="conf", config_name="config") # Specify version_base and config path/name
def main(cfg: DictConfig):
    """
    Main training function using Hydra configuration for D3QN.
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
    # Assuming HetNetEnvironment's action_spec is continuous,
    # D3QN will discretize internally based on num_bins_per_dimension
    print(f"  Action Spec (Continuous Env): {env.action_spec()}")


    print("\nStarting D3QN training...")
    # Train agent using parameters from the config
    # Pass parameters from the cfg object to train_d3qn
    agent, rewards, losses, powers, bandwidths, scheds = train_d3qn(
        env=env, # Pass the environment instance
        num_episodes=cfg.d3qn.training.num_episodes,
        batch_size=cfg.d3qn.training.batch_size,
        replay_capacity=cfg.d3qn.buffer.replay_capacity,
        seed=cfg.training.seed,
        # Agent parameters (using cfg.agent)
        lr=cfg.d3qn.agent.lr,
        gamma=cfg.d3qn.agent.gamma,
        target_update_freq=cfg.d3qn.agent.target_update_freq,
        tau=cfg.d3qn.agent.tau,
        epsilon_start=cfg.d3qn.agent.epsilon_start,
        epsilon_end=cfg.d3qn.agent.epsilon_end,
        epsilon_decay_steps=cfg.d3qn.agent.epsilon_decay_steps,
        # Buffer parameters (using cfg.buffer)
        warmup_steps=cfg.d3qn.buffer.warmup_steps,
        # WandB parameters (using cfg.wandb)
        wandb_project=cfg.d3qn.wandb.project,
        wandb_name=cfg.d3qn.wandb.name,
        use_wandb=cfg.d3qn.wandb.use,
    )
    print("D3QN training completed.")

    # --- Saving Logic using Hydra's working directory ---
    # Hydra creates a new working directory for each run (e.g., outputs/YYYY-MM-DD/HH-MM-SS)
    # Save results within this run-specific directory.
    run_dir_name = hydra.core.hydra_config.HydraConfig.get().run.dir.split('/')[-1]
    out_dir = os.path.join("train_results", run_dir_name) # Create a subdir within the run directory
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving results to: {os.path.abspath(out_dir)}") # Use abspath to see the full path

    # Save model parameters (Haiku params are PyTrees, pickle should work directly)
    try:
        with open(os.path.join(out_dir, "d3qn_agent_params.pkl"), "wb") as f:
            pickle.dump(agent.params, f) # Save D3QN params
        with open(os.path.join(out_dir, "d3qn_agent_target_params.pkl"), "wb") as f:
            pickle.dump(agent.target_params, f) # Save D3QN target params
        print("Agent parameters saved.")
    except Exception as e:
        print(f"Warning: Could not save Haiku params directly with pickle. Error: {e}")
        print("Consider using JAX/Flax serialization methods if needed for reliable reloading.")


    # Save rewards & losses
    with open(os.path.join(out_dir, "episode_rewards.pkl"), "wb") as f:
        pickle.dump(rewards, f)
    with open(os.path.join(out_dir, "episode_losses.pkl"), "wb") as f: # Corrected to use 'losses' variable
        pickle.dump(losses, f)
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
            yaml.dump(cfg.pretty(), f, default_flow_style=False) # Use cfg.pretty() for clean dump
        print("Run configuration saved.")
    except ImportError:
        print("Warning: PyYAML not installed. Skipping saving the full config to YAML.")
    except Exception as e:
         print(f"Warning: Could not save config to YAML. Error: {e}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # The @hydra.main decorator handles parsing arguments and calling main(cfg)
    main()
