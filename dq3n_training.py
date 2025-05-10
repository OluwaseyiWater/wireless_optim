
import os
import pickle
import hydra
from omegaconf import DictConfig
from flax import serialization
from models.D3QN import train_d3qn
from wireless_optim.environment import HetNetEnvironment

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Create environment
    env = HetNetEnvironment(
        num_macro_bs=cfg.env.num_macro_bs,
        num_small_bs=cfg.env.num_small_bs,
        num_users=cfg.env.num_users,
        max_steps=cfg.env.max_steps
    )

    print("\nStarting D3QN training...")
    # Train agent
    d3qn_agent, d3qn_rewards, d3qn_losses,  episode_powers, episode_bandwidths, episode_scheds = train_d3qn(
        env,
        num_episodes=cfg.d3qn.num_episodes,
        batch_size=cfg.d3qn.batch_size,
        replay_capacity=cfg.d3qn.replay_capacity,
        seed=cfg.training.seed,
        lr=cfg.d3qn.lr
    )
    print("D3QN training completed.")

    # Save outputs to original working directory
    out_dir = os.path.join(hydra.utils.get_original_cwd(), "train_models")
    os.makedirs(out_dir, exist_ok=True)

    # Save model parameters
    with open(os.path.join(out_dir, "d3qn_agent.pkl"), "wb") as f:
        f.write(serialization.to_bytes(d3qn_agent.params))
        
    # Save rewards & losses
    with open(os.path.join(out_dir, "d3qn_rewards.pkl"), "wb") as f:
        pickle.dump(d3qn_rewards, f)
    with open(os.path.join(out_dir, "d3qn_losses.pkl"), "wb") as f:
        pickle.dump(d3qn_losses, f)
    with open(os.path.join(out_dir, "episode_powers.pkl"), "wb") as f:
        pickle.dump(episode_powers, f)
    with open(os.path.join(out_dir, "episode_bandwidths.pkl"), "wb") as f:
        pickle.dump(episode_bandwidths, f)
    with open(os.path.join(out_dir, "episode_scheds.pkl"), "wb") as f:
        pickle.dump(episode_scheds, f)

if __name__ == "__main__":
    main()