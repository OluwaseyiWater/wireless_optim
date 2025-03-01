from matplotlib import pyplot as plt
import pickle


def main():

  # Load training data
    with open("train_models/ppo_rewards.pkl", "rb") as f:
        ppo_rewards = pickle.load(f)
    with open("train_models/d3qn_rewards.pkl", "rb") as f:
        d3qn_rewards = pickle.load(f)
    with open("train_models/ppo_losses.pkl", "rb") as f:
        ppo_losses = pickle.load(f)

      # Save training plots
    
    plt.figure(figsize=(10, 5))
    plt.plot(ppo_rewards, label='PPO Rewards')
    plt.plot(d3qn_rewards, label='D3QN Rewards')
    plt.xlabel('Epoch/Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress Comparison')
    plt.legend()
    plt.savefig("plots/training_progress_comparison.png")
    plt.close()
    
    if ppo_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(ppo_losses, label='PPO Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PPO Loss over Training')
        plt.legend()
        plt.savefig("plots/ppo_loss_over_training.png")
        plt.close()
    