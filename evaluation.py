from wireless_optim.environment import *
from train.PPO_training import *
from train.D3QN_training import *

def main():
     # Enhanced Evaluation.
    print("\nEvaluating PPO...")
    ppo_eval = evaluate_agent(env, ppo_params, is_ppo=True)
    print(f"PPO Average Reward: {np.mean(ppo_eval.episode_rewards):.2f}")
    
    print("\nEvaluating D3QN...")
    # For D3QN, pass the agent's network to ensure matching architecture.
    d3qn_eval = evaluate_agent(env, d3qn_agent.params, is_ppo=False, network=d3qn_agent.net)
    print(f"D3QN Average Reward: {np.mean(d3qn_eval.episode_rewards):.2f}")
    
    # Visualization.
    plot_action_distribution(ppo_eval.actions, d3qn_eval.actions)
    plot_metrics_comparison(ppo_eval.metrics, d3qn_eval.metrics)
    
    # Plot terrain visualization for evaluation episodes.
    plot_terrain(env, ppo_eval.states[:100], ppo_eval.actions[:100], episode=0)
    plot_terrain(env, d3qn_eval.states[:100], d3qn_eval.actions[:100], episode=1)
    
    # Plot cumulative evaluation rewards.
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(ppo_eval.episode_rewards), label='PPO')
    plt.plot(np.cumsum(d3qn_eval.episode_rewards), label='D3QN')
    plt.title("Cumulative Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig('evaluation_rewards.png')
    plt.close()


if __name__ == "__main__":
    main()