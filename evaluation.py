from wireless_optim.environment import *
from models.PPO import *
from models.D3QN import *
from plot.utils import *
import pickle
import numpy as np
from flax import serialization
import jax

def main():
    with open('train_models/env_ppo.pkl', 'rb') as f:
        ppo_env = pickle.load(f)
    with open('train_models/env_d3qn.pkl', 'rb') as f:
        d3qn_env = pickle.load(f)
    
    env_ppo = HetNetEnvironment(**ppo_env)
    env_d3qn = HetNetEnvironment(**d3qn_env)

    d3qn_agent = D3QN(env_d3qn)
    d3qn_agent_params = d3qn_agent.params

    # Use tree_util.tree_map instead of jax.tree_map (works with any JAX version)
    from jax import tree_util
    d3qn_agent_params_empty = tree_util.tree_map(lambda _: None, d3qn_agent_params)
    
    # Load D3QN agent parameters
    with open('train_models/d3qn_agent.pkl', 'rb') as f:
        try:
            serialized_bytes = f.read()
            d3qn_agent_params = serialization.from_bytes(d3qn_agent_params_empty, serialized_bytes)
        except Exception:
            # If that fails, try resetting file pointer and using pickle directly
            f.seek(0)
            d3qn_agent_params = pickle.load(f)
    
    # Initialize PPO parameters empty tree using the updated method
    ppo_params_empty = tree_util.tree_map(lambda _: None, d3qn_agent_params)  # Using d3qn as placeholder
    
    # Load PPO parameters
    with open('train_models/ppo_params.pkl', 'rb') as f:
        try:
            ppo_params = serialization.from_bytes(ppo_params_empty, f.read())
        except Exception:
            # If that fails, try resetting file pointer and using pickle directly
            f.seek(0)
            ppo_params = pickle.load(f)

    # Enhanced Evaluation.
    print("\nEvaluating PPO...")
    ppo_eval = evaluate_agent(env_ppo, ppo_params, is_ppo=True)
    print(f"PPO Average Reward: {np.mean(ppo_eval.episode_rewards):.2f}")
    
    print("\nEvaluating D3QN...")
    # For D3QN, pass the agent's network to ensure matching architecture.
    d3qn_eval = evaluate_agent(env_d3qn, d3qn_agent_params, is_ppo=False, network=d3qn_agent.net)
    print(f"D3QN Average Reward: {np.mean(d3qn_eval.episode_rewards):.2f}")
    
    # Visualization.
    plot_action_distribution(ppo_eval.actions, d3qn_eval.actions)
    plot_metrics_comparison(ppo_eval.metrics, d3qn_eval.metrics)
    
    # Plot terrain visualization for evaluation episodes.
    plot_terrain(env_ppo, ppo_eval.states[:100], ppo_eval.actions[:100], episode=0)
    plot_terrain(env_d3qn, d3qn_eval.states[:100], d3qn_eval.actions[:100], episode=1)
    
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