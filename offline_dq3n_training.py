import os
import pickle
from models.D3QN import train_d3qn, D3QN
from wireless_optim.environment import HetNetEnvironment
#parse parameters    
import argparse
from flax import serialization
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--num_macro_bs", type=int, default=3)
parser.add_argument("--num_small_bs", type=int, default=10)
parser.add_argument("--num_users", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--train_seed", type=int, default=0)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--replay_capacity", type=int, default=10000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--pretrain_epochs', type=int, default=1000)

args = parser.parse_args()
num_macro_bs = args.num_macro_bs
num_small_bs = args.num_small_bs
num_users = args.num_users
max_steps = args.max_steps
train_seed = args.train_seed
num_episodes = args.num_episodes
batch_size = args.batch_size
replay_capacity = args.replay_capacity
gamma = args.gamma
lr = args.lr
pretrain_epochs = args.pretrain_epochs

def generate_synthetic_dataset(num_transitions=10000, num_macro_bs=3, num_small_bs=10, num_users=50):
    """
    Generate synthetic time-series transitions that include a rich set of features.
    Each transition is a dict with keys: state, action, reward, next_state, done.
    The state dimension is set to match the environment's observation_spec.
    """
    dataset = []
    num_bs = num_macro_bs + num_small_bs
    state_size = 839  # Must match the environment's observation dimension
    action_shape = (num_bs, 3)  # each BS outputs a 3-dimensional action

    for i in range(num_transitions):
        state = np.random.rand(state_size).astype(np.float32)
        action = np.random.rand(*action_shape).astype(np.float32)
        reward = float(np.random.uniform(-1, 1))
        next_state = np.random.rand(state_size).astype(np.float32)
        done = bool(np.random.choice([False, True], p=[0.9, 0.1]))
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
        dataset.append(transition)
    return dataset


def main():
    env = HetNetEnvironment(num_macro_bs=num_macro_bs, num_small_bs=num_small_bs,
                    num_users=num_users, max_steps=max_steps)

    dataset = generate_synthetic_dataset()

    # Initialize the D3QN agent using the online environment (for state dimension).
    d3qn_agent = D3QN(env, gamma=0.99, lr=lr)

    # Offline pretraining using the synthetic dataset.
    offline_pretrain_policy(dataset, d3qn_agent, pretrain_epochs=pretrain_epochs, batch_size=batch_size)

    # Train the D3QN agent using the online environment.
    print("\nStarting D3QN training...")
    d3qn_agent, d3qn_rewards = train_d3qn_online(online_env, d3qn_agent, num_episodes=num_episodes, batch_size=batch_size)
    print("D3QN training completed.")

    #save model
    if not os.path.exists("train_models"):
        os.makedirs("train_models")
    with open("train_models/offline_d3qn_agent.pkl", "wb") as f:
        f.write(serialization.to_bytes(d3qn_agent.params))

    #save rewards
    with open("train_models/offline_d3qn_rewards.pkl", "wb") as f:
        pickle.dump(d3qn_rewards, f)

    #save parameters
    with open("train_models/offline_d3qn_params.pkl", "wb") as f:
        param_dict = {
            "num_macro_bs": num_macro_bs,
            "num_small_bs": num_small_bs,
            "num_users": num_users,
            "max_steps": max_steps,
            "train_seed": train_seed,
            "num_episodes": num_episodes,
            "batch_size": batch_size,
            "replay_capacity": replay_capacity
        }
        pickle.dump(param_dict, f)

    with open("train_models/env_offline_d3qn.pkl", "wb") as f:
        env_dict = {
            "num_macro_bs": num_macro_bs,
            "num_small_bs": num_small_bs,
            "num_users": num_users,
            "max_steps": max_steps,
            
        }

        pickle.dump(env_dict, f)
        
    print("D3QN model and rewards saved.")

if __name__ == "__main__":
    main()