import os
import pickle
from models.D3QN import train_d3qn
from wireless_optim.environment import HetNetEnvironment
#parse parameters    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_macro_bs", type=int, default=3)
parser.add_argument("--num_small_bs", type=int, default=10)
parser.add_argument("--num_users", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--env_seed", type=int, default=0)
parser.add_argument("--train_seed", type=int, default=0)
parser.add_argument("--num_episodes", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--replay_capacity", type=int, default=10000)

args = parser.parse_args()
num_macro_bs = args.num_macro_bs
num_small_bs = args.num_small_bs
num_users = args.num_users
max_steps = args.max_steps
env_seed = args.env_seed
train_seed = args.train_seed
num_episodes = args.num_episodes
batch_size = args.batch_size
replay_capacity = args.replay_capacity

def main():
    env = HetNetEnvironment(num_macro_bs=num_macro_bs, num_small_bs=num_small_bs,
                    num_users=num_users, max_steps=max_steps,seed=env_seed)

    print("\nStarting D3QN training...")
    d3qn_agent, d3qn_rewards = train_d3qn(env, num_episodes=num_episodes, 
                batch_size=batch_size, replay_capacity=replay_capacity
                ,seed=train_seed)
    print("D3QN training completed.")

    #save model
    if not os.path.exists("train_models"):
        os.makedirs("train_models")
    with open("train_models/d3qn_agent.pkl", "wb") as f:
        pickle.dump(d3qn_agent, f)

    #save rewards
    with open("train_models/d3qn_rewards.pkl", "wb") as f:
        pickle.dump(d3qn_rewards, f)

    #save parameters
    with open("train_models/d3qn_params.pkl", "w") as f:
        param_dict = {
            "num_macro_bs": num_macro_bs,
            "num_small_bs": num_small_bs,
            "num_users": num_users,
            "max_steps": max_steps,
            "env_seed": env_seed,
            "train_seed": train_seed,
            "num_episodes": num_episodes,
            "batch_size": batch_size,
            "replay_capacity": replay_capacity
        }
        pickle.dump(param_dict, f)

    with open("train_models/env_d3qn.pkl", "rb") as f:
        env_dict = {
            "num_macro_bs": num_macro_bs,
            "num_small_bs": num_small_bs,
            "num_users": num_users,
            "max_steps": max_steps,
            "env_seed": env_seed,
        }

        pickle.dump(env_dict, f)
        
    print("D3QN model and rewards saved.")

if __name__ == "__main__":
    main()