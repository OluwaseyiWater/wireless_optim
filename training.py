from train.PPO_training import *
from train.D3QN_training import *
from wireless_optim.environment import *


if __name__ == "__main__":
    # Train PPO using the library's provided training function
    print("Training PPO...")
    ppo_params = ppo_training_setup()
    
    # Train D3QN on the HetNet environment
    print("\nTraining D3QN...")
    hetnet_env = HetNetEnvironment()
    train_d3qn(hetnet_env, num_episodes=100) 