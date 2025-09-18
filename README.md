<p align="center">
  <img src="https://raw.githubusercontent.com/muhd-umer/rl-wireless/refs/heads/main/resources/logo.png" width="675">
</p>

This repository contains code for our paper on optimisation of resource allocation in heterogeneous wireless networks using deep reinforcement learning. The code is designed to optimize resource allocation in a dynamic wireless environment with changing network conditions and user demands.

The application of deep reinforcement learning (DRL) for dynamic resource allocation in wireless communication systems is explored in this project. An environment simulates an heterogeneous network with path loss fading and log-normal shadowing. DRL algorithms such as TD3 and PPO are used to optimise resource allocation, demonstrating improved efficiency over heuristics baselines. For more details, refer to the [paper](./paper/aims_project.pdf).

## Installation

To install the necessary dependencies and set up the project, follow these steps:

### Clone the repository

```shell
git clone https://github.com/OluwaseyiWater/wireless_optim.git
cd wireless_optim
```

### Create a new virtual environment

It is recommended to create a new virtual environment to avoid conflicts with other projects.

```shell
pip install -r requirements.txt
import wandb (#optional)
```
## Training

### PPO
```shell
python3 ppo_training.py --multirun training.seed=0,10,18,28,42,64,128,256,512,1024 ppo.gamma=0.99 ppo.num_epochs=1000 ppo.gae_lambda=0.95 ppo.clip_coef=0.2 ppo.ent_coef=0.01 ppo.vf_coef=0.5 ppo.hidden_size=256 ppo.lr=2e-5
```

### TD3
```shell
python3 td3_training.py --multirun training.seed=28,42,64,128,256,512,1024 td3.agent.lr_actor=3e-4 td3.agent.lr_critic=1e-5 td3.num_episodes=1000
```

### Heuristic Baselines
```shell
python3 heuristics_baselines.py
```

### Multiple Network Scenarios
```shell
!python run_multiscenarios.py --seeds 10 --eval_steps 300 --ppo_epochs 200 --ppo_steps 1024 --td3_episodes 200
```
## Contributing

Contributions are always welcome and highly appreciated.
