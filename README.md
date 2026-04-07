This repository contains code for our paper on the optimisation of resource allocation in heterogeneous wireless networks using deep reinforcement learning. The code is designed to optimise resource allocation in a dynamic wireless environment with changing network conditions and user demands.

This project explores the application of deep reinforcement learning (DRL) for dynamic resource allocation in wireless communication systems. An environment simulates a heterogeneous network with path loss fading and log-normal shadowing. DRL algorithms, such as TD3 and PPO, are used to optimise resource allocation, demonstrating improved efficiency over heuristic baselines. For more details, refer to the [paper](./paper/conference_latex_template%20(4).pdf). Note that the paper is accepted at the 2026 EuCNC & 6G Summit, for presentation.

## Installation

To install the necessary dependencies and set up the project, follow these steps:

### Clone the repository

```bash
git clone https://github.com/OluwaseyiWater/wireless_optim.git
cd wireless_optim
```

### Create a new virtual environment

It is recommended to create a new virtual environment to avoid conflicts with other projects.

```bash
pip install -r requirements.txt
import wandb (#optional)
```
## Training

### PPO
```python
python3 ppo_training.py --multirun training.seed=0,10,18,28,42,64,128,256,512,1024 ppo.gamma=0.99 ppo.num_epochs=1000 ppo.gae_lambda=0.95 ppo.clip_coef=0.2 ppo.ent_coef=0.01 ppo.vf_coef=0.5 ppo.hidden_size=256 ppo.lr=2e-5
```

### TD3
```python
python3 td3_training.py --multirun training.seed=28,42,64,128,256,512,1024 td3.agent.lr_actor=3e-4 td3.agent.lr_critic=1e-5 td3.num_episodes=1000
```

### Heuristic Baselines
```python
python3 heuristics_baselines.py
```

### Multiple Network Scenarios
```python
python3 run_multiscenarios.py --seeds 10 --eval_steps 300 --ppo_epochs 200 --ppo_steps 1024 --td3_episodes 200
```
## Contributing

Contributions are always welcome and highly appreciated.
