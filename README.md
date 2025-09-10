# wireless_optim
```python
pip install -r requirements.txt

```python
python3 ppo_training.py --multirun training.seed=0,10,18,28,42,64,128,256,512,1024 ppo.gamma=0.99 ppo.num_epochs=1000 ppo.gae_lambda=0.95 ppo.clip_coef=0.2 ppo.ent_coef=0.01 ppo.vf_coef=0.5 ppo.hidden_size=256 ppo.lr=2e-5

python3 td3_training.py --multirun training.seed=28,42,64,128,256,512,1024 td3.agent.lr_actor=3e-4 td3.agent.lr_critic=1e-5 td3.num_episodes=1000
