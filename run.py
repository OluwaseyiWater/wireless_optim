from wireless_optim.environment import *

if __name__ == "__main__":
    env = HetNetEnvironment()
    key = jax.random.PRNGKey(0)
    timestep = env.reset(key)
    
    # Run for a few steps using random actions from the action spec
    for step in range(1000):
        action = env.action_spec().generate_value()
        timestep = env.step(action)
        print(f"Step {step}: Reward {timestep.reward:.2f}")