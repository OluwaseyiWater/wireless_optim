import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb 

DEFAULT_SMOOTHING_WINDOW = 20



def load_wandb_data(run_path, reward_metric_key, loss_metric_keys, algorithm_name):
    """
    Loads rewards and losses for a given run_path from WandB.
    
    Args:
        run_path (str): The WandB run path (e.g., "entity/project/run_id").
        reward_metric_key (str): The WandB key for the reward metric.
        loss_metric_keys (list or str): WandB key(s) for the loss metric(s).
                                        If a list, losses will be summed.
        algorithm_name (str): Name of the algorithm (for print statements).

    Returns:
        tuple: (rewards_list, losses_list)
    """
    print(f"Fetching data for {algorithm_name} from WandB run: {run_path}...")
    api = wandb.Api()
    try:
        run = api.run(run_path)
    except wandb.errors.CommError as e:
        raise ValueError(
            f"Could not fetch run '{run_path}'. Ensure the path is correct and you have access. Original error: {e}"
        )

    history_keys = [reward_metric_key]
    if isinstance(loss_metric_keys, list):
        history_keys.extend(loss_metric_keys)
    else:
        history_keys.append(loss_metric_keys)

    print(f"Fetching keys: {history_keys}")
    history_df = run.history(keys=history_keys, pandas=True)

    if history_df.empty:
        raise ValueError(f"No history found for run '{run_path}' with keys '{history_keys}'. "
                         "Please check if the metrics were logged correctly.")

    if reward_metric_key not in history_df.columns:
        raise ValueError(f"Reward metric '{reward_metric_key}' not found in run '{run_path}'. Available keys: {history_df.columns.tolist()}")
    rewards = history_df[reward_metric_key].dropna().tolist()

    if isinstance(loss_metric_keys, list):
        total_loss = pd.Series(0.0, index=history_df.index)
        for key in loss_metric_keys:
            if key not in history_df.columns:
                 raise ValueError(f"Loss metric '{key}' not found in run '{run_path}'. Available keys: {history_df.columns.tolist()}")
            total_loss = total_loss + history_df[key].fillna(0)
        losses = total_loss.dropna().tolist()
    else:
        if loss_metric_keys not in history_df.columns:
            raise ValueError(f"Loss metric '{loss_metric_keys}' not found in run '{run_path}'. Available keys: {history_df.columns.tolist()}")
        losses = history_df[loss_metric_keys].dropna().tolist()
    
    if not rewards:
        print(f"Warning: No reward data found for {reward_metric_key} in run {run_path} after dropping NaNs.")
    if not losses:
        print(f"Warning: No loss data found for {loss_metric_keys} in run {run_path} after dropping NaNs.")

    print(f"Loaded {algorithm_name} data: {len(rewards)} reward points, {len(losses)} loss points.")
    return rewards, losses


def smooth_curve(data, window_size):
    """Applies a rolling average to smooth the curve."""
    if window_size <= 1 or len(data) < window_size:
        return np.asarray(data)
    if not data:
        return np.array([])
    
    series = pd.Series(data)
    smoothed_series = series.rolling(window=window_size, min_periods=1, center=False).mean()
    return smoothed_series.to_numpy()


def generate_comparison_plots(
    ppo_run_path,
    td3_run_path,
    ppo_reward_key="ppo/total_reward",
    ppo_loss_key="ppo/epoch_loss",
    td3_reward_key="total_reward",
    td3_actor_loss_key="episode_avg_actor_loss",
    td3_critic_loss_key="episode_avg_critic_loss",
    smoothing_window=DEFAULT_SMOOTHING_WINDOW,
    output_dir=".", 
    plot_title_suffix=""
):
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api_key = user_secrets.get_secret("WANDB_API_KEY") 
        wandb.login(key=wandb_api_key)
        print("Successfully logged into WandB using Kaggle secret.")
    except Exception as e:
        print(f"Could not log into WandB using Kaggle secret: {e}")
        print("Please ensure 'WANDB_API_KEY' is set as a Kaggle secret and you have internet enabled.")
        return


    
    os.makedirs(output_dir, exist_ok=True)

    try:
        ppo_rewards, ppo_losses = load_wandb_data(
            ppo_run_path,
            ppo_reward_key,
            ppo_loss_key,
            "PPO"
        )
        td3_rewards, td3_losses = load_wandb_data(
            td3_run_path,
            td3_reward_key,
            [td3_actor_loss_key, td3_critic_loss_key],
            "TD3"
        )
    except (ValueError, wandb.errors.CommError) as e:
        print(f"Error during data loading: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    ppo_rewards_smoothed = smooth_curve(ppo_rewards, smoothing_window)
    ppo_losses_smoothed = smooth_curve(ppo_losses, smoothing_window)
    td3_rewards_smoothed = smooth_curve(td3_rewards, smoothing_window)
    td3_losses_smoothed = smooth_curve(td3_losses, smoothing_window)

    title_suffix = f" {plot_title_suffix}" if plot_title_suffix else ""

    plt.figure(figsize=(12, 7))
    if len(ppo_rewards_smoothed) > 0:
        plt.plot(ppo_rewards_smoothed, label=f'PPO Reward (Smooth Window: {smoothing_window})', color='blue')
    if len(td3_rewards_smoothed) > 0:
        plt.plot(td3_rewards_smoothed, label=f'TD3 Reward (Smooth Window: {smoothing_window})', color='red')
    if smoothing_window > 1:
        if len(ppo_rewards) > 0:
            plt.plot(ppo_rewards, label='PPO Reward (Raw)', alpha=0.3, color='lightblue', linestyle='--')
        if len(td3_rewards) > 0:
            plt.plot(td3_rewards, label='TD3 Reward (Raw)', alpha=0.3, color='lightcoral', linestyle='--')
    plt.xlabel('Training Iteration (Logged Epoch/Episode)')
    plt.ylabel('Total Reward')
    plt.title(f'Comparison of Total Rewards (PPO vs. TD3){title_suffix}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    rewards_plot_path = os.path.join(output_dir, 'wandb_rewards_comparison.png')
    plt.savefig(rewards_plot_path)
    print(f"Rewards comparison plot saved to: {rewards_plot_path}")
    plt.show() 
  
    plt.figure(figsize=(12, 7))
    if len(ppo_losses_smoothed) > 0:
        plt.plot(ppo_losses_smoothed, label=f'PPO Loss (Smooth Window: {smoothing_window})', color='blue')
    if len(td3_losses_smoothed) > 0:
        plt.plot(td3_losses_smoothed, label=f'TD3 Loss (Actor + Critic, Smooth Window: {smoothing_window})', color='red')
    if smoothing_window > 1:
        if len(ppo_losses) > 0:
            plt.plot(ppo_losses, label='PPO Loss (Raw)', alpha=0.3, color='lightblue', linestyle='--')
        if len(td3_losses) > 0:
            plt.plot(td3_losses, label='TD3 Loss (Raw)', alpha=0.3, color='lightcoral', linestyle='--')
    plt.xlabel('Training Iteration (Logged Epoch/Episode)')
    plt.ylabel('Loss')
    plt.title(f'Comparison of Losses (PPO vs. TD3){title_suffix}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    losses_plot_path = os.path.join(output_dir, 'wandb_losses_comparison.png')
    plt.savefig(losses_plot_path)
    print(f"Losses comparison plot saved to: {losses_plot_path}")
    plt.show() 

# plotting function 
if __name__ == "__main__": 
    my_ppo_run_path = "giwaoluwaseyi/ppo-training/py09n52m"  #  CHANGE THIS
    my_td3_run_path = "giwaoluwaseyi/td3-per-hetnet/116s20ne"  #  CHANGE THIS
    

    output_save_dir = "/kaggle/working/" 

    generate_comparison_plots(
        ppo_run_path=my_ppo_run_path,
        td3_run_path=my_td3_run_path,
        smoothing_window=10,
        output_dir=output_save_dir,
        plot_title_suffix="HetNet Experiment"
    )
