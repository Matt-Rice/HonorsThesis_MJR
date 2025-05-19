import optuna
import os
import numpy as np
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from grid_env import GridEnvironment
from base_grid_env import BaseGridEnvironment

# Define environment setup for tuning
def create_env():
    return Monitor(GridEnvironment(sx=0, sy=0, gx=9, gy=9, grid_size=10, num_obs=0.2))

# Define objective function for Optuna
def objective(trial):
    """Optimize hyperparameters for TD3/PPO using Optuna."""
    
    # # Sample hyperparameters
    # learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    # gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    # tau = trial.suggest_float("tau", 0.005, 0.05)  # Target network update rate
    # policy_delay = trial.suggest_int("policy_delay", 1, 3)  # Policy update frequency
    # train_freq = trial.suggest_categorical("train_freq", [1, 10, 100])  # Steps before training
    # learning_starts = trial.suggest_categorical("learning_starts", [1000, 5000, 10000])  # When training starts

    # # Action noise (important for TD3 exploration)
    # action_noise_std = trial.suggest_float("action_noise_std", 0.1, 0.5)  # Standard deviation of action noise
    # action_noise = NormalActionNoise(mean=np.zeros(1), sigma=action_noise_std * np.ones(1))
    
    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 0.01)

    # Create environment
    env = create_env()

    # Define TD3 model with sampled hyperparameters
    # model = TD3(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=learning_rate,
    #     gamma=gamma,
    #     batch_size=batch_size,
    #     tau=tau,
    #     policy_delay=policy_delay,
    #     train_freq=train_freq,
    #     learning_starts=learning_starts,
    #     action_noise=action_noise,
    #     verbose=0
    # )

     # Define PPO model with sampled hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        clip_range=clip_range,
        n_steps=n_steps,
        ent_coef=ent_coef,
        verbose=0
    )

    # Train model for a fixed number of timesteps
    model.learn(total_timesteps=50000)

    # Evaluate performance
    avg_reward = evaluate_model(model, env)

    return avg_reward  # Optuna tries to maximize this

# Evaluation function
def evaluate_model(model, env, episodes=5):
    """Evaluate TD3 or PPO model by running test episodes and returning average reward."""
    total_rewards = []
    for _ in range(episodes):
        obs = env.reset()
        done, truncated, ep_reward = False, False, 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic policy
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)

    return np.mean(total_rewards)  # Return average reward across episodes

# Run Optuna Optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # Maximize reward
    study.optimize(objective, n_trials=20)  # Run 20

    # Print best hyperparameters
    print("Best hyperparameters:", study.best_params)