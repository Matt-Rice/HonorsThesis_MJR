import optuna
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from grid_env import GridEnvironment

# Define environment setup for tuning
# def create_env():
#     return Monitor(GridEnvironment(sx=0, sy=0, gx=9, gy=9, grid_size=10, training_mode=True))
maze_ox = [7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
maze_oy = [2, 3, 4, 4, 4, 4, 4, 4, 4, 4]
def create_env():
    return Monitor(GridEnvironment(sx=0, sy=0, gx=9, gy=9, grid_size=10, ox=maze_ox, oy=maze_oy))

# Define objective function for Optuna
def objective(trial):
    """Optimize hyperparameters for PPO using Optuna."""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 0.01)

    # Create environment
    env = create_env()

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
    """Evaluate PPO model by running test episodes and returning average reward."""
    total_rewards = []
    for _ in range(episodes):
        obs = env.reset()
        done, truncated, ep_reward = False, False, 0
        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)

    return np.mean(total_rewards)  # Return average reward across episodes

# Run Optuna Optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # Maximize reward
    study.optimize(objective, n_trials=20)  # Run 20 trials

    # Print best hyperparameters
    print("Best hyperparameters:", study.best_params)
