import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from grid_env import GridEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import imageio
import argparse


class SaveGifAndLogCallback(BaseCallback):
    def __init__(self, log_dir, graph_dir, model_dir, eval_env, save_freq=10, verbose=1):
        super(SaveGifAndLogCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.graph_dir = graph_dir
        self.model_dir = model_dir
        self.eval_env = eval_env
        self.save_freq = save_freq
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_planning_times = []

        self.episode_number = 0
        self.completions = 0
        self.truncations = 0

        self.current_episode_rewards = 0
        self.current_episode_length = 0
        self.current_episode_planning_time = 0

    def _on_step(self) -> bool:
        # Update current episode metrics
        self.current_episode_rewards += self.locals["rewards"].sum()
        self.current_episode_length += 1

        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("truncated", False):
                self.truncations += 1
            if info.get("terminated", False):
                self.completions += 1

        return True

    def _on_rollout_end(self):
        # Unwrap the environment to access the original GridEnvironment
        original_env = self.eval_env
        while hasattr(original_env, "env"):
            original_env = original_env.env

        # Log episode metrics
        self.episode_number += 1
        self.logger.record("custom/steps", self.current_episode_length)
        self.logger.record("custom/reward", self.current_episode_rewards)
        self.logger.record("custom/episode", self.episode_number)
        self.logger.record("custom/completions", self.completions)
        self.logger.record("custom/truncations", self.truncations)

        # Planning time
        planning_time = original_env.rollout_duration
        self.logger.record("custom/planning_time", planning_time)

        # Save metrics for debugging
        self.episode_rewards.append(self.current_episode_rewards)
        self.episode_lengths.append(self.current_episode_length)
        self.episode_planning_times.append(planning_time)

        # Reset metrics for the next episode
        self.current_episode_rewards = 0
        self.current_episode_length = 0

        # Save model and graph every `save_freq` rollouts
        if self.episode_number % self.save_freq == 0:
            model_path = os.path.join(self.model_dir, f"model_rollout_{self.episode_number}.zip")
            self.model.save(model_path)
            if self.verbose:
                print(f"Saved model to {model_path}")

            graph_path = os.path.join(self.graph_dir, f"graph_episode_{self.episode_number}.png")
            original_env.render(save_dir=graph_path)
            if self.verbose:
                print(f"Plot saved to {graph_path}")

            # Save a GIF of the evaluation environment
            # gif_path = os.path.join(self.graph_dir, f"episode_{self.episode_number}.gif")
            # self._save_gif(original_env, gif_path)
            # if self.verbose:
            #     print(f"GIF saved to {gif_path}")

    # def _save_gif(self, env, gif_path):
    #     frames = []
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         frames.append(env.render(mode="rgb_array"))
    #         action = self.model.predict(obs, deterministic=True)[0]
    #         obs, _, terminated, truncated, _ = env.step(action)
    #         done = terminated or truncated

    #     imageio.mimsave(gif_path, frames, fps=5)


def main():
    # Initialize training and evaluation environments

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a PPO model on a GridEnvironment.")
    parser.add_argument("--grid_size", type=int, default=10, help="Size of the grid environment.")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps for training.")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency to save models and graphs.")
    parser.add_argument("--train", type=str, default=None, help="Existing model to be trained.")
    args = parser.parse_args()

    # Wall (train to 40)
    ox = [5 for _ in range(args.grid_size - 2)]
    oy = [y for y in range(args.grid_size - 2)]

    # Small maze (train to 90)
    # ox = [7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
    # oy = [2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

    # Passage (train to 90)
    # ox = [7, 7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
    # oy = [0, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

    # Space (train to 180)
    # ox = [7, 7, 7, 7, 7, 4, 6, 3, 2, 1, 0]
    # oy = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4]

    # v walls
    # ox = [5, 5, 5, 5, 5, 5, 5, 
    # oy = [0, 1, 2, 3, 4, 5, 6, 7]


    train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=ox, oy=oy)
    train_env = Monitor(train_env)

    eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=ox, oy=oy)
    eval_env = Monitor(eval_env)

    # Directories for logs, models, and graphs
    log_dir = "./logs/"
    graph_dir = "./ppo_model_graphs/"
    model_dir = "./ppo_model_checkpoints/"

    if not args.train:
    # Define PPO model
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            clip_range=0.1,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tensorboard"),
        )
    else:
        # Load model
        model = PPO.load(args.train, env=train_env)

    # Create callback
    callback = SaveGifAndLogCallback(
        log_dir=log_dir,
        graph_dir=graph_dir,
        model_dir=model_dir,
        eval_env=eval_env,
        save_freq=args.save_freq,
    )

    # Train model
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

if __name__ == "__main__":
    main()
