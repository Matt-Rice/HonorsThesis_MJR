import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from grid_env import GridEnvironment
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
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
        if self.episode_number % (self.save_freq * 2048) == 0:
            model_path = os.path.join(self.model_dir, f"model_steps_{self.episode_number}.zip")
            self.model.save(model_path)
            if self.verbose:
                print(f"Saved model to {model_path}")

            graph_path = os.path.join(self.graph_dir, f"graph_episode_{self.episode_number}.png")
            original_env.render(save_dir=graph_path)
            if self.verbose:
                print(f"Plot saved to {graph_path}")
        if hasattr(self.model, "action_noise"):
            action_noise = np.mean(self.model.action_noise())
            self.logger.record("custom/action_noise", action_noise)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a PPO model on a GridEnvironment.")
    parser.add_argument("--grid_size", type=int, default=10, help="Size of the grid environment.")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps for training.")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency to save models and graphs.")
    parser.add_argument("--train", type=str, default=None, help="Existing model to be trained.")
    args = parser.parse_args()

    # Wall (train to 20480)
    wall_ox = [5 for _ in range(args.grid_size - 2)]
    wall_oy = [y for y in range(args.grid_size - 2)]

    # Small maze (train to 90)
    maze_ox = [7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
    maze_oy = [2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

    # Passage (train to 90)
    pass_ox = [7, 7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
    pass_oy = [0, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

    # Space (train to 180)
    space_ox = [7, 7, 7, 7, 7, 4, 6, 3, 2, 1, 0]
    space_oy = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4]

    # v walls
    # ox = [5, 5, 5, 5, 5, 5, 5, 
    # oy = [0, 1, 2, 3, 4, 5, 6, 7]

    # Wall envs
    wall_train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=wall_ox, oy=wall_oy)
    wall_train_env = Monitor(wall_train_env)

    wall_eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=maze_ox, oy=maze_oy)
    wall_eval_env = Monitor(wall_eval_env)

    # Passage envs
    pass_train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=pass_ox, oy=pass_oy)
    pass_train_env = Monitor(pass_train_env)

    pass_eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=space_ox, oy=space_oy)
    pass_eval_env = Monitor(pass_eval_env)

    # Maze envs
    maze_train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=maze_ox, oy=maze_oy)
    maze_train_env = Monitor(maze_train_env)

    maze_eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=space_ox, oy=space_oy)
    maze_eval_env = Monitor(maze_eval_env)

    # Space envs
    space_train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=space_ox, oy=space_oy)
    space_train_env = Monitor(space_train_env)

    space_eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=space_ox, oy=space_oy)
    space_eval_env = Monitor(space_eval_env)

    # Directories for logs, models, and graphs
    log_dir = "./logs/"
    graph_dir = "./model_graphs/"
    model_dir = "./model_checkpoints/"

    # Create action noise for exploration (important for DDPG)
    n_actions = wall_train_env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    train_envs = [wall_train_env, maze_train_env, pass_train_env, space_train_env]
    
    # Create callback
    wall_callback = SaveGifAndLogCallback(
        log_dir=log_dir,
        graph_dir=graph_dir + 'wall/',
        model_dir=model_dir + 'wall/',
        eval_env=wall_eval_env,
        save_freq=args.save_freq,
    )

    pass_callback = SaveGifAndLogCallback(
        log_dir=log_dir,
        graph_dir=graph_dir  + 'passage/',
        model_dir=model_dir  + 'passage/',
        eval_env=pass_eval_env,
        save_freq=args.save_freq,
    )

    maze_callback = SaveGifAndLogCallback(
        log_dir=log_dir,
        graph_dir=graph_dir + 'maze/',
        model_dir=model_dir + 'maze/',
        eval_env=maze_eval_env,
        save_freq=args.save_freq,
    )

    space_callback = SaveGifAndLogCallback(
        log_dir=log_dir,
        graph_dir=graph_dir + 'space/',
        model_dir=model_dir + 'space/',
        eval_env=space_eval_env,
        save_freq=args.save_freq,
    )

    callbacks = [wall_callback, maze_callback, pass_callback, space_callback]
    if not args.train:
        for env, callback in zip(train_envs, callbacks):
            
            model = DDPG(
                "MlpPolicy",
                env,
                action_noise=action_noise,
                learning_rate=1e-3,
                buffer_size=100000,
                batch_size=256,
                verbose=2,
                tensorboard_log=os.path.join(log_dir, "tensorboard"),
            )
            model.learn(total_timesteps=args.total_timesteps, callback=callback)
    else:
        model = DDPG.load(args.train, env=wall_train_env)
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    

    # Train model
    model.learn(total_timesteps=args.total_timesteps, callback=wall_callback)

if __name__ == "__main__":
    main()
