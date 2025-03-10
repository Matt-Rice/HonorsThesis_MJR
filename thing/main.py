# import os
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.monitor import Monitor
# from grid_env import GridEnvironment
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# import perun
# import time
# import argparse

# class SaveGifAndLogCallback(BaseCallback):
#     def __init__(self, log_dir, graph_dir, model_dir, eval_env, save_freq=10, verbose=1):
#         super(SaveGifAndLogCallback, self).__init__(verbose)
#         self.log_dir = log_dir
#         self.graph_dir = graph_dir
#         self.model_dir = model_dir
#         self.eval_env = eval_env
#         self.save_freq = save_freq
#         os.makedirs(log_dir, exist_ok=True)
#         os.makedirs(self.graph_dir, exist_ok=True)
#         os.makedirs(self.model_dir, exist_ok=True)

#         self.episode_rewards = []
#         self.episode_lengths = []
#         self.episode_losses = []
#         self.episode_planning_times = []

#         self.episode_number = 0
#         self.completions = 0
#         self.truncations = 0

#         self.current_episode_rewards = 0
#         self.current_episode_length = 0
#         self.current_episode_planning_time = 0

#     def _on_step(self) -> bool:
#         # Update current episode metrics
#         self.current_episode_rewards += self.locals["rewards"].sum()
#         self.current_episode_length += 1

#         infos = self.locals.get("infos", [])
#         for info in infos:
#             if info.get("truncated", False):
#                 self.truncations += 1
#             if info.get("terminated", False):
#                 self.completions += 1

#         return True

#     def _on_rollout_end(self):
#         # Unwrap the environment to access the original GridEnvironment
#         original_env = self.eval_env
#         while hasattr(original_env, "env"):
#             original_env = original_env.env

#         # Log episode metrics
#         self.episode_number += 1
#         self.logger.record("custom/steps", self.current_episode_length)
#         self.logger.record("custom/reward", self.current_episode_rewards)
#         self.logger.record("custom/episode", self.episode_number)
#         self.logger.record("custom/completions", self.completions)
#         self.logger.record("custom/truncations", self.truncations)

#         # Planning time
#         planning_time = original_env.rollout_duration
#         self.logger.record("custom/planning_time", planning_time)

#         # Save metrics for debugging
#         self.episode_rewards.append(self.current_episode_rewards)
#         self.episode_lengths.append(self.current_episode_length)
#         self.episode_planning_times.append(planning_time)

#         # Reset metrics for the next episode
#         self.current_episode_rewards = 0
#         self.current_episode_length = 0

#         # Save model and graph every `save_freq` rollouts
#         if self.episode_number % self.save_freq == 0:
#             model_path = os.path.join(self.model_dir, f"model_rollout_{self.episode_number}.zip")
#             self.model.save(model_path)
#             if self.verbose:
#                 print(f"Saved model to {model_path}")

#             graph_path = os.path.join(self.graph_dir, f"graph_episode_{self.episode_number}.png")
#             original_env.render(save_dir=graph_path)
#             if self.verbose:
#                 print(f"Plot saved to {graph_path}")

#             # Save a GIF of the evaluation environment
#             # gif_path = os.path.join(self.graph_dir, f"episode_{self.episode_number}.gif")
#             # self._save_gif(original_env, gif_path)
#             # if self.verbose:
#             #     print(f"GIF saved to {gif_path}")

#     # def _save_gif(self, env, gif_path):
#     #     frames = []
#     #     obs = env.reset()
#     #     done = False
#     #     while not done:
#     #         frames.append(env.render(mode="rgb_array"))
#     #         action = self.model.predict(obs, deterministic=True)[0]
#     #         obs, _, terminated, truncated, _ = env.step(action)
#     #         done = terminated or truncated

#     #     imageio.mimsave(gif_path, frames, fps=5)

    
# @perun.perun(data_out= "energy_results", format="json")
# def main():
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Train and evaluate a PPO model on a GridEnvironment.")
#     parser.add_argument("--grid_size", type=int, default=10, help="Size of the grid environment.")
#     parser.add_argument("--environment", type=str, default=None)
#     parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps for training.")
#     parser.add_argument("--save_freq", type=int, default=10, help="Frequency to save models and graphs.")
#     parser.add_argument("--train", type=str, default=None, help="Existing model to be trained.")
#     args = parser.parse_args()

#     # Wall (train to 20480)
#     wall_ox = [5 for _ in range(args.grid_size - 2)]
#     wall_oy = [y for y in range(args.grid_size - 2)]

#     # Small maze (train to 90)
#     maze_ox = [7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
#     maze_oy = [2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

#     # Passage (train to 90)
#     pass_ox = [7, 7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
#     pass_oy = [0, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

#     # Space (train to 180)
#     space_ox = [7, 7, 7, 7, 7, 4, 6, 3, 2, 1, 0]
#     space_oy = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4]

#     # v walls
#     # ox = [5, 5, 5, 5, 5, 5, 5, 
#     # oy = [0, 1, 2, 3, 4, 5, 6, 7]

#     # Wall envs
#     wall_train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=wall_ox, oy=wall_oy)
#     wall_train_env = Monitor(wall_train_env)

#     wall_eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=wall_ox, oy=wall_oy)
#     wall_eval_env = Monitor(wall_eval_env)

#     # Passage envs
#     pass_train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=pass_ox, oy=pass_oy)
#     pass_train_env = Monitor(pass_train_env)

#     pass_eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=pass_ox, oy=pass_oy)
#     pass_eval_env = Monitor(pass_eval_env)

#     # Maze envs
#     maze_train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=maze_ox, oy=maze_oy)
#     maze_train_env = Monitor(maze_train_env)

#     maze_eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=maze_ox, oy=maze_oy)
#     maze_eval_env = Monitor(maze_eval_env)

#     # Space envs
#     space_train_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=space_ox, oy=space_oy)
#     space_train_env = Monitor(space_train_env)

#     space_eval_env = GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=space_ox, oy=space_oy)
#     space_eval_env = Monitor(space_eval_env)

#     # Directories for logs, models, and graphs
#     log_dir = "./logs/"
#     graph_dir = "./model_graphs/"
#     model_dir = "./model_checkpoints/"

#     train_envs = [wall_train_env, maze_train_env, pass_train_env, space_train_env]
    
#     # Create callback
#     wall_callback = SaveGifAndLogCallback(
#         log_dir=log_dir,
#         graph_dir=graph_dir + 'wall/',
#         model_dir=model_dir + 'wall/',
#         eval_env=wall_eval_env,
#         save_freq=args.save_freq,
#     )

#     pass_callback = SaveGifAndLogCallback(
#         log_dir=log_dir,
#         graph_dir=graph_dir  + 'passage/',
#         model_dir=model_dir  + 'passage/',
#         eval_env=pass_eval_env,
#         save_freq=args.save_freq,
#     )

#     maze_callback = SaveGifAndLogCallback(
#         log_dir=log_dir,
#         graph_dir=graph_dir + 'maze/',
#         model_dir=model_dir + 'maze/',
#         eval_env=maze_eval_env,
#         save_freq=args.save_freq,
#     )

#     space_callback = SaveGifAndLogCallback(
#         log_dir=log_dir,
#         graph_dir=graph_dir + 'space/',
#         model_dir=model_dir + 'space/',
#         eval_env=space_eval_env,
#         save_freq=args.save_freq,
#     )

#     callbacks = [wall_callback, maze_callback, pass_callback, space_callback]
    
#     match args.environment:
#         case None:
#             for env, callback in zip(train_envs, callbacks):
#                 if not args.train:
#                     # Define PPO model
#                     model = PPO(
#                         "MlpPolicy",
#                         env,
#                         learning_rate=3e-4,
#                         clip_range=0.1,
#                         verbose=1,
#                         tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                         device="cuda"
#                     )
#                 else:
#                     # Load model
#                     model = PPO.load(args.train, env=env)

#                 # train model
#                 start_time = time.time()
#                 model.learn(total_timesteps=args.total_timesteps, callback=callback)
#                 end_time = time.time()

#                 # Compute efficiency
#                 total_time = end_time - start_time
#                 # energy_used = perun.get_total_energy()  # Total energy in Joules
#                 # print(f"Total Energy Used: {energy_used:.2f} J")
#                 # print(f"Energy per Episode: {energy_used / args.total_timesteps:.2f} J/episode")
#                 print(f"Execution Time: {total_time:.2f} sec")
    

#         case "wall":
#             env = wall_train_env
#             callback = wall_callback
#             if not args.train:
#                 # Define PPO model
#                 model = PPO(
#                     "MlpPolicy",
#                     env,
#                     learning_rate=3e-4,
#                     clip_range=0.1,
#                     verbose=1,
#                     tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                 )
#             else:
#                 # Load model
#                 model = PPO.load(args.train, env=env)

#             # train model
#             start_time = time.time()
#             model.learn(total_timesteps=args.total_timesteps, callback=callback)
#             end_time = time.time()

#             # Compute efficiency
#             total_time = end_time - start_time
#             # energy_used = perun.get_total_energy()  # Total energy in Joules
#             # print(f"Total Energy Used: {energy_used:.2f} J")
#             # print(f"Energy per Episode: {energy_used / args.total_timesteps:.2f} J/episode")
#             print(f"Execution Time: {total_time:.2f} sec")
    
        
#         case "maze":
#             env = maze_train_env
#             callback = maze_callback
#             if not args.train:
#                 # Define PPO model
#                 model = PPO(
#                     "MlpPolicy",
#                     env,
#                     learning_rate=3e-4,
#                     clip_range=0.1,
#                     verbose=1,
#                     tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                 )
#             else:
#                 # Load model
#                 model = PPO.load(args.train, env=env)

#             # train model
#             start_time = time.time()
#             model.learn(total_timesteps=args.total_timesteps, callback=callback)
#             end_time = time.time()

#             # Compute efficiency
#             total_time = end_time - start_time
 
#             print(f"Execution Time: {total_time:.2f} sec")
    
#         case "passage":
#             env = pass_train_env
#             callback = pass_callback
#             if not args.train:
#                 # Define PPO model
#                 model = PPO(
#                     "MlpPolicy",
#                     env,
#                     learning_rate=3e-4,
#                     clip_range=0.1,
#                     verbose=1,
#                     tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                 )
#             else:
#                 # Load model
#                 model = PPO.load(args.train, env=env)

#             # train model
#             start_time = time.time()
#             model.learn(total_timesteps=args.total_timesteps, callback=callback)
#             end_time = time.time()

#             # Compute efficiency
#             total_time = end_time - start_time
#             # energy_used = perun.get_total_energy()  # Total energy in Joules
#             # print(f"Total Energy Used: {energy_used:.2f} J")
#             # print(f"Energy per Episode: {energy_used / args.total_timesteps:.2f} J/episode")
#             print(f"Execution Time: {total_time:.2f} sec")
    
#         case "space":
#             env = space_train_env
#             callback = space_callback
#             if not args.train:
#                 # Define PPO model
#                 model = PPO(
#                     "MlpPolicy",
#                     env,
#                     learning_rate=3e-4,
#                     clip_range=0.1,
#                     verbose=1,
#                     tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                 )
#             else:
#                 # Load model
#                 model = PPO.load(args.train, env=env)

#             # train model
#             start_time = time.time()
#             model.learn(total_timesteps=args.total_timesteps, callback=callback)
#             end_time = time.time()

#             # Compute efficiency
#             total_time = end_time - start_time
#             # energy_used = perun.get_total_energy()  # Total energy in Joules
#             # print(f"Total Energy Used: {energy_used:.2f} J")
#             # print(f"Energy per Episode: {energy_used / args.total_timesteps:.2f} J/episode")
#             print(f"Execution Time: {total_time:.2f} sec")
    

# if __name__ == "__main__":
#     main()

import os
import perun 
import time
import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from grid_env import GridEnvironment
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

#@perun.perun(data_out="energy_results", format="json")
def train_curriculum(model, train_envs, eval_envs, total_timesteps=100000, save_freq=10000, reward_threshold=200):
    for i, (train_env, eval_env) in enumerate(zip(train_envs, eval_envs)):
        
        if i == 1:
            print(f"Loading from environment {i-1}")
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            model = PPO.load("./models/2/env_0/model_rollout_90.zip", train_env)
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/2/env_{i}/", f"./models/2/env_{i}/", eval_env, save_freq)

            model.learn(total_timesteps=total_timesteps, callback=callback)
        elif i == 2:
            print(f"Loading from environment {i-1}")
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            model = PPO.load("./models/2/env_1/model_rollout_90.zip", train_env)
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/2/env_{i}/", f"./models/2/env_{i}/", eval_env, save_freq)

            model.learn(total_timesteps=total_timesteps, callback=callback)
        elif i == 3:
            print(f"Loading from environment {i-1}")
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            model = PPO.load("./models/2/env_2/model_rollout_90.zip", train_env)
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/2/env_{i}/", f"./models/2/env_{i}/", eval_env, save_freq)

            model.learn(total_timesteps=total_timesteps, callback=callback)
        elif i == 4:
            print(f"Loading from environment {i-1}")
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            model = PPO.load("./models/2/env_3/model_rollout_90.zip", train_env)
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/2/env_{i}/", f"./models/2/env_{i}/", eval_env, save_freq)

            model.learn(total_timesteps=total_timesteps, callback=callback)
        else:
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/2/env_{i}/", f"./models/2/env_{i}/", eval_env, save_freq)

            model.learn(total_timesteps=total_timesteps, callback=callback)

        #model.learn(total_timesteps=total_timesteps, callback=callback)
        # mean_reward = evaluate_model(model, eval_env)
        # print(f"\nFinished Environment {i + 1} with mean reward: {mean_reward}")
        # if mean_reward < reward_threshold:
        #     print("\nNot ready to progress. Retraining...")
        #     model.learn(total_timesteps=total_timesteps//2, callback=callback)
    return model


# def evaluate_model(model, eval_env, episodes=10):
#     total_rewards = []
#     for _ in range(episodes):
#         obs, done, truncated, ep_reward = eval_env.reset(), False, False, 0
#         while not done or truncated:
#             action, _ = model.predict(obs)
#             obs, reward, done, truncated, _ = eval_env.step(action)
#             ep_reward += reward

#             if done or truncated:
#                 obs = eval_env.reset() 
#         total_rewards.append(ep_reward)
#     return np.mean(total_rewards)


# def optimize_hyperparams(trial):
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
#     buffer_size = trial.suggest_int("buffer_size", 50000, 1000000)
#     batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
#     model = TD3("MlpPolicy", train_envs[0], learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, verbose=1, tensorboard_log="./logs/tensorboard")
#     model = train_curriculum(model, train_envs, eval_envs, total_timesteps=50)
#     mean_reward = evaluate_model(model, eval_envs[-1])
#     return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    # Wall (train to 20480)
    # wall_ox = [5 for _ in range(args.grid_size - 2)]
    # wall_oy = [y for y in range(args.grid_size - 2)]

    # # Small maze (train to 90)
    # maze_ox = [7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
    # maze_oy = [2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

    # # Passage (train to 90)
    # pass_ox = [7, 7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
    # pass_oy = [0, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

    # # Space (train to 180)
    # space_ox = [7, 7, 7, 7, 7, 4, 6, 3, 2, 1, 0]
    # space_oy = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4]

    # None 
    # none_ox = []
    # none_oy = []

    # # a few
    # few_ox = [2, 3, 4]
    # few_oy = [3, 3, 3]

    # # More
    # more_ox = [i+2 for i in range(7)]
    # more_oy = [4 for _ in range (7)]

    # # wall
    # wall_ox = [5 for _ in range(4)]
    # wall_oy = [3, 4, 5, 6]

    # v walls
    # ox = [5, 5, 5, 5, 5, 5, 5, 
    # oy = [0, 1, 2, 3, 4, 5, 6, 7]

    envs = [
        GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size),
        GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, num_obs=0.1),
        GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, num_obs=0.15),
        GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, num_obs=0.2)
    ]

    train_envs = [Monitor(env) for env in envs]
    eval_envs = [Monitor(env) for env in envs]

    # if args.optimize:
    #     study = optuna.create_study(direction="maximize")
    #     study.optimize(optimize_hyperparams, n_trials=10)
    #     print(f"Best hyperparameters: {study.best_params}")
    # else:

    model = PPO(
        "MlpPolicy", 
        train_envs[0], 
        learning_rate=0.0009465151432244565, 
        gamma= 0.9256136864027436,
        clip_range=0.26122827380929725,
        n_steps=1024,
        ent_coef= 0.00017341259491965202,
        verbose=1, 
        tensorboard_log="./logs/tensorboard"
        )

    #callback = SaveGifAndLogCallback("./logs/", f"./graphs/more/", f"./models/more/", eval_envs[3], args.save_freq)    
    model = PPO.load("./models/2/env_3/model_rollout_90.zip", env=train_envs[0])
    #model.learn(total_timesteps=args.total_timesteps, callback=callback)
    train_curriculum(model, train_envs, eval_envs, args.total_timesteps, args.save_freq)
