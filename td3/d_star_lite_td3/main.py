# import os
# import perun 
# import time
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.noise import NormalActionNoise
# import numpy as np
# from grid_env import GridEnvironment
# from stable_baselines3 import TD3
# from stable_baselines3.common.monitor import Monitor
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
#         if self.episode_number % (self.save_freq) == 0:
#             model_path = os.path.join(self.model_dir, f"model_steps_{self.episode_number}.zip")
#             self.model.save(model_path)
#             if self.verbose:
#                 print(f"Saved model to {model_path}")

#             graph_path = os.path.join(self.graph_dir, f"graph_episode_{self.episode_number}.png")
#             original_env.render(save_dir=graph_path)
#             if self.verbose:
#                 print(f"Plot saved to {graph_path}")
#         if hasattr(self.model, "action_noise"):
#             action_noise = np.mean(self.model.action_noise())
#             self.logger.record("custom/action_noise", action_noise)
    
# @perun.perun(data_out= "energy_results", format="json")
# def main():
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Train and evaluate a PPO model on a GridEnvironment.")
#     parser.add_argument("--grid_size", type=int, default=10, help="Size of the grid environment.")
#     parser.add_argument("--environment", type=str, default=None)
#     parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps for training.")
#     parser.add_argument("--save_freq", type=int, default=10000, help="Frequency to save models and graphs.")
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

#     # Create action noise for exploration (important for TD3)
#     n_actions = wall_train_env.action_space.shape[0]
#     action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

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
#                     model = TD3(
#                         "MlpPolicy",
#                         env,
#                         action_noise=action_noise,
#                         learning_rate=1e-3,
#                         buffer_size=1000000,
#                         batch_size=256,
#                         verbose=2,
#                         tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                     )

#                 else:
#                     model = TD3.load(args.train, env=wall_train_env)

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
#                 model = TD3(
#                     "MlpPolicy",
#                     env,
#                     action_noise=action_noise,
#                     learning_rate=1e-3,
#                     buffer_size=1000000,
#                     batch_size=256,
#                     verbose=2,
#                     tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                 )

#             else:
#                 model = TD3.load(args.train, env=wall_train_env)

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
#                 model = TD3(
#                     "MlpPolicy",
#                     env,
#                     action_noise=action_noise,
#                     learning_rate=1e-3,
#                     buffer_size=1000000,
#                     batch_size=256,
#                     verbose=2,
#                     tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                 )

#             else:
#                 model = TD3.load(args.train, env=wall_train_env)

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
#                 model = TD3(
#                     "MlpPolicy",
#                     env,
#                     action_noise=action_noise,
#                     learning_rate=1e-3,
#                     buffer_size=1000000,
#                     batch_size=256,
#                     verbose=2,
#                     tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                 )

#             else:
#                 model = TD3.load(args.train, env=wall_train_env)

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
#                 model = TD3(
#                     "MlpPolicy",
#                     env,
#                     action_noise=action_noise,
#                     learning_rate=1e-3,
#                     buffer_size=1000000,
#                     batch_size=256,
#                     verbose=2,
#                     tensorboard_log=os.path.join(log_dir, "tensorboard"),
#                 )

#             else:
#                 model = TD3.load(args.train, env=wall_train_env)

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
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from grid_env import GridEnvironment
import argparse

class SaveGifAndLogCallback(BaseCallback):
    def __init__(self, log_dir, graph_dir, model_dir, eval_env, save_freq=10, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.graph_dir = graph_dir
        self.model_dir = model_dir
        self.eval_env = eval_env
        self.save_freq = save_freq
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.episode_number = 0
        self.episode_reward = 0

    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        self.episode_number += 1
        if self.episode_number % self.save_freq == 0:
            model_path = os.path.join(self.model_dir, f"model_steps_{self.episode_number}.zip")
            self.model.save(model_path)
            print(f"Saved model to {model_path}")

@perun.perun(data_out="energy_results", format="json")
def train_curriculum(model, train_envs, eval_envs, total_timesteps=100000, save_freq=10000, reward_threshold=200):
    for i, (train_env, eval_env) in enumerate(zip(train_envs, eval_envs)):
        
        if i == 1:
            print(f"Loading from environment {i-1}")
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            model = TD3(
            "MlpPolicy", 
            train_envs[i], 
            learning_rate=0.0009021781483419764,  # Learning rate
            gamma=0.9413403331670541,  # Discount factor
            batch_size=128,  # Batch size
            tau= 0.03664764890096477,
            policy_delay=2,  # Policy update delay
            train_freq=1,  # Train frequency (every step)
            learning_starts=10000,  # Steps before training begins
            action_noise=action_noise,  # Exploration noise
            verbose=1, 
            tensorboard_log="./logs/tensorboard"
            )
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/env_{i}/", f"./models/env_{i}/", eval_env, save_freq)
        elif i == 2:
            print(f"Loading from environment {i-1}")
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            model = TD3(
            "MlpPolicy", 
            train_envs[i], 
            learning_rate=0.0009021781483419764,  # Learning rate
            gamma=0.9413403331670541,  # Discount factor
            batch_size=128,  # Batch size
            tau= 0.03664764890096477,
            policy_delay=2,  # Policy update delay
            train_freq=1,  # Train frequency (every step)
            learning_starts=10000,  # Steps before training begins
            action_noise=action_noise,  # Exploration noise
            verbose=1, 
            tensorboard_log="./logs/tensorboard"
            )
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/env_{i}/", f"./models/env_{i}/", eval_env, save_freq)
        elif i == 3:
            print(f"Loading from environment {i-1}")
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            model = TD3(
            "MlpPolicy", 
            train_envs[i], 
            learning_rate=0.0009021781483419764,  # Learning rate
            gamma=0.9413403331670541,  # Discount factor
            batch_size=128,  # Batch size
            tau= 0.03664764890096477,
            policy_delay=2,  # Policy update delay
            train_freq=1,  # Train frequency (every step)
            learning_starts=10000,  # Steps before training begins
            action_noise=action_noise,  # Exploration noise
            verbose=1, 
            tensorboard_log="./logs/tensorboard"
            )
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/env_{i}/", f"./models/env_{i}/", eval_env, save_freq)
        else:
            print(f"\nTraining on environment {i + 1}/{len(train_envs)}")
            model = TD3(
            "MlpPolicy", 
            train_envs[i], 
            learning_rate=0.0009021781483419764,  # Learning rate
            gamma=0.9413403331670541,  # Discount factor
            batch_size=128,  # Batch size
            tau= 0.03664764890096477,
            policy_delay=2,  # Policy update delay
            train_freq=1,  # Train frequency (every step)
            learning_starts=10000,  # Steps before training begins
            action_noise=action_noise,  # Exploration noise
            verbose=1, 
            tensorboard_log="./logs/tensorboard"
            )
            callback = SaveGifAndLogCallback("./logs/", f"./graphs/env_{i}/", f"./models/env_{i}/", eval_env, save_freq)

        model.learn(total_timesteps=total_timesteps, callback=callback)
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

    envs = [
        GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=wall_ox, oy=wall_oy),
        GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=maze_ox, oy=maze_oy),
        GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=pass_ox, oy=pass_oy),
        GridEnvironment(sx=0, sy=0, gx=args.grid_size-1, gy=args.grid_size-1, grid_size=args.grid_size, ox=space_ox, oy=space_oy)
    ]

    train_envs = [Monitor(env) for env in envs]
    eval_envs = [Monitor(env) for env in envs]

    # Define action noise
    action_noise = NormalActionNoise(mean=np.zeros(envs[0].action_space.shape), 
                                 sigma=0.2610364729463586 * np.ones(envs[0].action_space.shape))
    
    model = TD3(
    "MlpPolicy", 
    train_envs[0], 
    learning_rate=0.0009021781483419764,  # Learning rate
    gamma=0.9413403331670541,  # Discount factor
    batch_size=128,  # Batch size
    tau= 0.03664764890096477,
    policy_delay=2,  # Policy update delay
    train_freq=1,  # Train frequency (every step)
    learning_starts=10000,  # Steps before training begins
    action_noise=action_noise,  # Exploration noise
    verbose=1, 
    tensorboard_log="./logs/tensorboard"
    )
    # model = TD3(
    # "MlpPolicy", 
    # train_envs[0], 
    # learning_rate=0.0003,  # Learning rate
    # gamma=0.99,  # Discount factor
    # batch_size=128,  # Batch size
    # policy_delay=2,  # Policy update delay
    # train_freq=100,  # Train frequency (every 100 steps)
    # learning_starts=5000,  # Steps before training begins
    # action_noise=action_noise,  # Exploration noise
    # verbose=1, 
    # tensorboard_log="./logs/tensorboard"
    # )
    train_curriculum(model=model, train_envs=train_envs, eval_envs=eval_envs, total_timesteps=args.total_timesteps)