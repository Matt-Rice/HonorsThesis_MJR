import os
import matplotlib.pyplot as plt
import numpy as np
import time
from grid_env import GridEnvironment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import perun

# Wall (train to 20480)
wall_ox = [5 for _ in range(10 - 2)]
wall_oy = [y for y in range(10 - 2)]

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
wall_train_env = GridEnvironment(sx=0, sy=0, gx=10-1, gy=10-1, grid_size=10, ox=wall_ox, oy=wall_oy)

# Passage envs
pass_train_env = GridEnvironment(sx=0, sy=0, gx=10-1, gy=10-1, grid_size=10, ox=pass_ox, oy=pass_oy)

# Maze envs
maze_train_env = GridEnvironment(sx=0, sy=0, gx=10-1, gy=10-1, grid_size=10, ox=maze_ox, oy=maze_oy)

# Space envs
space_train_env = GridEnvironment(sx=0, sy=0, gx=10-1, gy=10-1, grid_size=10, ox=space_ox, oy=space_oy)

# Set up sim params
train_envs = [wall_train_env, maze_train_env, pass_train_env, space_train_env]

models = ["./models/env_0/model_rollout_190", "./models/env_1/model_rollout_190", "./models/env_2/model_rollout_190", "./models/env_3/model_rollout_190"]

names = ["wall", "maze", "passage", "space"]

output_dirs = ["./test_runs/wall/", "./test_runs/maze/", "./test_runs/passage/", "./test_runs/space/"]

for dir in output_dirs:
    os.makedirs(dir, exist_ok=True)

def test(env, model, episodes, output_dir, name):
    test_env = Monitor(env)
    model = PPO.load(model)
    episodes = episodes
    rewards = []
    steps_list = []
    trajectories = []
    planning_times = []  # Store total planning times per episode

    for episode in range(episodes):
        obs = test_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        trajectory_length = 0
        times = []  # Store planning time for each step in this episode

        while not done and not truncated:
            start_time = time.time()  # Start timer before action selection
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            end_time = time.time()  # End timer after step
            planning_time = end_time - start_time  # Calculate step planning time
            times.append(planning_time)  # Append step time to the list

            trajectory_length = info.get("trajectory_length", trajectory_length)  # Update trajectory length
            total_reward += reward
            steps += 1

        # Calculate total planning time for this episode
        total_planning_time = sum(times)
        planning_times.append(total_planning_time)  # Append total time for this episode
        trajectories.append(trajectory_length)
        rewards.append(total_reward)
        steps_list.append(steps)

        with open(f"{output_dir}{name}.txt", "a") as file:
            file.write(f"Episode {episode+1}: Reward: {total_reward}, Steps: {steps}, Trajectory Length: {trajectory_length}, Total Time: {total_planning_time:.6f}\n")

        env.render(save_dir=f"{output_dir}{episode}")

    # Final averages across all episodes
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}, "
        f"Average Steps: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}, "
        f"Average Trajectory Length: {np.mean(trajectories):.2f} ± {np.std(trajectories):.2f}, "
        f"Average Planning Time: {np.mean(planning_times):.6f} ± {np.std(planning_times):.6f}")

    
    with open(f"{output_dir}{name}.txt", "a") as file:
        file.write(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}, "
        f"Average Steps: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}, "
        f"Average Trajectory Length: {np.mean(trajectories):.2f} ± {np.std(trajectories):.2f}, "
        f"Average Planning Time: {np.mean(planning_times):.6f} ± {np.std(planning_times):.6f}")

@perun.perun(data_out="energy_results", format="json")
def main():
    for model, env, output_dir, name in zip(models, train_envs, output_dirs, names):
        test(env, model, 100, output_dir, name)

if __name__ == "__main__":
    main()