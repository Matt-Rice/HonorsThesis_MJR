import os
import matplotlib.pyplot as plt
import numpy as np
import time
from grid_env import GridEnvironment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import perun

# Grid size
grid_size = 30

# Wall (train to 20480)
wall_ox = [5 for _ in range(10 - 2)]
wall_oy = [y for y in range(10 - 2)]

wall_ox.append(1)
wall_oy.append(4)

# Small maze (train to 90)
maze_ox = [7, 7, 7, 4, 5, 6, 3, 2, 1, 0, 9]
maze_oy = [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 6]

# Passage (train to 90)
pass_ox = [7, 7, 7, 7, 4, 5, 6, 3, 2, 1, 0, 9]
pass_oy = [0, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 6]

# Space (train to 180)
space_ox = [7, 7, 7, 7, 7, 4, 6, 3, 2, 1, 0, 7]
space_oy = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 7]

# v walls
# ox = [5, 5, 5, 5, 5, 5, 5, 
# oy = [0, 1, 2, 3, 4, 5, 6, 7]

# # Wall envs
# wall_train_env = GridEnvironment(sx=0, sy=0, gx=10-1, gy=10-1, grid_size=10, ox=wall_ox, oy=wall_oy)

# # Passage envs
# pass_train_env = GridEnvironment(sx=0, sy=0, gx=10-1, gy=10-1, grid_size=10, ox=pass_ox, oy=pass_oy)

# # Maze envs
# maze_train_env = GridEnvironment(sx=0, sy=0, gx=10-1, gy=10-1, grid_size=10, ox=maze_ox, oy=maze_oy)

# # Space envs
# space_train_env = GridEnvironment(sx=0, sy=0, gx=10-1, gy=10-1, grid_size=10, ox=space_ox, oy=space_oy)


# Set up sim params
train_envs = [
        GridEnvironment(sx=0, sy=0, gx=grid_size-1, gy=grid_size-1, grid_size=grid_size),
        GridEnvironment(sx=0, sy=0, gx=grid_size-1, gy=grid_size-1, grid_size=grid_size, num_obs=0.1),
        GridEnvironment(sx=0, sy=0, gx=grid_size-1, gy=grid_size-1, grid_size=grid_size, num_obs=0.15),
        GridEnvironment(sx=0, sy=0, gx=grid_size-1, gy=grid_size-1, grid_size=grid_size, num_obs=0.2)
        ]

model = "./models/2/env_3/model_rollout_90.zip"

names = ["none", ".1", ".15", ".2"]

output_dirs = ["./test_runs/2/none/", "./test_runs/2/.1/", "./test_runs/2/.15/", "./test_runs/2/.2/"]

for dir in output_dirs:
    os.makedirs(dir, exist_ok=True)

@perun.perun(data_out= "energy_results", format="json")
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
    print(f"Average Reward: {np.mean(rewards):.2f}, Average Steps: {np.mean(steps_list):.2f}, "
        f"Average Trajectory Length: {np.mean(trajectories):.2f}, "
        f"Average Planning Time: {np.mean(planning_times):.6f}")
    
    with open(f"{output_dir}{name}.txt", "a") as file:
        file.write(f"Average Reward: {np.mean(rewards):.2f}, Average Steps: {np.mean(steps_list):.2f}, "
        f"Average Trajectory Length: {np.mean(trajectories):.2f}, "
        f"Average Planning Time: {np.mean(planning_times):.6f}")

def main():
    for env, output_dir, name in zip(train_envs, output_dirs, names):
        test(env, model, 10, output_dir, name)

if __name__ == "__main__":
    main()