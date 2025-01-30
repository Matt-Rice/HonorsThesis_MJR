import os
import matplotlib.pyplot as plt
import numpy as np
import time
from grid_env import GridEnvironment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DDPG

# Directory to save the images
output_dir = "./test_runs/ddpg_maze"
os.makedirs(output_dir, exist_ok=True)

# Wall (train to 40)
# ox = [5 for _ in range(8)]
# oy = [y for y in range(8)]

# Small maze (train to 90)
ox = [7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
oy = [2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

# Passage (train to 90)
# ox = [7, 7, 7, 7, 4, 5, 6, 3, 2, 1, 0]
# oy = [0, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

# space
# ox = [7, 7, 7, 7, 7, 4, 6, 3, 2, 1, 0]
# oy = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4]

# Reset the environment
env = GridEnvironment(sx=0, sy=0, gx=9, gy=9, grid_size=10, ox=ox, oy=oy)
test_env = Monitor(env)
model = DDPG.load('./curriculum_learning/maze/overfitted_wall.zip')
episodes = 10
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

    print(f"Episode {episode+1}: Reward: {total_reward}, Steps: {steps}, Trajectory Length: {trajectory_length}, Total Time: {total_planning_time:.6f}")

    env.render(save_dir=f"{output_dir}/{episode}")

# Final averages across all episodes
print(f"Average Reward: {np.mean(rewards):.2f}, Average Steps: {np.mean(steps_list):.2f}, "
      f"Average Trajectory Length: {np.mean(trajectories):.2f}, "
      f"Average Planning Time: {np.mean(planning_times):.6f}")