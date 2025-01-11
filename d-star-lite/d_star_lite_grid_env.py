import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np


class GridEnv(gym.Env):
    def __init__(self, grid_size=10, obstacle_positions=None, sx=0, sy=0, gx=50, gy=50):
        super(GridEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to the 8 possible movements in the grid
        self.action_space = spaces.Discrete(8)  # 8 possible motions

        # Observation space is the x, y position in the grid
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)

        self.grid_size = grid_size

        # Start and goal positions
        self.start_pos = np.array([sx, sy])
        self.goal_pos = np.array([gx, gy])

        # Place obstacles on the grid
        if obstacle_positions:
            self.obstacles = set(tuple(obstacle) for obstacle in obstacle_positions)
        else:
            self.obstacles = set()  # No obstacles initially

        # Initialize agent position
        self.agent_pos = np.array(self.start_pos)

    def reset(self):
        self.agent_pos = np.array(self.start_pos)
        return self.agent_pos

    def step(self, action):
        # Possible motions in the grid
        motions = [
            np.array([1, 0]),  # Right
            np.array([0, 1]),  # Up
            np.array([-1, 0]),  # Left
            np.array([0, -1]),  # Down
            np.array([1, 1]),  # Up-Right
            np.array([1, -1]),  # Down-Right
            np.array([-1, 1]),  # Up-Left
            np.array([-1, -1])  # Down-Left
        ]

        # Move the agent
        next_pos = self.agent_pos + motions[action]

        # Ensure the move is within the grid bounds and not into an obstacle
        if (0 <= next_pos[0] < self.grid_size and
                0 <= next_pos[1] < self.grid_size and
                tuple(next_pos) not in self.obstacles):
            self.agent_pos = next_pos

        # Check if goal is reached
        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1 if done else -0.1  # Give a small penalty for each step

        return self.agent_pos, reward, done, {}

    def render(self, path=None, detected_obstacles_xy=None, mode='human'):
        plt.clf()  # Clear previous frames

        # Initialize a blank grid for plotting
        grid = np.zeros((self.grid_size, self.grid_size))

        # Mark the agent's current position
        grid[int(self.agent_pos[1]), int(self.agent_pos[0])] = 1  # Agent

        # Mark the obstacles, making sure they are within grid bounds
        for obstacle in self.obstacles:
            ox, oy = int(obstacle[0]), int(obstacle[1])
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                grid[oy, ox] = -1  # Obstacles

        # Mark the goal
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            grid[gy, gx] = 2  # Goal

        # Draw the grid
        plt.imshow(grid, extent=(0, self.grid_size, 0, self.grid_size))

        # Plot the agent's path, if provided
        if path:
            px = [node.x for node in path]
            py = [node.y for node in path]
            plt.plot(px, py, color="red", label="Path taken")

            # Plot the agent's path, if provided
            if path:
                px = [node.x for node in path]
                py = [node.y for node in path]
                plt.plot(px, py, color="red", label="Path taken")

        # Plot detected obstacles, if provided
        if detected_obstacles_xy is not None and detected_obstacles_xy.size > 0:
            ox = detected_obstacles_xy[:, 0]
            oy = detected_obstacles_xy[:, 1]
            plt.plot(ox, oy, ".k", label="Detected Obstacles")

        plt.grid(True)
        plt.legend(loc='upper right')
       # plt.pause(0.001)  # Pause for a brief moment to create an animation effect
