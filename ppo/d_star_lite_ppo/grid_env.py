import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from d_star_lite import DStarLite, Node
import time
import random

def reward_shaping_function(
    agent_position, goal_position, pathx, pathy, obstacle_hit=False, 
    previous_distance_to_goal=None, previous_position=None, visited_positions=None, 
    last_nearest_index=None, grid_size=60
):
    """
    Reward shaping function with directional progress incentive, path following reward,
    penalties for backtracking, and revisiting penalties.
    """
    path = list(zip(pathx, pathy))
    current_distance_to_goal = np.linalg.norm(np.array(goal_position) - np.array(agent_position))
    progress_reward = (previous_distance_to_goal - current_distance_to_goal) / grid_size  # Normalize

    # Minimum distance to path penalty
    min_distance_to_path = min(
        np.linalg.norm(np.array(agent_position) - np.array(path_point))
        for path_point in path
    )
    path_penalty_weight = 0.5  # Reduce importance of path penalty
    path_penalty = path_penalty_weight * (-min_distance_to_path / grid_size)  # Normalize

    # Obstacle penalty
    obstacle_penalty = -1 if obstacle_hit else 0

    # Penalize backward motion (using dot product with path direction)
    backward_penalty = 0
    if previous_position is not None:
        movement_vector = np.array(agent_position) - np.array(previous_position)
        goal_direction_vector = np.array(goal_position) - np.array(previous_position)
        goal_direction_unit_vector = goal_direction_vector / (np.linalg.norm(goal_direction_vector) + 1e-6)
        forward_motion = np.dot(movement_vector, goal_direction_unit_vector)  # Positive if moving toward goal
        if forward_motion < 0:
            backward_penalty = -1

    # Reward for following the D* Lite path sequentially
    nearest_index = np.argmin([np.linalg.norm(np.array(agent_position) - np.array(p)) for p in path])
    if nearest_index + 1 < len(path):
        next_path_point = np.array(path[nearest_index + 1])
        movement_vector = np.array(agent_position) - np.array(previous_position)
        path_progress_vector = next_path_point - np.array(path[nearest_index])
        path_progress_unit_vector = path_progress_vector / (np.linalg.norm(path_progress_vector) + 1e-6)
        path_follow_reward = np.dot(movement_vector, path_progress_unit_vector)  # Reward for moving in the path's direction
    else:
        path_follow_reward = 0

    # Penalize backtracking on the path
    backtracking_penalty = 0
    if last_nearest_index is not None and nearest_index <= last_nearest_index:
        backtracking_penalty = -1

    # Penalize revisiting positions
    revisit_penalty = 0
    if tuple(agent_position) in visited_positions:
        revisit_penalty = -0.5
    visited_positions.add(tuple(agent_position))

    # Reward shaping for closer proximity to goal
    shaping_term = -current_distance_to_goal / grid_size  # Normalize

    # Final reward
    reward = (
        progress_reward + path_penalty + obstacle_penalty + backward_penalty +
        path_follow_reward + backtracking_penalty + revisit_penalty + shaping_term
    )

    # print(f"Progress Reward: {progress_reward}, Path Follow Reward: {path_follow_reward}, "
    #   f"Backward Penalty: {backward_penalty}, Revisit Penalty: {revisit_penalty}")


    return reward, current_distance_to_goal, nearest_index





class GridEnvironment(gym.Env):
    def __init__(self, sx=0, sy=0, gx=59, gy=59, ox=None, oy=None, grid_size=60, max_radius=1, num_obs=0.0):
        super(GridEnvironment, self).__init__()

        self.grid_size = grid_size
        self.max_radius = max_radius
        
        self.num_obs = num_obs

        max_grid_size = 60
        self.observation_space = spaces.Box(
            low=0, high=max_grid_size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-max_radius, high=max_radius, shape=(2,), dtype=np.float32)

         # Define start and goal positions
        self.sx, self.sy = sx, sy
        self.gx, self.gy = gx, gy
        self.start_position = (sx, sy)
        self.goal_position = (gx, gy)

        # Define obstacle lists
        self.ox = ox if ox is not None else []
        self.oy = oy if oy is not None else []
        self.obstacles = list(zip(self.ox, self.oy))

        # Initialize grid and add obstacles
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.add_obstacles()
        
        # Initialize D* Lite
        self.d_star_lite = DStarLite(self.grid_size, detected_obstacles=self.obstacles)
        self.d_star_lite.update_obstacles(self.obstacles)
        self.pathx = []
        self.pathy = []
        
        # Set initial position of the agent
        self.agent_position = self.start_position

        # Set trajectory and trajectory length
        self.trajectory = [self.agent_position]
        self.trajectory_length = 0

        # List of visited positions to incentivize not repeating locations
        self.visited_positions = set([self.start_position])

        self.previous_distance_to_goal = np.linalg.norm(np.array(self.goal_position) - np.array(self.agent_position))

        # Attributes to keep track of rollout duration
        self.rollout_start_time = None
        self.rollout_duration = 0
        self.episode_number = 0

    def add_obstacles(self):
        
        for x, y in zip(self.ox, self.oy):
            # Avoid placing obstacles at the start or goal position
            if (x, y) != self.start_position and (x, y) != self.goal_position:
                self.grid[x, y] = 1  # 1 represents an obstacle
    
    def add_new_obstacles(self, obstacles: list):
        for x, y in obstacles:
            # Avoid placing obstacles at the start or goal position or obstacles are repeated
            if (x, y) != self.start_position and (x, y) != self.goal_position or self.grid[x, y] == 1:
                self.grid[x, y] = 1  # 1 represents an obstacle
                self.obstacles.append((x, y))
        self.d_star_lite.update_obstacles(self.obstacles)


    def step(self, action):
        # Increase step count by 1
        self.step_count += 1

        # Previous position
        previous_position = self.agent_position

        # Scale the action if it exceeds max_radius
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > self.max_radius:
            action = (action / action_magnitude) * self.max_radius

        # Calculate the target position based on the action
        target_position = np.array(self.agent_position) + action
        target_position = np.clip(target_position, 0, self.grid_size - 1)

        # Initialize the current position for collision checking
        current_position = np.array(self.agent_position, dtype=np.float32)

        # Calculate the direction vector for interpolation
        direction = target_position - current_position
        steps = max(int(np.ceil(np.linalg.norm(direction))), 1)  # Ensure at least 1 step
        direction_step = direction / steps  # Small step increments

        obstacle_hit = False
        for _ in range(steps):
            # Move one step
            current_position += direction_step

            # Round to the nearest grid cell for obstacle checking
            grid_position = np.round(current_position).astype(int)

            # Check for obstacles
            if self.grid[grid_position[0], grid_position[1]] == 1:
                obstacle_hit = True
                break  # Stop checking further if an obstacle is hit

        # Update the agent's position if no obstacle is hit
        if not obstacle_hit:
            # Calculate the distance traveled in this step
            step_distance = np.linalg.norm(np.array(target_position) - np.array(previous_position))
            self.trajectory_length += step_distance  # Add the step distance to the total trajectory length

            previous_position = self.agent_position
            self.agent_position = target_position  # Store as float for smooth rendering
            self.trajectory.append(self.agent_position.tolist())  # Append to trajectory

        # Check if the goal is reached
        terminated = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal_position)) < 5e-1

        truncated = False

        # Calculate reward using the updated reward shaping function
        reward, self.previous_distance_to_goal, nearest_index = reward_shaping_function(
            agent_position=self.agent_position,
            goal_position=self.goal_position,
            pathx=self.pathx,
            pathy=self.pathy,
            obstacle_hit=obstacle_hit,
            previous_distance_to_goal=self.previous_distance_to_goal,
            previous_position=previous_position,
            visited_positions=self.visited_positions,
            last_nearest_index=getattr(self, "last_nearest_index", None),
            grid_size=self.grid_size
        )
        self.last_nearest_index = nearest_index

        # Assign high reward for reaching the goal
        if terminated:
            self.rollout_duration = time.time() - self.rollout_start_time
            reward += 100
            print(f"Episode terminated on step {self.step_count}")

        if self.step_count >= 2048:
            truncated = True
            print(f"Truncated on step {self.step_count}")

        return (
            np.array(self.agent_position, dtype=np.float32),
            reward,
            terminated,
            truncated,
            {"rollout_duration": self.rollout_duration, "trajectory_length": self.trajectory_length, "path_length": {len(self.pathx)}}
        )

    def reset(self, **kwargs):

        # print("resetting")
        self.step_count = 0
        self.rollout_duration = 0
        self.rollout_start_time = time.time()
        self.trajectory_length = 0
        self.episode_number += 1

        # # Randomize start and goal positions after the first 100 episodes of training
        # if self.episode_number > 100:    
        #     self.sx, self.sy = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
        #     self.gx, self.gy = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
        #     # print("start")
        #     # Ensure start and goal positions are not the same
        #     while (self.sx, self.sy) == (self.gx, self.gy):
        #         self.gx, self.gy = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)


        # self.start_position = (self.sx, self.sy)
        # self.goal_position = (self.gx, self.gy)
        self.agent_position = self.start_position

        # Randomize obstacles
        if self.num_obs != 0.0:
            num_obstacles = random.randint(5, int(self.grid_size ** 2 * self.num_obs))  # obs_num% of the grid as obstacles
            self.ox = []
            self.oy = []

            for _ in range(num_obstacles):
                ox, oy = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)

                # Avoid placing obstacles at the start or goal positions
                while (ox, oy) == (self.sx, self.sy) or (ox, oy) == (self.gx, self.gy) or (ox, oy) in zip(self.ox, self.oy):
                    ox, oy = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)

                self.ox.append(ox)
                self.oy.append(oy)

        self.obstacles = list(zip(self.ox, self.oy))
        print("FINISH OBSTACLE")
        # Update the grid with obstacles
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.add_obstacles()
        self.d_star_lite.update_obstacles(self.obstacles)

        # Check for a valid path using D* Lite
        self.d_star_lite.initialize(Node(self.sx, self.sy), Node(self.gx, self.gy))
        path = self.d_star_lite.main(Node(self.sx, self.sy), Node(self.gx, self.gy))
        self.pathx = [node.x for node in path]
        self.pathy = [node.y for node in path]
        #print(f"PATH: {self.pathx}")

        # Reset trajectory and visited positions
        self.trajectory = [self.agent_position]
        self.visited_positions = set([self.agent_position])
        self.previous_distance_to_goal = np.linalg.norm(np.array(self.goal_position) - np.array(self.agent_position))

        print(f"s:({self.sx},{self.sy}), g:({self.gx},{self.gy}), path: {True if path else False}")
        if path:  # If a valid path is found
            return np.array(self.agent_position, dtype=np.float32)
        else:
            self.reset()


    def render(self, mode='human', save_dir=None):
        grid = np.full((self.grid_size, self.grid_size, 3), [46, 149, 209], dtype=np.uint8)

        # Mark the start and goal
        grid[self.sx, self.sy] = [0, 255, 0]  # Green for start
        grid[self.gx, self.gy] = [0, 0, 255]  # Blue for goal

        # Obstacles
        for x, y in zip(self.ox, self.oy):
            grid[x, y] = [0, 0, 0]  # Black for obstacles

        # D* Lite Path
        if self.pathx:
            for x, y in zip(self.pathx, self.pathy):
                grid[int(x), int(y)] = [255, 255, 0]  # Yellow for D* Lite path

        if mode == 'rgb_array':
            agentx, agenty = map(int, self.agent_position)
            grid[agentx, agenty] = [255, 0, 0]  # Red for agent
            return grid

        elif mode == 'human':
            # Create a plot to visualize the grid and agent's trajectory
            fig, ax = plt.subplots(figsize=(8, 8))

            # Draw the grid
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    rect = plt.Rectangle((x, y), 1, 1, linewidth=0.5, edgecolor='gray', facecolor='none')
                    ax.add_patch(rect)

            # Mark the start, goal, and obstacles
            ax.add_patch(plt.Rectangle((self.sx, self.sy), 1, 1, color='green', label='Start'))
            ax.add_patch(plt.Rectangle((self.gx, self.gy), 1, 1, color='blue', label='Goal'))
            for x, y in zip(self.ox, self.oy):
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))

            # Plot the D* Lite path
            #print(f" render d* Lite path: {list(zip(self.pathx, self.pathy))}")

            if self.pathx and self.pathy:
                ax.plot(self.pathx, self.pathy, color='yellow', label='D* Lite Path', linewidth=2)
                #print("plotting the d*")

            # Plot the agent's trajectory (smooth floating-point positions)
            trajectory = np.array(self.trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color='red', label='Agent Path', linewidth=2)

            # Plot the current agent position
            ax.scatter(self.agent_position[0], self.agent_position[1], color='red', s=50, label='Agent')

            # Set up the plot
            ax.set_xlim(0, self.grid_size)
            ax.set_ylim(0, self.grid_size)
            ax.set_aspect('equal')
            ax.legend()
            plt.title(f"Agent: {self.agent_position} | Start: {self.start_position} | Goal: {self.goal_position}")

            if save_dir:
                plt.savefig(save_dir)
            else:
                plt.show()
