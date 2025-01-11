"""
D* Lite grid planning
author: vss2sn (28676655+vss2sn@users.noreply.github.com)
Link to papers:
D* Lite (Link: http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf)
Improved Fast Replanning for Robot Navigation in Unknown Terrain
(Link: http://www.cs.cmu.edu/~maxim/files/dlite_icra02.pdf)
Implemented maintaining similarity with the pseudocode for understanding.
Code can be significantly optimized by using a priority queue for U, etc.
Avoiding additional imports based on repository philosophy.
"""
import math
import matplotlib.pyplot as plt
from d_star_lite_grid_env import GridEnv
import random
import numpy as np

show_animation = False
pause_time = 0.001

# Probability of creating a random obstacle
p_create_random_obstacle = 0


class Node:
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost


def add_coordinates(node1: Node, node2: Node):
    new_node = Node()
    new_node.x = node1.x + node2.x
    new_node.y = node1.y + node2.y
    new_node.cost = node1.cost + node2.cost
    return new_node


def compare_coordinates(node1: Node, node2: Node):
    return node1.x == node2.x and node1.y == node2.y


class DStarLite:
    # Please adjust the heuristic function (h) if you change the list of
    # possible motions
    motions = [
        Node(1, 0, 1),
        Node(0, 1, 1),
        Node(-1, 0, 1),
        Node(0, -1, 1),
        Node(1, 1, math.sqrt(2)),
        Node(1, -1, math.sqrt(2)),
        Node(-1, 1, math.sqrt(2)),
        Node(-1, -1, math.sqrt(2))
    ]

    def __init__(self, ox: list, oy: list):
        # Ensure that within the algorithm implementation all node coordinates
        # are indices in the grid and extend
        # from 0 to abs(<axis>_max - <axis>_min)
        self.x_min_world = int(min(ox))
        self.y_min_world = int(min(oy))
        self.x_max = int(abs(max(ox) - self.x_min_world))
        self.y_max = int(abs(max(oy) - self.y_min_world))
        self.obstacles = [Node(x - self.x_min_world, y - self.y_min_world)
                          for x, y in zip(ox, oy)]
        self.obstacles_xy = np.array(
            [[obstacle.x, obstacle.y] for obstacle in self.obstacles]
        )
        self.start = Node(0, 0)
        self.goal = Node(0, 0)
        self.U = list()  # type: ignore
        self.km = 0.0
        self.kold = 0.0
        self.rhs = self.create_grid(float("inf"))
        self.g = self.create_grid(float("inf"))
        self.detected_obstacles_xy = np.empty((0, 2))
        self.xy = np.empty((0, 2))
        # if show_animation:
        #     self.detected_obstacles_for_plotting_x = list()  # type: ignore
        #     self.detected_obstacles_for_plotting_y = list()  # type: ignore
        self.initialized = False

    def create_grid(self, val: float):
        return np.full((self.x_max, self.y_max), val)

    # def move_obstacles(self):
    #     motion_choices = [Node(1, 0), Node(0, 1), Node(-1, 0), Node(0, -1)]
    #     for obs in self.moving_obstacles:
    #         motion = random.choice(motion_choices)
    #         new_pos = add_coordinates(obs, motion)
    #         if self.is_valid(new_pos) and not self.is_obstacle(new_pos):
    #             obs.x, obs.y = new_pos.x, new_pos.y
    #             if show_animation:
    #                 plt.plot(obs.x, obs.y, ".k")
    #                 plt.pause(pause_time)

    def is_obstacle(self, node: Node):
        x = np.array([node.x])
        y = np.array([node.y])
        obstacle_x_equal = self.obstacles_xy[:, 0] == x
        obstacle_y_equal = self.obstacles_xy[:, 1] == y
        is_in_obstacles = (obstacle_x_equal & obstacle_y_equal).any()

        is_in_detected_obstacles = False
        if self.detected_obstacles_xy.shape[0] > 0:
            is_x_equal = self.detected_obstacles_xy[:, 0] == x
            is_y_equal = self.detected_obstacles_xy[:, 1] == y
            is_in_detected_obstacles = (is_x_equal & is_y_equal).any()

        return is_in_obstacles or is_in_detected_obstacles

    def c(self, node1: Node, node2: Node):
        if self.is_obstacle(node2):
            # Attempting to move from or to an obstacle
            return math.inf
        new_node = Node(node1.x - node2.x, node1.y - node2.y)
        detected_motion = list(filter(lambda motion:
                                      compare_coordinates(motion, new_node),
                                      self.motions))
        return detected_motion[0].cost

    def h(self, s: Node):
        # Cannot use the 2nd euclidean norm as this might sometimes generate
        # heuristics that overestimate the cost, making them inadmissible,
        # due to rounding errors etc (when combined with calculate_key)
        # To be admissible heuristic should
        # never overestimate the cost of a move
        # hence not using the line below
        # return math.hypot(self.start.x - s.x, self.start.y - s.y)

        # Below is the same as 1; modify if you modify the cost of each move in
        # motion
        # return max(abs(self.start.x - s.x), abs(self.start.y - s.y))
        return 1

    def calculate_key(self, s: Node):
        return (min(self.g[s.x][s.y], self.rhs[s.x][s.y]) + self.h(s)
                + self.km, min(self.g[s.x][s.y], self.rhs[s.x][s.y]))

    def is_valid(self, node: Node):
        if 0 <= node.x < self.x_max and 0 <= node.y < self.y_max:
            return True
        return False

    def get_neighbours(self, u: Node):
        return [add_coordinates(u, motion) for motion in self.motions
                if self.is_valid(add_coordinates(u, motion))]

    def pred(self, u: Node):
        # Grid, so each vertex is connected to the ones around it
        return self.get_neighbours(u)

    def succ(self, u: Node):
        # Grid, so each vertex is connected to the ones around it
        return self.get_neighbours(u)

    def initialize(self, start: Node, goal: Node):
        self.start.x = start.x - self.x_min_world
        self.start.y = start.y - self.y_min_world
        self.goal.x = goal.x - self.x_min_world
        self.goal.y = goal.y - self.y_min_world
        if not self.initialized:
            self.initialized = True
            print('Initializing')
            self.U = list()  # Would normally be a priority queue
            self.km = 0.0
            self.rhs = self.create_grid(math.inf)
            self.g = self.create_grid(math.inf)
            self.rhs[self.goal.x][self.goal.y] = 0
            self.U.append((self.goal, self.calculate_key(self.goal)))
            self.detected_obstacles_xy = np.empty((0, 2))

    def update_vertex(self, u: Node):
        if not compare_coordinates(u, self.goal):
            self.rhs[u.x][u.y] = min([self.c(u, sprime) +
                                      self.g[sprime.x][sprime.y]
                                      for sprime in self.succ(u)])
        if any([compare_coordinates(u, node) for node, key in self.U]):
            self.U = [(node, key) for node, key in self.U
                      if not compare_coordinates(node, u)]
            self.U.sort(key=lambda x: x[1])
        if self.g[u.x][u.y] != self.rhs[u.x][u.y]:
            self.U.append((u, self.calculate_key(u)))
            self.U.sort(key=lambda x: x[1])

    def compare_keys(self, key_pair1: tuple[float, float],
                     key_pair2: tuple[float, float]):
        return key_pair1[0] < key_pair2[0] or \
            (key_pair1[0] == key_pair2[0] and key_pair1[1] < key_pair2[1])

    def compute_shortest_path(self):
        self.U.sort(key=lambda x: x[1])
        has_elements = len(self.U) > 0
        start_key_not_updated = self.compare_keys(
            self.U[0][1], self.calculate_key(self.start)
        )
        rhs_not_equal_to_g = self.rhs[self.start.x][self.start.y] != \
                             self.g[self.start.x][self.start.y]
        while has_elements and start_key_not_updated or rhs_not_equal_to_g:
            self.kold = self.U[0][1]
            u = self.U[0][0]
            self.U.pop(0)
            if self.compare_keys(self.kold, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
                self.U.sort(key=lambda x: x[1])
            elif (self.g[u.x, u.y] > self.rhs[u.x, u.y]).any():
                self.g[u.x, u.y] = self.rhs[u.x, u.y]
                for s in self.pred(u):
                    self.update_vertex(s)
            else:
                self.g[u.x, u.y] = math.inf
                for s in self.pred(u) + [u]:
                    self.update_vertex(s)
            self.U.sort(key=lambda x: x[1])
            start_key_not_updated = self.compare_keys(
                self.U[0][1], self.calculate_key(self.start)
            )
            rhs_not_equal_to_g = self.rhs[self.start.x][self.start.y] != \
                                 self.g[self.start.x][self.start.y]

    def detect_changes(self):
        changed_vertices = list()
        if len(self.spoofed_obstacles) > 0:
            for spoofed_obstacle in self.spoofed_obstacles[0]:
                if compare_coordinates(spoofed_obstacle, self.start) or \
                        compare_coordinates(spoofed_obstacle, self.goal):
                    continue
                changed_vertices.append(spoofed_obstacle)
                self.detected_obstacles_xy = np.concatenate(
                    (
                        self.detected_obstacles_xy,
                        [[spoofed_obstacle.x, spoofed_obstacle.y]]
                    )
                )
                # if show_animation:
                #     self.detected_obstacles_for_plotting_x.append(
                #         spoofed_obstacle.x + self.x_min_world)
                #     self.detected_obstacles_for_plotting_y.append(
                #         spoofed_obstacle.y + self.y_min_world)
                #     plt.plot(self.detected_obstacles_for_plotting_x,
                #              self.detected_obstacles_for_plotting_y, ".k")
                #     plt.pause(pause_time)
            self.spoofed_obstacles.pop(0)

        # Allows random generation of obstacles
        random.seed()
        if random.random() > 1 - p_create_random_obstacle:
            x = random.randint(0, self.x_max - 1)
            y = random.randint(0, self.y_max - 1)
            new_obs = Node(x, y)
            if compare_coordinates(new_obs, self.start) or \
                    compare_coordinates(new_obs, self.goal):
                return changed_vertices
            changed_vertices.append(Node(x, y))
            self.detected_obstacles_xy = np.concatenate(
                (
                    self.detected_obstacles_xy,
                    [[x, y]]
                )
            )
            if show_animation:
                self.detected_obstacles_for_plotting_x.append(x +
                                                              self.x_min_world)
                self.detected_obstacles_for_plotting_y.append(y +
                                                              self.y_min_world)
                plt.plot(self.detected_obstacles_for_plotting_x,
                         self.detected_obstacles_for_plotting_y, ".k")
                plt.pause(pause_time)
        return changed_vertices

    def compute_current_path(self):
        path = list()
        current_point = Node(self.start.x, self.start.y)
        while not compare_coordinates(current_point, self.goal):
            path.append(current_point)
            current_point = min(self.succ(current_point),
                                key=lambda sprime:
                                self.c(current_point, sprime) +
                                self.g[sprime.x][sprime.y])
        path.append(self.goal)
        return path

    def compare_paths(self, path1: list, path2: list):
        if len(path1) != len(path2):
            return False
        for node1, node2 in zip(path1, path2):
            if not compare_coordinates(node1, node2):
                return False
        return True

    # def display_path(self, path: list, colour: str, alpha: float = 1.0):
    #     px = [(node.x + self.x_min_world) for node in path]
    #     py = [(node.y + self.y_min_world) for node in path]
    #     drawing = plt.plot(px, py, colour, alpha=alpha)
    #     plt.pause(pause_time)
    #     return drawing
    def get_action(self, current_pos, next_step):
        # Map the movement between current_pos and next_step to an action
        motions = {
            (1, 0): 0,  # Right
            (0, 1): 1,  # Up
            (-1, 0): 2,  # Left
            (0, -1): 3,  # Down
            (1, 1): 4,  # Up-right
            (1, -1): 5,  # Down-right
            (-1, 1): 6,  # Up-left
            (-1, -1): 7  # Down-left
        }
        # Calculate the difference between the current position and the next step
        dx = next_step.x - current_pos[0]
        dy = next_step.y - current_pos[1]
        return motions.get((dx, dy), None)  # Return the action or None if invalid

    def main(self, start: Node, goal: Node, env: GridEnv,
             spoofed_ox: list, spoofed_oy: list):
        self.spoofed_obstacles = [[Node(x - self.x_min_world,
                                        y - self.y_min_world)
                                   for x, y in zip(rowx, rowy)]
                                  for rowx, rowy in zip(spoofed_ox, spoofed_oy)
                                  ]
        pathx = []
        pathy = []
        self.initialize(start, goal)
        last = self.start
        self.compute_shortest_path()
        pathx.append(self.start.x + self.x_min_world)
        pathy.append(self.start.y + self.y_min_world)

        # Initial Render
        path = self.compute_current_path()
        env.render(path=path, detected_obstacles_xy=self.detected_obstacles_xy)
        plt.pause(.001)
        # if show_animation:
        #     current_path = self.compute_current_path()
        #     previous_path = current_path.copy()
        #     previous_path_image = self.display_path(previous_path, ".c",
        #                                             alpha=0.3)
        #     current_path_image = self.display_path(current_path, ".c")

        while not compare_coordinates(self.goal, self.start):
            # self.move_obstacles()
            if self.g[self.start.x][self.start.y] == math.inf:
                print("No path possible")
                return False, pathx, pathy
            self.start = min(self.succ(self.start),
                             key=lambda sprime:
                             self.c(self.start, sprime) +
                             self.g[sprime.x][sprime.y])
            pathx.append(self.start.x + self.x_min_world)
            pathy.append(self.start.y + self.y_min_world)

            # Render after each step
            path = self.compute_current_path()
            env.render(path=path, detected_obstacles_xy=self.detected_obstacles_xy)
            plt.pause(.001)

            # Detect changes in the environment
            changed_vertices = self.detect_changes()
            if len(changed_vertices) != 0:
                print("New obstacle detected")
                self.km += self.h(last)
                last = self.start
                for u in changed_vertices:
                    if compare_coordinates(u, self.start):
                        continue
                    self.rhs[u.x][u.y] = math.inf
                    self.g[u.x][u.y] = math.inf
                    self.update_vertex(u)
                self.compute_shortest_path()

                # if show_animation:
                #     new_path = self.compute_current_path()
                #     if not self.compare_paths(current_path, new_path):
                #         current_path_image[0].remove()
                #         previous_path_image[0].remove()
                #         previous_path = current_path.copy()
                #         current_path = new_path.copy()
                #         previous_path_image = self.display_path(previous_path,
                #                                                 ".c",
                #                                                 alpha=0.3)
                #         current_path_image = self.display_path(current_path,
                #                                                ".c")
                #         plt.pause(pause_time)
        print("Path found")
        return True, pathx, pathy


def main():
    # Set obstacle positions (if not using the environment's default obstacle positions)
    ox, oy = [], []
    for i in range(-10, 60):  # -10 to 59
        ox.append(i)
        oy.append(-10)
    for i in range(-10, 60):  # -10 to 59
        ox.append(59)
        oy.append(i)
    for i in range(-10, 60):  # -10 to 59
        ox.append(i)
        oy.append(59)
    for i in range(-10, 60):  # -10 to 59
        ox.append(-10)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)

    # Obstacles discovered at time = row (simulated here)
    spoofed_ox = [[], [], [], [i for i in range(0, 21)] + [0 for _ in range(0, 20)], [i for i in range(40, 58)]]
    spoofed_oy = [[], [], [], [20 for _ in range(0, 21)] + [i for i in range(0, 20)], [42 for i in range(40, 58)]]

    # Start and goal position
    sx = 10  # [m]
    sy = 10  # [m]
    gx = 50  # [m]
    gy = 50  # [m]

    # Create a grid environment with custom size and obstacles
    env = GridEnv(grid_size=60, obstacle_positions=list(zip(ox, oy)), sx=sx, sy=sy, gx=gx, gy=gy)

    # Reset the environment to the initial state
    env.reset()

    # Initialize D* Lite algorithm
    dstarlite = DStarLite(ox, oy)
    dstarlite.main(Node(x=sx, y=sy), Node(x=gx, y=gy), env=env, spoofed_ox=spoofed_ox, spoofed_oy=spoofed_oy)

    # # Main loop for agent movement and rendering
    # done = False
    # while not done:
    #     # Compute the current path
    #     path = dstarlite.compute_current_path()
    #
    #     # Render the environment with the current path and detected obstacles
    #     env.render(path=path, detected_obstacles_xy=dstarlite.detected_obstacles_xy)
    #
    #     # If there is a valid path, take the next step
    #     if len(path) > 1:
    #         next_step = path[1]  # The next node in the path
    #     else:
    #         next_step = path[0]  # If at the goal, stay at the goal
    #
    #     # Get the action (e.g., move up, down, left, or right)
    #     action = dstarlite.get_action(env.agent_pos, next_step)
    #
    #     # Take the step in the environment based on the action
    #     current_state, reward, done, _ = env.step(action)
    #
    #     # Detect any changes in the environment (e.g., new obstacles)
    #     detected_changes = dstarlite.detect_changes()
    #
    #     # If changes are detected, recompute the shortest path
    #     if detected_changes:
    #         print("New obstacle detected, replanning...")
    #         dstarlite.compute_shortest_path()

    # Final render when the loop is done
    plt.show()


if __name__ == "__main__":
    main()
