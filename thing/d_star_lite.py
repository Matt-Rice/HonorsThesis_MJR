import math
import numpy as np


class Node:
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost


def add_coordinates(node1: Node, node2: Node):
    return Node(node1.x + node2.x, node1.y + node2.y, node1.cost + node2.cost)


def compare_coordinates(node1: Node, node2: Node):
    return node1.x == node2.x and node1.y == node2.y


class DStarLite:
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

    def __init__(self, grid_size: int, detected_obstacles=None):
        self.x_max = grid_size
        self.y_max = grid_size
        self.detected_obstacles_xy = (
            np.empty((0, 2)) if detected_obstacles is None else np.array(detected_obstacles)
        )
        self.start = Node(0, 0)
        self.goal = Node(0, 0)
        self.U = []  # Priority queue
        self.km = 0.0
        self.kold = 0.0
        self.rhs = self.create_grid(float("inf"))
        self.g = self.create_grid(float("inf"))
        self.initialized = False

    def create_grid(self, val: float):
        return np.full((self.x_max, self.y_max), val)

    def is_valid(self, node: Node):
        return 0 <= node.x < self.x_max and 0 <= node.y < self.y_max

    def is_obstacle(self, node: Node):
        if self.detected_obstacles_xy.shape[0] > 0:
            x = np.array([node.x])
            y = np.array([node.y])
            is_x_equal = self.detected_obstacles_xy[:, 0] == x
            is_y_equal = self.detected_obstacles_xy[:, 1] == y
            return (is_x_equal & is_y_equal).any()
        return False

    def c(self, node1: Node, node2: Node):
        if self.is_obstacle(node2):
            return math.inf
        motion = Node(node1.x - node2.x, node1.y - node2.y)
        detected_motion = [m for m in self.motions if compare_coordinates(m, motion)]
        return detected_motion[0].cost if detected_motion else math.inf

    def h(self, s: Node):
        return max(abs(self.start.x - s.x), abs(self.start.y - s.y))  # Chebyshev distance

    def calculate_key(self, s: Node):
        return (
            min(self.g[s.x][s.y], self.rhs[s.x][s.y]) + self.h(s) + self.km,
            min(self.g[s.x][s.y], self.rhs[s.x][s.y]),
        )

    def get_neighbours(self, u: Node):
        return [add_coordinates(u, motion) for motion in self.motions if self.is_valid(add_coordinates(u, motion))]

    def update_vertex(self, u: Node):
        if not compare_coordinates(u, self.goal):
            self.rhs[u.x][u.y] = min(
                [self.c(u, sprime) + self.g[sprime.x][sprime.y] for sprime in self.get_neighbours(u)]
            )
        if any([compare_coordinates(u, node) for node, key in self.U]):
            self.U = [(node, key) for node, key in self.U if not compare_coordinates(node, u)]
            self.U.sort(key=lambda x: x[1])
        if self.g[u.x][u.y] != self.rhs[u.x][u.y]:
            self.U.append((u, self.calculate_key(u)))
            self.U.sort(key=lambda x: x[1])

    def initialize(self, start: Node, goal: Node):
        self.start = start
        self.goal = goal
        self.U = []  # Clear the priority queue
        self.km = 0.0
        self.rhs = self.create_grid(math.inf)
        self.g = self.create_grid(math.inf)
        self.rhs[self.goal.x][self.goal.y] = 0
        self.U.append((self.goal, self.calculate_key(self.goal)))
        self.initialized = True  # Ensure reinitialization flag is set

    def compare_keys(self, key_pair1: tuple, key_pair2: tuple):
        return key_pair1[0] < key_pair2[0] or (key_pair1[0] == key_pair2[0] and key_pair1[1] < key_pair2[1])

    def compute_shortest_path(self):
        while len(self.U) > 0 and (
            self.compare_keys(self.U[0][1], self.calculate_key(self.start))
            or self.rhs[self.start.x][self.start.y] != self.g[self.start.x][self.start.y]
        ):
            self.U.sort(key=lambda x: x[1])
            self.kold = self.U[0][1]
            u = self.U.pop(0)[0]
            if self.compare_keys(self.kold, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
                self.U.sort(key=lambda x: x[1])
            elif self.g[u.x, u.y] > self.rhs[u.x, u.y]:
                self.g[u.x, u.y] = self.rhs[u.x, u.y]
                for s in self.get_neighbours(u):
                    self.update_vertex(s)
            else:
                self.g[u.x, u.y] = math.inf
                for s in self.get_neighbours(u) + [u]:
                    self.update_vertex(s)

    def compute_current_path(self):
        path = []
        current = self.start
        while not compare_coordinates(current, self.goal):
            path.append(current)
            current = min(
                self.get_neighbours(current),
                key=lambda sprime: self.c(current, sprime) + self.g[sprime.x][sprime.y],
            )
        path.append(self.goal)
        return path

    def update_obstacles(self, detected_obstacles):
        self.detected_obstacles_xy = np.array(detected_obstacles)

    def main(self, start: Node, goal: Node):
        self.initialize(start, goal)
        self.compute_shortest_path()
        return self.compute_current_path()

