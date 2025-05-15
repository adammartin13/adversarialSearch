from os import WCONTINUED

import numpy as np
import heapq
import math
import random
from collections import deque
from typing import List, Tuple, Optional


def get_legal_actions(world, player):
    directions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]])
    actions = []
    for action in directions:
        try:
            if world[action[0] + player[0]][action[1] + player[1]] == 0:  # isn't collision
                actions.append(action)
        except IndexError:
            continue  # location out of bounds
    return actions


# Calculate heuristic using the diagonal distance technique
def heuristic(start, end):
    # H = max(|x - end_x|, |y - end_y|)
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    return max(dx, dy)

def apply_position(action, player):
    position = player.copy()
    position[0] += action[0]
    position[1] += action[1]
    return position

# Simulate pursuers actions for greater avoidance
def pursuer_positions(world, pursuer, depth):
    reachable = set()
    frontier = {tuple(pursuer)}

    for it in range(depth):
        next_frontier = set()
        for position in frontier:
            actions = get_legal_actions(world, position)
            for action in actions:
                new_position = tuple(apply_position(position, action))
                if new_position not in reachable:
                    reachable.add(new_position)
                    next_frontier.add(new_position)
        frontier = next_frontier

    return reachable


def check_winner(player, pursued):
    # Check if the provided player/target condition is a winning one
    return player[0] == pursued[0] and player[1] == pursued[1]


# If the target is never reachable, hit a wall so they don't get points
def closest_obstacle(world, player):
    visited = set()
    queue = deque()
    start = tuple(player)
    queue.append((start, 0))
    visited.add(start)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        (x, y), dist = queue.popleft()

        if world[y][x] == 1:  # Found obstacle
            return np.array([x, y])

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < world.shape[1] and 0 <= ny < world.shape[0]:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

    return np.array(player)  # No obstacle found; return player's position

# If we're close to capture and there is an obstacle nearby, crash into it
def crash(world, player):
    directions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]])
    actions = get_legal_actions(world, player)
    action_tuples = [tuple(action) for action in actions]

    for direction in directions:
        if tuple(direction) not in action_tuples:
            return direction

    return np.array([0,0])  # no wall found


# A* Search
def a_star(world, current, pursued, pursuer):
    start = tuple(map(int, current))
    end = tuple(map(int, pursued))

    # f = g + h
    # open_set priority queue
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    came_from = {start: None}  # backtracking for path reconstruction
    g = {start: 0}
    visited = set()

    while open_set:
        _, current_g, current_node = heapq.heappop(open_set)  # f, g, node

        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == end:  # goal reached, backtrack to determine step taken
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path = path[::-1]  # list of nodes traveled
            if len(path) > 1:
                next_step = np.array(path[1])
                # return next_step - current  # Return next step
                # crash condition
                if heuristic(current, pursued) > 1 and heuristic(next_step - current, pursuer) >= 1 and heuristic(
                        current, crash(world, current)) == 1:
                    return crash(world, current)
                else:
                    return next_step - current  # Return next step
            else:
                return np.array([0, 0])  # Otherwise we're at the goal, stand still

        # check where we can go next
        for action in get_legal_actions(world, current_node):
            neighbor = (current_node[0] + action[0], current_node[1] + action[1])
            temp_g = current_g + 1
            if neighbor not in g or temp_g < g[neighbor]:  # best way to reach neighbor
                g[neighbor] = temp_g
                f = temp_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f, temp_g, neighbor))
                came_from[neighbor] = current_node

    # return np.array([0, 0])  # No path found
    return closest_obstacle(world, current)  # No path found


def apply_pursuer(state, player, pursuer, pursued):
    # Calculate the best move the pursuer could make
    # For all directions they could take, which is legal and minimizes distance to the player
    return apply_position(a_star(state, pursuer, player, pursued), pursuer)


def apply_pursued(state, player, pursued):
    # Calculate the best move the pursued could make
    # For all directions they could take, which is best for evading us while reaching its target
    actions = get_legal_actions(state, pursued)
    if not actions:
        return pursued

    best_score = -float("inf")
    best_action = actions[0]

    for action in actions:
        new_pos = apply_position(action, pursued)
        dist_from_player = heuristic(new_pos, player)
        dist_to_target = heuristic(new_pos, pursued)
        score = dist_from_player - dist_to_target
        if score > best_score:
            best_score = score
            best_action = action

    return apply_position(best_action, pursued)

def simulate(node):
    # simulate a single round with A*
    # A* for player and pursuer, random for pursued
    new_player = apply_position(a_star(node.state, node.player, node.pursued, node.pursuer), node.player)
    actions = get_legal_actions(node.state, node.pursued)
    new_pursued = apply_position(random.choice(actions), node.pursued) if actions else node.pursued
    new_pursuer = apply_position(a_star(node.state, node.pursuer, new_player, node.pursued), node.pursuer)
    simulation = Node(node.state, new_player, new_pursued, new_pursuer)
    return simulation


# Monte Carlo Tree Search
def mcts(state, player, pursued, pursuer):
    root = Node(state, player, pursued, pursuer)
    simulations = 100  # SIMULATIONS = 1 FOR TESTING ONLY

    for it in range(simulations):
        node = root

        # Selection
        # print('selection')
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion
        # print('expansion')
        if not check_winner(node.player, node.pursued):
            node = node.expand()
            if node is None:  # no expansion possible
                continue

        # Simulation
        # print('simulation')
        result = node.rollout()

        # Backpropagation
        # print('backpropagation')
        depth = 0
        while node is not None:
            node.visits += 1
            node.value += result * (0.99 ** depth)  # Gamma = 0.99
            node = node.parent
            depth += 1

    chosen = root.best_child()
    return chosen.player - player

class Node:
    def __init__(self, state, player, pursued, pursuer, parent=None):
        self.state = state
        self.player = player
        self.pursued = pursued
        self.pursuer = pursuer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    directions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]])

    def is_fully_expanded(self):
        # Checks if every action that can be taken has been accounted for in children
        actions = get_legal_actions(self.state, self.player)
        children = []
        for child in self.children:
            children.append(tuple(child.player))

        for action in actions:
            next_player = apply_position(action, self.player)
            if tuple(next_player) not in children:
                return False
        return True

    def is_terminal(self):
        return self.player == self.pursued or self.player == self.pursuer

    def best_child(self):
        # Upper Confidence Bound for Trees
        # The sqrt has been changed as a safety against unvisited parents
        # UCT = Q / N + c * sqrt(ln(N_p + 1) / (N + 1e-6))
        # Q = total reward
        # N = number of visits
        # c = constant, typically sqrt(2)
        # N_p = number of parent visits
        chosen_child = None
        UCTs = []
        for child in self.children:
            Q = child.value
            N = child.visits + 1e-9
            c = 1.41
            Np = child.parent.visits

            if N == 0:
                UCT = math.inf
            else:
                UCT = Q / N + c * math.sqrt(math.log(Np + 1) / N)
            UCTs.append((UCT, child))

            highest_UCT = max(UCTs, key=lambda x: x[0])[0]
            best_children = [child for UCT, child in UCTs if UCT == highest_UCT]
            return random.choice(best_children)

        return chosen_child if chosen_child is not None else random.choice(self.children)

    def expand(self):
        # Try a new child that hasn't been pursued yet
        children = []
        for child in self.children:
            children.append(tuple(child.player))

        # Rather than picking child at random, pick the one with the best heuristic value
        actions = sorted(get_legal_actions(self.state, self.player),
                         key=lambda a: heuristic(apply_position(a, self.player), self.pursued))

        for action in actions:
            next_player = apply_position(action, self.player)
            if tuple(next_player) not in children:
                next_pursuer = apply_pursuer(self.state, next_player, self.pursuer, self.pursued)
                next_pursued = apply_pursued(self.state, next_player, self.pursued)
                player = self
                child = Node(self.state, next_player, next_pursued, next_pursuer, parent=player)
                player.children.append(child)
                return child

    def rollout(self, depth=30):  # DEPTH = 1 FOR TESTING ONLY
        # We simulate the game X number of turns from the current node or until we win or lose
        # We track values from these simulations so that when we backpropagate we can determine the best route
        simulation = self

        for step in range(depth):
            if simulation.player[0] == simulation.pursued[0] and simulation.player[1] == simulation.pursued[1]:
                return 100
            if simulation.player[0] == simulation.pursuer[0] and simulation.player[1] == simulation.pursuer[1]:
                return -100

            simulation = simulate(simulation)

        # smooth reward
        dist_pursued = heuristic(simulation.player, simulation.pursued)
        dist_pursuer = heuristic(simulation.player, simulation.pursuer)
        reward = 100 * (1 - dist_pursued / 30)  # 30 = max grid distance
        risk = -50 * (1 / (dist_pursuer + 1))

        return reward + risk

class PlannerAgent:
    def __init__(self):
        pass

    def plan_action(self, world: np.ndarray, current: np.ndarray, pursued: np.ndarray, pursuer: np.ndarray) -> Optional[
        np.ndarray]:
        """
        Parameters:
        - world (np.ndarray): A 2D numpy array representing the grid environment.
        - 0 represents a walkable cell.
        - 1 represents an obstacle.
        - current (np.ndarray): The (row, column) coordinates of the current position.
        - pursued (np.ndarray): The (row, column) coordinates of the agent to be pursued.
        - pursuer (np.ndarray): The (row, column) coordinates of the agent to evade from.
        """

        if heuristic(current, pursued) > 5 and heuristic(current, pursuer) > 5:
            return a_star(world, current, pursued, pursuer)
        else:
            return mcts(world, current, pursued, pursuer)

'''''''''''''''''''''''''''''
# OTHER DEVELOPED PLANNERS
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
# ADVERSARIAL A*
'''''''''''''''''''''''''''''
'''
import numpy as np
import heapq
from collections import deque
from typing import List, Tuple, Optional


def get_legal_actions(world, player):
    directions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]])
    actions = []
    for action in directions:
        try:
            if world[action[0] + player[0]][action[1] + player[1]] == 0:  # isn't collision
                actions.append(action)
        except IndexError:
            continue  # location out of bounds
    return actions


# Calculate heuristic using the diagonal distance technique
def heuristic(start, end):
    # H = max(|x - end_x|, |y - end_y|)
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    return max(dx, dy)

def apply_position(action, player):
    position = player.copy()
    position[0] += action[0]
    position[1] += action[1]
    return position

# Simulate pursuers actions for greater avoidance
def pursuer_positions(world, pursuer, depth):
    reachable = set()
    frontier = {tuple(pursuer)}

    for it in range(depth):
        next_frontier = set()
        for position in frontier:
            actions = get_legal_actions(world, position)
            for action in actions:
                new_position = tuple(apply_position(position, action))
                if new_position not in reachable:
                    reachable.add(new_position)
                    next_frontier.add(new_position)
        frontier = next_frontier

    return reachable


class PlannerAgent:

    def __init__(self):
        pass

    def plan_action(self, world: np.ndarray, current: np.ndarray, pursued: np.ndarray, pursuer: np.ndarray) -> Optional[
        np.ndarray]:
        start = tuple(map(int, current))
        end = tuple(map(int, pursued))

        # f = g + h
        # open_set priority queue
        open_set = []
        heapq.heappush(open_set, (heuristic(start, end), 0, start))
        came_from = {start: None}  # backtracking for path reconstruction
        g = {start: 0}
        visited = set()

        while open_set:
            _, current_g, current_node = heapq.heappop(open_set)  # f, g, node

            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node == end:  # goal reached, backtrack to determine step taken
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path = path[::-1]  # list of nodes traveled
                if len(path) > 1:
                    next_step = np.array(path[1])
                    return next_step - current  # Return next step
                else:
                    return np.array([0, 0])  # Otherwise we're at the goal, stand still

            # check where we can go next
            for action in get_legal_actions(world, current_node):
                neighbor = (current_node[0] + action[0], current_node[1] + action[1])
                temp_g = current_g + 1
                if neighbor not in g or temp_g < g[neighbor]:  # best way to reach neighbor
                    g[neighbor] = temp_g
                    danger_zone = pursuer_positions(world, pursuer, depth=2)
                    if neighbor in danger_zone:
                        danger_penalty = 30  # penalty for closeness to pursuer
                    else:
                        danger_penalty = 0
                    f = temp_g + heuristic(neighbor, end) + danger_penalty
                    heapq.heappush(open_set, (f, temp_g, neighbor))
                    came_from[neighbor] = current_node

        # return closest_obstacle(world, current)  # No path found
        return np.array([0, 0])  # No path found
'''
'''''''''''''''''''''''''''''
# MCST WITH A* HEURISTICS
'''''''''''''''''''''''''''''
'''
from operator import truediv
import numpy as np
import math
import random
import heapq
from typing import List, Tuple, Optional

# Calculate heuristic using the Chebyshev/diagonal distance technique
def heuristic(start, end):
    # H = max(|x - end_x|, |y - end_y|)
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    return max(dx, dy)

# A* Search to supplement MCTS randomization/heuristics
def a_star(state, player, target):
    rows, cols = state.shape
    start = tuple(map(int, player))
    end = tuple(map(int, target))

    # f = g + h
    # open_set priority queue
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    came_from = {start: None}  # backtracking for path reconstruction
    g = {start: 0}

    while open_set:
        _, current_g, current_node = heapq.heappop(open_set)  # f, g, node

        if current_node == end:  # goal reached, backtrack to determine step taken
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path = path[::-1]  # list of nodes traveled
            if len(path) > 1:
                next_step = np.array(path[1])
                return next_step - player  # Return next step
            else:
                return np.array([0, 0])  # Otherwise we're at the goal, stand still

        # check where we can go next
        for action in get_legal_actions(state, current_node):
            neighbor = (current_node[0] + action[0], current_node[1] + action[1])
            tentative_g = current_g + 1
            if neighbor not in g or tentative_g < g[neighbor]:  # best way to reach neighbor
                g[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
                came_from[neighbor] = current_node

    return np.array([0, 0])  # No path found

def get_legal_actions(world, player):
    directions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]])
    actions = []
    for action in directions:
        try:
            if world[action[0] + player[0]][action[1] + player[1]] == 0:  # isn't collision
                actions.append(action)
        except IndexError:
            continue  # location out of bounds
    return actions


def apply_position(action, player):
    position = player.copy()
    position[0] += action[0]
    position[1] += action[1]
    return position


def apply_pursuer(state, player, pursuer):
    # Calculate the best move the pursuer could make
    # For all directions they could take, which is legal and minimizes distance to the player
    return apply_position(a_star(state, pursuer, player), pursuer)


def apply_pursued(state, player, pursued):
    # Calculate the best move the pursued could make
    # For all directions they could take, which is best for evading us while reaching its target
    actions = get_legal_actions(state, pursued)
    if not actions:
        return pursued

    best_score = -float("inf")
    best_action = actions[0]

    for action in actions:
        new_pos = apply_position(action, pursued)
        dist_from_player = heuristic(new_pos, player)
        dist_to_target = heuristic(new_pos, pursued)
        score = dist_from_player - dist_to_target
        if score > best_score:
            best_score = score
            best_action = action

    return apply_position(best_action, pursued)


def check_winner(player, pursued):
    # Check if the provided player/target condition is a winning one
    return player[0] == pursued[0] and player[1] == pursued[1]


def simulate(node):
    # simulate a single round with A*
    # A* for player and pursuer, random for pursued
    new_player = apply_position(a_star(node.state, node.player, node.pursued), node.player)
    actions = get_legal_actions(node.state, node.pursued)
    new_pursued = apply_position(random.choice(actions), node.pursued) if actions else node.pursued
    new_pursuer = apply_position(a_star(node.state, node.pursuer, new_player), node.pursuer)
    simulation = Node(node.state, new_player, new_pursued, new_pursuer)
    return simulation

class Node:
    def __init__(self, state, player, pursued, pursuer, parent=None):
        self.state = state
        self.player = player
        self.pursued = pursued
        self.pursuer = pursuer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    directions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]])

    def is_fully_expanded(self):
        # Checks if every action that can be taken has been accounted for in children
        actions = get_legal_actions(self.state, self.player)
        children = []
        for child in self.children:
            children.append(tuple(child.player))

        for action in actions:
            next_player = apply_position(action, self.player)
            if tuple(next_player) not in children:
                return False
        return True

    def is_terminal(self):
        return self.player == self.pursued or self.player == self.pursuer

    def best_child(self):
        # Upper Confidence Bound for Trees
        # The sqrt has been changed as a safety against unvisited parents
        # UCT = Q / N + c * sqrt(ln(N_p + 1) / (N + 1e-6))
        # Q = total reward
        # N = number of visits
        # c = constant, typically sqrt(2)
        # N_p = number of parent visits
        chosen_child = None
        UCTs = []
        for child in self.children:
            Q = child.value
            N = child.visits + 1e-9
            c = 1.41
            Np = child.parent.visits

            if N == 0:
                UCT = math.inf
            else:
                UCT = Q / N + c * math.sqrt(math.log(Np + 1) / N)
            UCTs.append((UCT, child))

            highest_UCT = max(UCTs, key=lambda x: x[0])[0]
            best_children = [child for UCT, child in UCTs if UCT == highest_UCT]
            return random.choice(best_children)

        return chosen_child if chosen_child is not None else random.choice(self.children)


    def expand(self):
        # Try a new child that hasn't been pursued yet
        children = []
        for child in self.children:
            children.append(tuple(child.player))

        # Rather than picking child at random, pick the one with the best heuristic value
        actions = sorted(get_legal_actions(self.state, self.player),
                         key=lambda a: heuristic(apply_position(a, self.player), self.pursued))

        for action in actions:
            next_player = apply_position(action, self.player)
            if tuple(next_player) not in children:
                next_pursuer = apply_pursuer(self.state, next_player, self.pursuer)
                next_pursued = apply_pursued(self.state, next_player, self.pursued)
                player = self
                child = Node(self.state, next_player, next_pursued, next_pursuer, parent=player)
                player.children.append(child)
                return child

    def rollout(self, depth=30):  # DEPTH = 1 FOR TESTING ONLY
        # We simulate the game X number of turns from the current node or until we win or lose
        # We track values from these simulations so that when we backpropagate we can determine the best route
        simulation = self

        for step in range(depth):
            if simulation.player[0] == simulation.pursued[0] and simulation.player[1] == simulation.pursued[1]:
                return 100
            if simulation.player[0] == simulation.pursuer[0] and simulation.player[1] == simulation.pursuer[1]:
                return -100

            simulation = simulate(simulation)

        # smooth reward
        dist_pursued = heuristic(simulation.player, simulation.pursued)
        dist_pursuer = heuristic(simulation.player, simulation.pursuer)
        reward = 100 * (1 - dist_pursued / 30)  # 30 = max grid distance
        risk = -50 * (1 / (dist_pursuer + 1))

        return reward + risk


class PlannerAgent:
    def __init__(self):
        pass

    def plan_action(self, state: np.ndarray, player: np.ndarray, pursued: np.ndarray, pursuer: np.ndarray) -> Optional[
        np.ndarray]:
        root = Node(state, player, pursued, pursuer)
        simulations = 5  # SIMULATIONS = 1 FOR TESTING ONLY

        for it in range(simulations):
            node = root

            # Selection
            # print('selection')
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            # print('expansion')
            if not check_winner(node.player, node.pursued):
                node = node.expand()
                if node is None:  # no expansion possible
                    continue

            # Simulation
            # print('simulation')
            result = node.rollout()

            # Backpropagation
            # print('backpropagation')
            depth = 0
            while node is not None:
                node.visits += 1
                node.value += result * (0.99 ** depth)  # Gamma = 0.99
                node = node.parent
                depth += 1

        chosen = root.best_child()
        return chosen.player - player
'''