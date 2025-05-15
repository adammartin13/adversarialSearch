import numpy as np
import heapq
import math
import random
from functools import lru_cache

directions = [
    (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]

def get_legal_actions(world, player):
    rows, cols = world.shape
    x, y = player
    actions = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            if world[nx][ny] == 0:  # Valid move
                actions.append((dx, dy))

    return [np.array((dx, dy)) for dx, dy in actions]

def heuristic(start, end):
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    return max(dx, dy)

def apply_position(action, player):
    return player + action

def obstacle_penalty(world, position, radius=1, weight=1.0):
    x, y = position
    rows, cols = world.shape
    penalty = 0.0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and world[nx][ny] == 1:
                distance = max(abs(dx), abs(dy))
                penalty += weight / (distance + 1)
    return penalty

@lru_cache(maxsize=8192)
def get_estimated_probabilities(rotation_dict_tuple):
    left, safe, right = rotation_dict_tuple
    total = left + safe + right
    if total == 0:
        return {"left": 1/3, "safe": 1/3, "right": 1/3}
    return {
        "left": left / total,
        "safe": safe / total,
        "right": right / total
    }


def get_rotation_counts_tuple(agent):
    return tuple(agent.rotation[k] for k in ["left", "safe", "right"])

def action_freedom(world, position):
    return len(get_legal_actions(world, position))

def vectorized_obstacle_penalty(world, position, radius=1):
    x, y = position
    rows, cols = world.shape
    x_min = max(0, x - radius)
    x_max = min(rows, x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(cols, y + radius + 1)
    view = world[x_min:x_max, y_min:y_max]
    return np.sum(view == 1)

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

def a_star(world, current, pursued, pursuer, agent=None):
    start = tuple(current)
    end = tuple(pursued)
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    came_from = {start: None}
    g = {start: 0}

    if agent is not None:
        prob = get_estimated_probabilities(get_rotation_counts_tuple(agent))
    else:
        prob = {"left": 0.3, "safe": 0.4, "right": 0.3}

    def rotate_left(vec): return np.array([-vec[1], vec[0]])
    def rotate_right(vec): return np.array([vec[1], -vec[0]])

    while open_set:
        _, current_g, current_node = heapq.heappop(open_set)
        if current_node == end:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            if len(path) > 1:
                dx = path[1][0] - start[0]
                dy = path[1][1] - start[1]
                return np.array([dx, dy])
            return np.array([0, 0])

        for action in get_legal_actions(world, current_node):
            neighbor = (current_node[0] + action[0], current_node[1] + action[1])
            temp_g = current_g + 1

            if neighbor not in g or temp_g < g[neighbor]:
                danger_penalty = 5 / (heuristic(neighbor, pursuer) + 1)
                obstacle = obstacle_penalty(world, neighbor, radius=2, weight=5)
                freedom = action_freedom(world, neighbor)

                left_score = heuristic(apply_position(rotate_left(action), current_node), pursued)
                safe_score = heuristic(apply_position(action, current_node), pursued)
                right_score = heuristic(apply_position(rotate_right(action), current_node), pursued)

                expected_score = (
                    prob["left"] * left_score +
                    prob["safe"] * safe_score +
                    prob["right"] * right_score
                )

                f = temp_g + expected_score + danger_penalty + obstacle - freedom
                g[neighbor] = temp_g
                heapq.heappush(open_set, (f, temp_g, neighbor))
                came_from[neighbor] = current_node

    return crash(world, current)

def simulate(node):
    # Simulate a single round with A*
    # Assume target is greedy, select randomly from k-highest A* outcomes
    # Player selects best option for selected target location
    # Aggressor selects the best option provided player location
    k = 3
    intended = a_star(node.state, node.pursued, node.pursuer, node.player)
    all_actions = get_legal_actions(node.state, node.pursued)
    if not all_actions:
        new_pursued = node.pursued.copy()
    else:
        def action_distance(a):
            return heuristic(apply_position(a, node.pursued), apply_position(intended, node.pursued))

        closest = sorted(all_actions, key=action_distance)
        top_choices = [apply_position(a, node.pursued) for a in closest[:k]]
        new_pursued = random.choice(top_choices)

    new_player = apply_position(a_star(node.state, node.player, new_pursued, node.pursuer), node.player)

    pursuer_actions = get_legal_actions(node.state, node.pursuer)
    if pursuer_actions:
        new_pursuer = min(
            [apply_position(a, node.pursuer) for a in pursuer_actions],
            key=lambda pos: heuristic(pos, new_player)
        )
    else:
        new_pursuer = node.pursuer.copy()

    return Node(node.state, new_player, new_pursued, new_pursuer, parent=node)

class Node:
    def __init__(self, state, player, pursued, pursuer, parent=None):
        self.state = state
        self.player = player
        self.pursued = pursued
        self.pursuer = pursuer
        self.parent = parent
        self.children = []
        self.state_key = (tuple(player), tuple(pursued), tuple(pursuer))
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(get_legal_actions(self.state, self.player))

    def best_child(self, prob=None):
        # Upper Confidence Bound for Trees
        # The sqrt has been changed as a safety against unvisited parents
        # UCT = Q / N + c * sqrt(ln(N_p + 1) / (N + 1e-6))
        # Q = total reward
        # N = number of visits
        # c = constant, typically sqrt(2)
        # N_p = number of parent visits
        UCTs = []

        for child in self.children:
            Q = child.value
            N = child.visits + 1e-9
            c = 1.41
            Np = child.parent.visits

            UCT = Q / N + c * math.sqrt(math.log(Np + 1) / N)

            UCTs.append((UCT, child))

        highest_UCT = max(UCTs, key=lambda x: x[0])[0]
        best_children = [child for UCT, child in UCTs if UCT == highest_UCT]
        return random.choice(best_children)

    def expand(self):
        actions = get_legal_actions(self.state, self.player)
        sorted_actions = sorted(actions, key=lambda a: vectorized_obstacle_penalty(self.state, self.player + a))
        for action in sorted_actions:
            next_player = self.player + action
            if not any(np.array_equal(c.player, next_player) for c in self.children):
                next_pursued = simulate(self).pursued
                next_pursuer = simulate(self).pursuer
                child = Node(self.state, next_player, next_pursued, next_pursuer, parent=self)
                self.children.append(child)
                return child
        return None

    def rollout(self, root, max_depth=5, min_depth=3) -> float:  # DEPTH = 1 FOR TESTING ONLY
        # We simulate the game X number of turns from the current node or until we win or lose
        # We track values from these simulations so that when we backpropagate we can determine the best route

        # Longer rollouts increase simulation time and decrease value of unique paths
        scaled_depth = int(min(max_depth, max(min_depth, (
                heuristic(self.player, self.pursued) + heuristic(self.player, self.pursuer)) // 2)))

        simulation = self

        for step in range(scaled_depth):
            if simulation.player[0] == simulation.pursued[0] and simulation.player[1] == simulation.pursued[1]:
                return 100 - step
            if simulation.player[0] == simulation.pursuer[0] and simulation.player[1] == simulation.pursuer[1]:
                return -100 + step

            simulation = simulate(simulation)

        # smooth reward
        dist_pursued = heuristic(simulation.player, simulation.pursued)
        dist_pursuer = heuristic(simulation.player, simulation.pursuer)
        reward = 100 * math.exp(-dist_pursued / 6)
        risk = -50 / ((dist_pursuer + 1) ** 0.5)
        crash_risk = -100 if simulation.state[simulation.player[0], simulation.player[1]] == 1 else 0

        return reward + risk + crash_risk

# Monte Carlo Tree Search
def mcts(agent, root):
    simulations = 15  # SIMULATIONS = 1 FOR TESTING ONLY
    prob = get_estimated_probabilities(get_rotation_counts_tuple(agent))
    for it in range(simulations):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(prob)
        if not np.array_equal(node.player, node.pursued):
            expanded = node.expand()
            if expanded: node = expanded
        reward = node.rollout(root)
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    return root.best_child(prob).player - root.player

class PlannerAgent:
    def __init__(self):
        self.root = None
        self.rotation = {"left": 1, "safe": 1, "right": 1}
        self.last_position = None
        self.last_intended_action = None

    def observe_actual_move(self, new_position, weight=1):
        '''
        Combines MLE with Bayesian Estimation and prior known knowledge, called Dirichlet Prior.
        Pi = Ni/sum_j(Nj)
        Pi = Estimated probability for category i
        Ni = Number of times category i was observed
        sum_j(Nj) = The total number of actions observed

        Because we combine Maximum Likelihood Estimation with laplace smoothing, we have a
        Bayesian Estimator with a Dirichlet prior belief.
        This means rather than just using observed frequencies, we assume p ~ Dirichlet (a1, a2, ..., ak)
        where a_k represents the number of potential actions (in our case left, right, or no rotation).
        '''
        if self.last_position is None or self.last_intended_action is None:
            return

        actual = new_position - self.last_position
        intended = self.last_intended_action

        if np.all(intended == 0): return

        left = np.array([-intended[1], intended[0]])
        right = np.array([intended[1], -intended[0]])

        if np.array_equal(actual, intended): self.rotation["safe"] += weight
        elif np.array_equal(actual, left): self.rotation["left"] += weight
        elif np.array_equal(actual, right): self.rotation["right"] += weight

        self.last_position = None
        self.last_intended_action = None

    def plan_action(self, world, current, pursued, pursuer):
        self.observe_actual_move(current)
        dist_to_pursued = heuristic(current, pursued)
        dist_to_pursuer = heuristic(current, pursuer)

        if self.root is None or self.root.state_key != (tuple(current), tuple(pursued), tuple(pursuer)):
            self.root = Node(world, current, pursued, pursuer)

        if dist_to_pursued <= 3 or dist_to_pursuer <= 3:
            action = mcts(self, self.root)
        else:
            action = a_star(world, current, pursued, pursuer, self)

        '''
        CRASH CONDITION
        If the target is not reachable OR
        the target is not within range
        AND the next action is within range of aggressor
        AND an obstacle is within range.
        '''
        crash_location = crash(world, current)
        if (heuristic(current, pursued) > 1 >= heuristic(action - current, pursuer) and
                not np.array_equal(crash_location, np.array([0,0]))):
            return crash_location

        '''
        DODGE CONDITION
        If the aggressor is next to you
        AND the target is not within range
        AND the action is within range of the aggressor
        '''
        if (heuristic(current, pursuer) == 1 and heuristic(current, pursued) > 1 >=
                heuristic(apply_position(action, current), pursuer)):
            return pursuer - current

        '''
        SLIDE CONDITION
        If our player is adjacent to an obstacle
        Given our probabilities and position relative to the obstacle
        Determine best action for moving away from the obstacle
        '''
        if obstacle_penalty(world, current, radius=1, weight=1.0) > 0:
            actions = get_legal_actions(world, current)
            prob = get_estimated_probabilities(get_rotation_counts_tuple(self))
            best_score = -math.inf
            best_action = np.array([0, 0])

            def rotate_left(vec):
                return np.array([-vec[1], vec[0]])

            def rotate_right(vec):
                return np.array([vec[1], -vec[0]])

            for action in actions:
                new_pos = current + action
                if obstacle_penalty(world, new_pos + rotate_left(action), 1) > 0: continue
                if obstacle_penalty(world, new_pos + rotate_right(action), 1) > 0: continue
                score = heuristic(new_pos, pursued) * prob["safe"]
                if score > best_score:
                    best_score = score
                    best_action = action
            self.last_position = current.copy()
            self.last_intended_action = best_action.copy()
            return best_action

        self.last_position = current.copy()
        self.last_intended_action = action.copy()
        return action
