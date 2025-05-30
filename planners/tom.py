import numpy as np
from typing import List, Tuple, Optional
import scipy

# Calculate heuristic using the diagonal distance technique
def diagonal_heuristic(x, y, end):
    # H = max(|x - end_x|, |y - end_y|)
    dx = abs(x - end[0])
    dy = abs(y - end[1])
    # return dx + dy - min(dx, dy)
    return max(dx, dy)


def f_value(x, y, end, g):
    # f = g + h
    return diagonal_heuristic(x, y, end) + (g + 1)


def a_star(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    visited = set()
    f = np.inf  # f = g + h, instantiate as inf
    g = 0
    parent = {start: None}

    # Consider all 8 possible moves (up, down, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal moves

    while stack:
        x, y = stack.pop()
        if (x, y) == end:
            # Reconstruct the path
            path = []
            while (x, y) is not None:
                path.append((x, y))
                if parent[(x, y)] is None:
                    break  # Stop at the start node
                x, y = parent[(x, y)]
            return path[::-1]  # Return reversed path
        if (x, y) in visited: continue
        visited.add((x, y))
        succ_x = x  # successor values, temporary
        succ_y = y
        succ_f = f  # successor is chosen based on lowest f value
        for dx, dy in directions:
            nx, ny = dx + x, dy + y
            # make sure successor is valid, not a constraint, and unvisited
            # grid[nx][ny] == 0 checks if the cell is a collision, where a collision = 1
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and (nx, ny) not in visited:
                # check if successor is the goal
                if (nx, ny) == end:
                    # Reconstruct the path
                    path = [(nx, ny)]
                    while (x, y) is not None:
                        path.append((x, y))
                        if parent[(x, y)] is None:
                            break  # Stop at the start node
                        x, y = parent[(x, y)]
                    return path[::-1]  # Return reversed path
                # calculate new f
                ng = g + 1
                nh = diagonal_heuristic(nx, ny, end)
                nf = ng + nh
                if nf < succ_f:
                    succ_f = nf
                    succ_x = nx
                    succ_y = ny
        # add successor node
        stack.append((succ_x, succ_y))
        parent[(succ_x, succ_y)] = (x, y)

    return None  # Return None if no path is found


def plan_path(world: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Computes a path from the start position to the end position 
    using a certain planning algorithm (DFS is provided as an example).

    Parameters:
    - world (np.ndarray): A 2D numpy array representing the grid environment.
      - 0 represents a walkable cell.
      - 1 represents an obstacle.
    - start (Tuple[int, int]): The (row, column) coordinates of the starting position.
    - end (Tuple[int, int]): The (row, column) coordinates of the goal position.

    Returns:
    - np.ndarray: A 2D numpy array where each row is a (row, column) coordinate of the path.
      The path starts at 'start' and ends at 'end'. If no path is found, returns None.
    """
    # Ensure start and end positions are tuples of integers
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))

    # Convert the numpy array to a list of lists for compatibility with the example DFS function
    world_list: List[List[int]] = world.tolist()

    # Perform A* search
    path = a_star(world_list, start, end)

    return np.array(path) if path else None
