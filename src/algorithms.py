"""
algorithms.py
From-scratch implementations of:
  - A* Search (Manhattan-distance heuristic)
  - Dijkstra's Algorithm (uniform-cost search, heuristic = 0)
  - Q-Learning (tabular, epsilon-greedy training + greedy path extraction)

Each function returns a dict with: path, path_length, cost, nodes_expanded,
runtime_seconds, success, expanded_nodes (for visualization).
"""

import heapq
import time
import random
from collections import defaultdict


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(grid, heuristic=manhattan):
    start, goal = grid.start, grid.goal
    t0 = time.perf_counter()

    open_heap = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()
    expanded_nodes = []

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)
        expanded_nodes.append(current)

        if current == goal:
            runtime = time.perf_counter() - t0
            path = _reconstruct_path(came_from, current)
            return {
                "algorithm": "A*",
                "success": True,
                "path": path,
                "path_length": len(path) - 1,
                "cost": g_score[current],
                "nodes_expanded": len(visited),
                "runtime_seconds": runtime,
                "expanded_nodes": expanded_nodes,
            }

        for neighbor in grid.neighbors(current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    runtime = time.perf_counter() - t0
    return {
        "algorithm": "A*", "success": False, "path": [], "path_length": 0,
        "cost": float("inf"), "nodes_expanded": len(visited),
        "runtime_seconds": runtime, "expanded_nodes": expanded_nodes,
    }


def dijkstra(grid):
    # Dijkstra is A* with a zero heuristic (uniform cost search)
    result = a_star(grid, heuristic=lambda a, b: 0)
    result["algorithm"] = "Dijkstra"
    return result


def q_learning(grid, episodes: int = 500, alpha: float = 0.1, gamma: float = 0.95,
                epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                epsilon_decay: float = 0.995, max_steps_per_episode: int = 500,
                seed: int | None = None):
    """
    Tabular Q-Learning agent. Trains on the given grid, then extracts the
    greedy path from start to goal.
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    rng = random.Random(seed)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    start, goal = grid.start, grid.goal

    Q = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    epsilon = epsilon_start
    nodes_touched = set()

    t0 = time.perf_counter()

    for _ in range(episodes):
        state = start
        for _ in range(max_steps_per_episode):
            nodes_touched.add(state)

            if rng.random() < epsilon:
                action = rng.randrange(4)
            else:
                action = max(range(4), key=lambda a: Q[state][a])

            dr, dc = actions[action]
            next_state = (state[0] + dr, state[1] + dc)

            if not grid.is_free(next_state):
                reward = -5.0
                next_state = state  # bounce back / blocked
            elif next_state == goal:
                reward = 100.0
            else:
                reward = -1.0

            best_next = max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

            state = next_state
            if state == goal:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    training_runtime = time.perf_counter() - t0

    # --- Greedy path extraction ---
    t1 = time.perf_counter()
    path = [start]
    state = start
    visited_in_extraction = set()
    max_extract_steps = grid.rows * grid.cols * 2
    success = False

    for _ in range(max_extract_steps):
        if state == goal:
            success = True
            break
        if state in visited_in_extraction:
            break  # stuck in a loop -> extraction failed
        visited_in_extraction.add(state)

        action = max(range(4), key=lambda a: Q[state][a])
        dr, dc = actions[action]
        next_state = (state[0] + dr, state[1] + dc)

        if not grid.is_free(next_state):
            break  # policy leads into a wall -> extraction failed
        state = next_state
        path.append(state)

    extraction_runtime = time.perf_counter() - t1
    total_runtime = training_runtime + extraction_runtime

    return {
        "algorithm": "Q-Learning",
        "success": success,
        "path": path if success else [],
        "path_length": len(path) - 1 if success else 0,
        "cost": (len(path) - 1) if success else float("inf"),
        "nodes_expanded": len(nodes_touched),  # unique states explored during training
        "runtime_seconds": total_runtime,
        "expanded_nodes": list(nodes_touched),
        "episodes": episodes,
    }