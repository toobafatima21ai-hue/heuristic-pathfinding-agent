"""
maze.py
Generates grid mazes with configurable size and obstacle density.
0 = free cell, 1 = obstacle. Start is top-left, goal is bottom-right
unless overridden.
"""

import numpy as np


class Grid:
    def __init__(self, rows: int, cols: int, obstacle_density: float = 0.2,
                 seed: int | None = None, start=None, goal=None):
        """
        rows, cols: grid dimensions
        obstacle_density: fraction of cells (excluding start/goal) that are obstacles (0.0-1.0)
        seed: RNG seed for reproducibility
        """
        if not (0.0 <= obstacle_density < 1.0):
            raise ValueError("obstacle_density must be in [0, 1)")

        self.rows = rows
        self.cols = cols
        self.obstacle_density = obstacle_density
        self.seed = seed
        self.start = start or (0, 0)
        self.goal = goal or (rows - 1, cols - 1)

        self.grid = self._generate()

    def _generate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        grid = (rng.random((self.rows, self.cols)) < self.obstacle_density).astype(int)

        # Guarantee start/goal are always free
        grid[self.start] = 0
        grid[self.goal] = 0

        # Reject mazes where no path can possibly exist (quick connectivity check)
        if not self._is_solvable(grid):
            return self._generate_until_solvable()

        return grid

    def _generate_until_solvable(self, max_attempts: int = 50) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        for attempt in range(max_attempts):
            grid = (rng.random((self.rows, self.cols)) < self.obstacle_density).astype(int)
            grid[self.start] = 0
            grid[self.goal] = 0
            if self._is_solvable(grid):
                return grid
        # Fall back: carve a guaranteed straight path if nothing solvable found
        grid = np.zeros((self.rows, self.cols), dtype=int)
        return grid

    def _is_solvable(self, grid: np.ndarray) -> bool:
        """Simple BFS connectivity check between start and goal."""
        from collections import deque
        visited = np.zeros_like(grid, dtype=bool)
        q = deque([self.start])
        visited[self.start] = True
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if not visited[nr, nc] and grid[nr, nc] == 0:
                        visited[nr, nc] = True
                        q.append((nr, nc))
        return False

    def neighbors(self, node):
        r, c = node
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                yield (nr, nc)

    def is_free(self, node) -> bool:
        r, c = node
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0