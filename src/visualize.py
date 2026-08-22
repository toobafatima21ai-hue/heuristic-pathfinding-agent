"""
visualize.py
Renders grid + path + expanded-node density, and exports comparison charts
across algorithms and obstacle densities.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_maze_run(grid, result: dict, save_path: str, title: str = ""):
    """Draws the maze, obstacles, expanded-node heatmap, and final path."""
    fig, ax = plt.subplots(figsize=(6, 6))

    display = np.zeros((grid.rows, grid.cols))
    for (r, c) in result.get("expanded_nodes", []):
        display[r, c] = 1  # explored

    ax.imshow(display, cmap="Blues", alpha=0.5, origin="upper")

    obstacle_mask = np.ma.masked_where(grid.grid == 0, grid.grid)
    ax.imshow(obstacle_mask, cmap="Greys", vmin=0, vmax=1, origin="upper")

    path = result.get("path", [])
    if path:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, color="red", linewidth=2, label="Final Path")

    sr, sc = grid.start
    gr, gc = grid.goal
    ax.scatter([sc], [sr], c="green", s=120, marker="o", label="Start", zorder=5)
    ax.scatter([gc], [gr], c="gold", s=140, marker="*", label="Goal", zorder=5)

    ax.set_title(title or f"{result['algorithm']} | success={result['success']}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_comparison(df, output_dir: str = "results"):
    """
    df: pandas DataFrame with columns
        [obstacle_density, algorithm, runtime_seconds, nodes_expanded, path_length]
    Produces 3 comparison charts across obstacle density, one per metric.
    """
    metrics = [
        ("runtime_seconds", "Runtime (seconds)"),
        ("nodes_expanded", "Nodes Expanded"),
        ("path_length", "Path Length (steps)"),
    ]
    saved = []

    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for algo in df["algorithm"].unique():
            sub = df[df["algorithm"] == algo].groupby("obstacle_density")[metric].mean()
            ax.plot(sub.index, sub.values, marker="o", label=algo)

        ax.set_xlabel("Obstacle Density")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Obstacle Density")
        ax.legend()
        ax.grid(alpha=0.3)

        path = os.path.join(output_dir, f"comparison_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved