"""
experiment.py
Runs A*, Dijkstra, and Q-Learning across mazes of varying obstacle density,
logs metrics, and exports visualizations.
"""

import pandas as pd

from src.maze import Grid
from src.algorithms import a_star, dijkstra, q_learning
from src.metrics import MetricsLogger, RunRecord
from src.visualize import plot_maze_run, plot_comparison


def run_experiments(
    grid_size: tuple[int, int] = (20, 20),
    densities: list[float] = (0.1, 0.2, 0.3, 0.4),
    trials_per_density: int = 3,
    q_learning_episodes: int = 500,
    output_dir: str = "results",
):
    logger = MetricsLogger(output_dir=output_dir)
    rows, cols = grid_size

    for density in densities:
        for trial in range(trials_per_density):
            seed = int(density * 1000) + trial
            maze = Grid(rows, cols, obstacle_density=density, seed=seed)
            maze_id = f"d{density}_t{trial}"

            for algo_name, algo_fn in [
                ("A*", a_star),
                ("Dijkstra", dijkstra),
                ("Q-Learning", lambda g: q_learning(g, episodes=q_learning_episodes, seed=seed)),
            ]:
                result = algo_fn(maze)

                logger.add(RunRecord(
                    maze_id=maze_id, rows=rows, cols=cols,
                    obstacle_density=density, algorithm=algo_name,
                    success=result["success"], path_length=result["path_length"],
                    cost=result["cost"], nodes_expanded=result["nodes_expanded"],
                    runtime_seconds=result["runtime_seconds"],
                ))

                # Save a visual for the first trial of each density only (avoid clutter)
                if trial == 0:
                    fname = f"{output_dir}/maze_{algo_name.replace(' ', '').replace('*','star')}_d{density}.png"
                    plot_maze_run(maze, result, fname,
                                  title=f"{algo_name} | density={density}")

    csv_path = logger.to_csv()
    logger.to_json()

    df = pd.DataFrame([r.__dict__ for r in logger.records])
    plot_comparison(df, output_dir=output_dir)

    return df, csv_path