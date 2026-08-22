"""
main.py
Entry point: runs the full experiment suite and prints a summary table.
"""

from src.experiment import run_experiments


def main():
    df, csv_path = run_experiments(
        grid_size=(20, 20),
        densities=[0.1, 0.2, 0.3, 0.4],
        trials_per_density=3,
        q_learning_episodes=500,
        output_dir="results",
    )

    print("\n=== Summary: mean metrics per algorithm per density ===")
    summary = df.groupby(["algorithm", "obstacle_density"]).agg(
        success_rate=("success", "mean"),
        avg_path_length=("path_length", "mean"),
        avg_nodes_expanded=("nodes_expanded", "mean"),
        avg_runtime_seconds=("runtime_seconds", "mean"),
    ).round(4)
    print(summary)

    print(f"\nFull metrics exported to: {csv_path}")
    print("Visualizations exported to: results/")


if __name__ == "__main__":
    main()