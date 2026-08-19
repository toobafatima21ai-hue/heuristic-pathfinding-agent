# 🧭 Heuristic Graph Pathfinding Agent Search Engine

An optimization search framework that lets virtual agents navigate complex grid
mazes efficiently. Implements **A\* Search**, **Dijkstra's Algorithm**, and a
**tabular Q-Learning** agent from scratch, benchmarks them across varying
obstacle densities, and exports visualization logs of the search process.

## Overview

This project builds and compares three pathfinding strategies on procedurally
generated grid mazes:

| Algorithm     | Type                          | Guarantees Optimal Path | Learns From Experience |
|---------------|--------------------------------|:------------------------:|:------------------------:|
| A\* Search    | Informed search (heuristic)    | ✅ (admissible heuristic) | ❌ |
| Dijkstra      | Uninformed / uniform-cost search | ✅                       | ❌ |
| Q-Learning    | Model-free reinforcement learning | ❌ (approximate, trained) | ✅ |

Each agent is evaluated on identical mazes across multiple obstacle densities,
and results are logged with runtime, step count, and node-expansion metrics.

## Features

- 🧱 Procedural maze generator with configurable size and obstacle density,
  with a connectivity check to guarantee a solvable maze.
- 🔍 A\* search using a Manhattan-distance heuristic, built on a binary heap.
- 📉 Dijkstra's algorithm (equivalent to A\* with a zero heuristic).
- 🤖 Tabular Q-Learning agent with epsilon-greedy exploration and decay.
- 📊 Metrics tracked per run: path length, path cost, nodes expanded, runtime.
- 🖼️ Visualization exports: explored-node heatmaps, final path overlays, and
  cross-algorithm comparison charts.
- 📁 CSV/JSON export of every run for further analysis.

## Project Structure

\`\`\`
heuristic-pathfinding-agent/
├── src/
│   ├── maze.py          # Grid generation
│   ├── algorithms.py    # A*, Dijkstra, Q-Learning
│   ├── metrics.py       # Metrics logging + export
│   ├── visualize.py     # Plotting / visualization export
│   └── experiment.py    # Experiment orchestration
├── results/              # Generated logs, CSVs, PNGs
├── main.py
├── requirements.txt
└── README.md
\`\`\`

## Installation

\`\`\`bash
git clone https://github.com/<your-username>/heuristic-pathfinding-agent.git
cd heuristic-pathfinding-agent
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

## Usage

Run the full benchmark suite:

\`\`\`bash
python main.py
\`\`\`

Or run a single experiment programmatically:

\`\`\`python
from src.maze import Grid
from src.algorithms import a_star, dijkstra, q_learning

maze = Grid(rows=20, cols=20, obstacle_density=0.25, seed=42)

result_astar = a_star(maze)
result_dijkstra = dijkstra(maze)
result_qlearning = q_learning(maze, episodes=500)

print(result_astar["path_length"], result_astar["nodes_expanded"], result_astar["runtime_seconds"])
\`\`\`

## Metrics Tracked

For every run, the engine records:

- **Success** — whether a valid path to the goal was found
- **Path length** — number of steps in the final path
- **Path cost** — total traversal cost
- **Nodes expanded** — number of grid cells explored during search
- **Runtime** — wall-clock time in seconds

These are exported to \`results/run_metrics.csv\` and \`results/run_metrics.json\`.

## Visualization Outputs

- Per-run maze snapshots showing obstacles, explored-node density (heatmap),
  and the final path overlay.
- Cross-density comparison line charts for runtime, nodes expanded, and path
  length, one chart per metric, one line per algorithm.

## Algorithms — Design Notes

**A\* Search** uses `f(n) = g(n) + h(n)` where `g(n)` is the cost so far and
`h(n)` is the Manhattan distance to the goal — admissible on a 4-connected
grid with unit step cost, so the path found is optimal.

**Dijkstra's Algorithm** is implemented as a special case of A\* with
`h(n) = 0`, which reduces it to uniform-cost search — it will generally
expand more nodes than A\* since it has no directional guidance toward the
goal.

**Q-Learning** treats each grid cell as a state and the four cardinal moves
as actions. The agent is trained over a configurable number of episodes with
an epsilon-greedy policy (decaying exploration rate), receiving +100 for
reaching the goal, -5 for hitting a wall, and -1 per step otherwise. After
training, the path is extracted by following the greedy (highest-Q) action
from the start state.

## Requirements

\`\`\`
numpy
matplotlib
pandas
\`\`\`

 
