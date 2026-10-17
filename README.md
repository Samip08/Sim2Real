# Sim2Real Robotics Hackathon - Simulation Round Submission

## Status

The autonomy stack for all three tiers is implemented and tested in the Gazebo Harmonic simulation environment.

| Tier | Description | Node | Status |
| :--- | :--- | :--- | :--- |
| 1 | Dead reckoning navigation | `tier1_node.py` | Complete, tested |
| 2 | Perception-guided navigation | `tier2_node.py` | Complete, tested |
| 3 | Cluttered arena and obstacle avoidance | `tier3_node.py` | Complete, tested |

All solution code is contained in `src/postman_solution`. The organizer package `src/postman_pkg` is unmodified.

## Build

Run from the workspace root (the directory containing `src/`):

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Intended Test Procedure

A single run requires three terminals. In each one, source both ROS 2 Jazzy and the workspace overlay first:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

**Terminal 1: simulation**

```bash
ros2 launch postman_pkg sim.launch.py tier:=<1|2|3> seed:=<SEED>
```

**Terminal 2: evaluator**

```bash
python3 src/postman_pkg/scripts/evaluator.py --tier <1|2|3>
```

**Terminal 3: solution**

```bash
ros2 launch postman_solution solution.launch.py tier:=<1|2|3>
```

Start the simulation first, then the evaluator, then the solution. The evaluator reports waypoints collected, collisions, return-to-home status, run duration, and total odometry distance once the run completes.

## Batch Seed Sweep

`run_seeds.py` was written to evaluate a tier across many seeds automatically. For each seed it launches the simulator, evaluator, and solution in sequence, waits for completion or timeout, parses the evaluator output, and then cleans up all processes before the next seed.

Run it from the workspace root in a shell that has already sourced ROS 2 and the workspace overlay:

```bash
python3 run_seeds.py --tier 1                      # seeds 1 to 50 (default)
python3 run_seeds.py --tier 2 --start 10 --end 20  # a seed range
python3 run_seeds.py --tier 3 --seeds 42,456,7     # specific seeds
```

Useful options:

| Option | Default | Purpose |
| :--- | :--- | :--- |
| `--timeout` | 1500 | Wall-clock seconds allowed per seed |
| `--startup` | 90 | Maximum wall-clock seconds to wait for the simulator to spawn |
| `--sim-args` | none | Extra arguments passed to `sim.launch.py` (for example `headless:=true`) |
| `--logdir` | `seed_logs` | Directory for per-seed simulator, evaluator, and solution logs |

### Output

Results are written to a CSV file, `seed_results_tier<N>.csv`, with one row per seed:

| Column | Meaning |
| :--- | :--- |
| `seed` | Seed used for the run |
| `status` | `completed`, `solution_done`, `timeout`, or `process_died` |
| `collected` | Waypoints collected, formatted as `n/7` |
| `collisions` | Number of collisions detected |
| `home` | Whether the robot returned to the start zone (`YES` or `NO`) |
| `home_err_cm` | Final position error from the start position, in centimeters |
| `coll_at` | Location of each collision, as `type (x,y)` |

At the end of the sweep a summary is printed to the console: the number of seeds run, the number of full successes (7/7 waypoints and home return), the number that were also collision-free, the list of failed seeds, and collision locations by seed. Full logs for each run are kept in the log directory.