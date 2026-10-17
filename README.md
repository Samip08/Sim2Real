> ## 🎥 **[Watch the Demo](https://drive.google.com/file/d/1HD4kX3E88NXtezg3EDZHFyh9k9SEE223/view?usp=sharing)**
> **Demo video:** https://drive.google.com/file/d/1HD4kX3E88NXtezg3EDZHFyh9k9SEE223/view?usp=sharing

---

# Sim2Real Robotics Hackathon - Simulation Round

## Repository Structure

```
.
├── src/
│   ├── postman_pkg/                 # Organizer package (unmodified)
│   └── postman_solution/
│       ├── launch/
│       ├── postman_solution/
│       │   ├── __init__.py
│       │   ├── solution.launch.py
│       │   ├── solution_node.py
│       │   ├── tier1_node.py
│       │   ├── tier2_node.py
│       │   └── tier3_node.py
│       ├── resource/
│       ├── package.xml
│       ├── setup.cfg
│       └── setup.py
├── README.md
└── run_seeds.py                     # Batch seed sweep
```

The files evaluated are `solution.launch.py`, `solution_node.py`, `tier1_node.py`, `tier2_node.py` and `tier3_node.py`.

## Status

| Tier | Implementation | Tested |
| :--- | :--- | :--- |
| 1 | Complete | Yes |
| 2 | Complete | Yes |
| 3 | Complete | Yes |

## Intended Run

Download the repository, then from the workspace root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Terminal 1 (simulation):

```bash
ros2 launch postman_pkg sim.launch.py tier:=1 seed:=42
```

Terminal 2 (evaluator):

```bash
python3 src/postman_pkg/scripts/evaluator.py --tier 1
```

Terminal 3 (solution):

```bash
ros2 launch postman_solution solution.launch.py tier:=1
```

Replace `tier:=1` with `2` or `3` for the other tiers. Source ROS 2 and `install/setup.bash` in every terminal.

## Seed Sweep

`run_seeds.py` runs the simulator, evaluator and solution for many seeds in sequence. Run from the workspace root after sourcing:

```bash
python3 run_seeds.py --tier 1
python3 run_seeds.py --tier 2 --start 10 --end 20
python3 run_seeds.py --tier 3 --seeds 42,456,7
```

Results are written to `seed_results_tier<N>.csv`. Per-seed logs are written to `seed_logs/`.