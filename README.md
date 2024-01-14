# JetracerMPC

Simple MPC using forcespro for the jetracer rc car.
Main usage of this repository is the hackathon at CoR 2024.

# Installation
> :warning: **This repository requires embotech forces pro.**

Install using poetry as
```bash
poetry install
```
or pip as
```bash
pip install .
```

# Usage

Create solvers using
```bash
python3 scripts/create_solver.py config/<configuration> solvers/
```

Use solvers using
```bash
python3 examples/jetracer_bicycle_model.py solvers/<solverfolder> goal_x goal_y
```

Good luck.

