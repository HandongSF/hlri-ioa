# Isaac Sim Assets & Scripts

This folder bundles the USD assets and helper script used for the OMY-F3M arm
demonstrations/logging pipeline.

- `OMY.usd`: robot model for the OMY-F3M manipulator.
- `OMY_stage.usd`: scene that spawns the robot arm and a manipulable cube target so
  you can position the arm interactively inside Isaac Sim.
- `OMY_standalone.py`: interactive Isaac Sim app that spawns the stage (or just the
  robot + cube), randomizes targets, subscribes to ROS2 joint commands, and logs
  end-effector/target poses to CSV while toggling logging via a small GUI window.

---

## OMY_standalone.py Usage

Launch from the repository root (or any directory where relative paths resolve) to
spawn Isaac Sim with the GUI enabled:

```bash
python isaac_sim/OMY_standalone.py \
    --stage-root ~/stages/OMY_sim \
    --usd-path ~/stages/OMY_sim/OMY.usd \
    --log-dir ~/stages/OMY_sim/logs \
    --joint-topic /joint_states \
    --run-start-index 0
```

Common arguments:

- `--stage-root`: folder containing `OMY.usd`, `OMY_stage.usd`, and logs. Defaults to
  `~/stages/OMY_sim`.
- `--usd-path`: robot asset to load (defaults to `<stage-root>/OMY.usd`).
- `--log-dir`: CSV destination (defaults to `<stage-root>/logs`).
- `--joint-topic`: ROS2 `sensor_msgs/JointState` topic to subscribe to for commanding
  the arm.
- `--run-start-index`: initial counter for `omy_imitation_log_runXXX.csv`.

Run Isaac Sim interactively, click the “OMY Imitation Logger” window to toggle
logging, and the script will randomize target cube poses, log EE vs target poses at
≈60 Hz, and auto-stop when the EE marker enters the cube.

---

## USD Assets

- Use `OMY_stage.usd` if you want to open the full environment inside Isaac Sim and
  manipulate the cube/robot pose manually.
- Use `OMY.usd` for lightweight scripting (e.g., the standalone script) that loads the
  robot programmatically and spawns its own cube/lighting.

Both USD files can be referenced from other Isaac Sim extensions or scripts using
`add_reference_to_stage` or via the stage manager GUI.

---

## Workflow Overview

1. Open/preview `OMY_stage.usd` when designing the scene or table-top layout.
2. Run `OMY_standalone.py` to collect demonstration logs and interact with the robot +
   cube in real time.
3. Feed the produced CSV logs into the scripts under `training/` to build datasets,
   train the BC policy, and test checkpoints.

Keep this README alongside the assets so collaborators know how the USD files relate to
the data-collection script.***
