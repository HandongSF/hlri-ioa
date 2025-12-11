# Training Utilities

This directory hosts the end-to-end utilities used to create an EE-delta imitation
dataset, train a behavioral cloning (BC) policy, and replay that policy on a recorded
demonstration log for quick debugging/visualization.

The scripts assume you already collected CSV logs with
`isaac_sim/OMY_standalone.py` (or an equivalent logger) and that the resulting
`omy_imitation_log_runXXX.csv` files live under a chosen log directory.

> **Python environment**: the scripts rely on `numpy`, `pandas`, `scikit-learn`,
> `torch`, and `matplotlib`. Install them in the same environment you use for Isaac
> Sim or your training workstation.

---

## 1. `build_ee_dataset.py`

Aggregate many per-run CSV logs into a single supervised dataset.

**One-line default run**

```bash
python training/build_ee_dataset.py
```

**Custom paths example**

```bash
python training/build_ee_dataset.py \
    --log-dir ~/stages/OMY_sim/logs \
    --pattern "omy_imitation_log_run*.csv" \
    --output ~/stages/OMY_sim/ee_imitation_dataset.csv
```

Key options:

- `--log-dir`: folder that contains the individual log CSVs.
- `--pattern`: glob pattern used inside `log-dir` (defaults to
  `omy_imitation_log_run*.csv`).
- `--output`: path for the aggregated dataset (defaults to `<log-dir>/ee_imitation_dataset.csv`).

The output CSV stores `err_*` columns (observation) and `d_ee_*` columns
(actions), plus `episode/step/done` metadata.

---

## 2. `train_bc_policy.py`

Train the BC model from the aggregated dataset.

**One-line default run**

```bash
python training/train_bc_policy.py
```

**Custom hyperparameters example**

```bash
python training/train_bc_policy.py \
    --dataset ~/stages/OMY_sim/ee_imitation_dataset.csv \
    --checkpoint ~/stages/OMY_sim/ee_bc_policy.pt \
    --epochs 50 \
    --batch-size 256 \
    --hidden-dim 128 \
    --lr 1e-3
```

Important arguments:

- `--dataset`: CSV produced by `build_ee_dataset.py`.
- `--checkpoint`: where to save the best-performing model (includes normalization
  stats).
- `--val-ratio`, `--epochs`, `--batch-size`, `--hidden-dim`, `--lr`: training
  hyperparameters.

Outputs include the PyTorch checkpoint and console logs showing epoch-wise train/val
losses.

---

## 3. `test_bc_policy.py`

Load a trained checkpoint, optionally compute one-step RMSE on a specific log, and run
a rollout in EE space while plotting both the demo and policy trajectories in 3D.

**One-line default run**

```bash
python training/test_bc_policy.py
```

**Customized evaluation**

```bash
python training/test_bc_policy.py \
    --csv ~/stages/OMY_sim/logs/omy_imitation_log_run053.csv \
    --checkpoint ~/stages/OMY_sim/ee_bc_policy.pt \
    --max-steps 500 \
    --action-scale 1.0 \
    --pos-tol 0.05 \
    --ang-tol-deg 5.0 \
    --one-step-eval
```

Flags worth tweaking:

- `--csv`: demo log used for analysis/rollout.
- `--checkpoint`: trained model with normalization stats.
- `--max-steps`, `--action-scale`: rollout behavior.
- `--pos-tol`, `--ang-tol-deg`: success thresholds.
- `--one-step-eval`: compute RMSE between predicted deltas and ground truth deltas
  before the rollout.

The script renders a Matplotlib animation comparing the demonstration EE path and the
policy-generated path; close the window to exit.

---

## Dataset & Log Schema

| Column Group | Columns | Description |
|--------------|---------|-------------|
| Observation (`err_*`) | `err_x`, `err_y`, `err_z`, `err_roll`, `err_pitch`, `err_yaw` | Target minus EE pose in position/orientation. |
| Action (`d_ee_*`) | `d_ee_x`, `d_ee_y`, `d_ee_z`, `d_ee_roll`, `d_ee_pitch`, `d_ee_yaw` | EE pose deltas between consecutive timesteps. |
| Metadata | `episode`, `step`, `done` | Episode index, step index, and terminal flag |
| Raw Pose (logs only) | `ee_*`, `target_*` | Absolute EE and target poses used to compute errors. |

The dataset contains observation/action/metadata columns only, while the raw logs keep
`ee_*`/`target_*` pose information for visualization or debugging.

---

## Runtime Integration

The checkpoint produced by `train_bc_policy.py` includes the model weights and
normalization statistics expected by `runtime/policy_runner.py`. Load it there to feed
policy outputs into your robot-specific runtime (ROS, IK servo, etc.) without
retraining. The same checkpoint can also be replayed in sim via `test_bc_policy.py`
for qualitative inspection before deployment.

---

## Recommended Workflow

1. **Collect** logs with the Isaac Sim interactive script.
2. **Aggregate** logs into a dataset via `build_ee_dataset.py`.
3. **Train** the BC policy with `train_bc_policy.py`.
4. **Test/visualize** the resulting checkpoint with `test_bc_policy.py`.

All scripts expose CLI arguments so you can keep this repository public without
hardcoding private directory structures—override the defaults to match your own
project layout.
