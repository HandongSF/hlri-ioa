#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregate IK imitation logs into a single supervised dataset."""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STAGE_ROOT = Path.home() / "stages" / "OMY_sim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan log CSVs (omy_imitation_log_runXXX.csv) and build a single "
            "delta-EE dataset for behavioral cloning."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_STAGE_ROOT / "logs",
        help="Directory containing per-run logs (default: ~/stages/OMY_sim/logs).",
    )
    parser.add_argument(
        "--pattern",
        default="omy_imitation_log_run*.csv",
        help="Glob pattern used inside the log directory (default: omy_imitation_log_run*.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the aggregated dataset (default: <log-dir>/ee_imitation_dataset.csv).",
    )
    return parser.parse_args()


args = parse_args()
LOG_DIR = args.log_dir.expanduser()
PATTERN = args.pattern
OUT_PATH = args.output.expanduser() if args.output else LOG_DIR / "ee_imitation_dataset.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def angle_diff(a2, a1):
    """Return wrapped difference a2 - a1 within [-pi, pi]."""
    d = a2 - a1
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return d


def main():
    pattern = str(LOG_DIR / PATTERN)
    csv_files = sorted(glob(pattern))

    if not csv_files:
        print(f"[WARN] No log files found: {pattern}")
        return

    rows = []

    for ep_id, csv_path in enumerate(csv_files):
        print(f"[INFO] Loading episode {ep_id} from {csv_path}")
        df = pd.read_csv(csv_path)

        N = len(df)
        if N < 2:
            print(f"[WARN]  Skip {csv_path} (rows={N} < 2)")
            continue

        for i in range(N - 1):
            cur = df.iloc[i]
            nxt = df.iloc[i + 1]

            obs = np.array(
                [
                    cur["err_x"],
                    cur["err_y"],
                    cur["err_z"],
                    cur["err_roll"],
                    cur["err_pitch"],
                    cur["err_yaw"],
                ],
                dtype=np.float32,
            )

            dx = nxt["ee_x"] - cur["ee_x"]
            dy = nxt["ee_y"] - cur["ee_y"]
            dz = nxt["ee_z"] - cur["ee_z"]

            droll = angle_diff(nxt["ee_roll"], cur["ee_roll"])
            dpitch = angle_diff(nxt["ee_pitch"], cur["ee_pitch"])
            dyaw = angle_diff(nxt["ee_yaw"], cur["ee_yaw"])

            act = np.array([dx, dy, dz, droll, dpitch, dyaw], dtype=np.float32)

            done = 1 if i == N - 2 else 0

            rows.append(
                {
                    "episode": ep_id,
                    "step": i,
                    "done": int(done),
                    "err_x": float(obs[0]),
                    "err_y": float(obs[1]),
                    "err_z": float(obs[2]),
                    "err_roll": float(obs[3]),
                    "err_pitch": float(obs[4]),
                    "err_yaw": float(obs[5]),
                    "d_ee_x": float(act[0]),
                    "d_ee_y": float(act[1]),
                    "d_ee_z": float(act[2]),
                    "d_ee_roll": float(act[3]),
                    "d_ee_pitch": float(act[4]),
                    "d_ee_yaw": float(act[5]),
                }
            )

    if not rows:
        print("[WARN] No transitions collected, nothing to save.")
        return

    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUT_PATH, index=False)

    print("[INFO] Saved imitation dataset:")
    print(f"       path          = {OUT_PATH}")
    print(f"       num_episodes  = {len(csv_files)}")
    print(f"       num_samples   = {len(rows)}")


if __name__ == "__main__":
    main()
