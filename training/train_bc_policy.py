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
"""Train an EE-delta behavioral cloning policy from imitation logs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEFAULT_STAGE_ROOT = Path.home() / "stages" / "OMY_sim"
BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavioral cloning trainer for EE deltas.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default = DEFAULT_STAGE_ROOT / "logs" / "ee_imitation_dataset.csv"
        help="CSV dataset generated from logging (default: ~/stages/OMY_sim/ee_imitation_dataset.csv).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_STAGE_ROOT / "ee_bc_policy.pt",
        help="Where to store the trained checkpoint (default: ~/stages/OMY_sim/ee_bc_policy.pt).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data reserved for validation (default: 0.1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for both loaders (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=HIDDEN_DIM,
        help=f"Hidden dimension of the MLP (default: {HIDDEN_DIM}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Adam learning rate (default: {LR}).",
    )
    return parser.parse_args()


class EEDeltaDataset(Dataset):
    def __init__(
        self,
        df,
        obs_mean=None,
        obs_std=None,
        act_mean=None,
        act_std=None,
        normalize_obs=True,
        normalize_act=True,
    ):
        self.obs_cols = ["err_x", "err_y", "err_z", "err_roll", "err_pitch", "err_yaw"]
        self.act_cols = ["d_ee_x", "d_ee_y", "d_ee_z", "d_ee_roll", "d_ee_pitch", "d_ee_yaw"]

        self.obs = df[self.obs_cols].values.astype(np.float32)
        self.act = df[self.act_cols].values.astype(np.float32)

        if normalize_obs:
            if obs_mean is None or obs_std is None:
                self.obs_mean = self.obs.mean(axis=0, keepdims=True)
                self.obs_std = self.obs.std(axis=0, keepdims=True) + EPS
            else:
                self.obs_mean = obs_mean
                self.obs_std = obs_std

            self.obs = (self.obs - self.obs_mean) / self.obs_std
        else:
            self.obs_mean = None
            self.obs_std = None

        if normalize_act:
            if act_mean is None or act_std is None:
                self.act_mean = self.act.mean(axis=0, keepdims=True)
                self.act_std = self.act.std(axis=0, keepdims=True) + EPS
            else:
                self.act_mean = act_mean
                self.act_std = act_std

            self.act = (self.act - self.act_mean) / self.act_std
        else:
            self.act_mean = None
            self.act_std = None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        x = self.obs[idx]
        y = self.act[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class EEDeltaPolicy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def main():
    args = parse_args()
    dataset_path = args.dataset.expanduser()
    ckpt_path = args.checkpoint.expanduser()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    num_epochs = args.epochs
    hidden_dim = args.hidden_dim
    lr = args.lr

    print(f"[INFO] Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    train_df, val_df = train_test_split(df, test_size=args.val_ratio, random_state=42)

    train_dataset = EEDeltaDataset(train_df, normalize_obs=True, normalize_act=True)
    obs_mean = train_dataset.obs_mean
    obs_std = train_dataset.obs_std
    act_mean = train_dataset.act_mean
    act_std = train_dataset.act_std

    val_dataset = EEDeltaDataset(
        val_df,
        obs_mean=obs_mean,
        obs_std=obs_std,
        act_mean=act_mean,
        act_std=act_std,
        normalize_obs=True,
        normalize_act=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EEDeltaPolicy(obs_dim=6, act_dim=6, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"[INFO] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"[INFO] Using device: {DEVICE}")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(DEVICE)
            act_batch = act_batch.to(DEVICE)

            pred = model(obs_batch)
            loss = criterion(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(DEVICE)
                act_batch = act_batch.to(DEVICE)

                pred = model(obs_batch)
                loss = criterion(pred, act_batch)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "act_mean": act_mean,
                    "act_std": act_std,
                    "config": {
                        "obs_dim": 6,
                        "act_dim": 6,
                        "hidden_dim": hidden_dim,
                        "normalize_obs": True,
                        "normalize_act": True,
                    },
                },
                ckpt_path,
            )
            print(f"  -> [INFO] Best model updated, saved to {ckpt_path}")

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
