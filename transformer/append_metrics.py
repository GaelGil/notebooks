#!/usr/bin/env python3
import json
import os

txt_file = "./metrics/chckpnts_phase_2_transfer_dropout_weight_decay_0.01.txt"
checkpoint_dir = "./chckpnts_phase_2_transfer_dropout_weight_decay_0.01"

# Get all checkpoint folders after 99
epochs = []
for name in os.listdir(checkpoint_dir):
    path = os.path.join(checkpoint_dir, name)
    if os.path.isdir(path) and name.isdigit() and int(name) > 53:
        epochs.append(int(name))

epochs.sort()
print(f"Found {len(epochs)} checkpoints: {epochs}")

# Read metrics from each folder and append to file
with open(txt_file, "a") as f:
    for epoch in epochs:
        metrics_file = os.path.join(checkpoint_dir, str(epoch), "metrics", "metrics")
        with open(metrics_file, "r") as mf:
            data = json.load(mf)
            train_loss = data["train_loss"]
            eval_loss = data["eval_loss"]
            eval_accuracy = data["eval_accuracy"]
            f.write(f"{epoch},{eval_loss},{eval_accuracy},{train_loss}\n")
