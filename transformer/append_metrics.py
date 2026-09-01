import json
import os

txt_file = "./metrics/mini_model_phase_two.csv"
checkpoint_dir = "./chckpnts/mini_model_phase_two/"

# Get all checkpoint folders after n
epochs = []
for name in os.listdir(checkpoint_dir):
    path = os.path.join(checkpoint_dir, name)
    if os.path.isdir(path) and name.isdigit() and int(name) > 58:
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
