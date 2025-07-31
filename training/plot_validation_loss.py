import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

current_script_dir = os.path.dirname(os.path.abspath(__file__))
log_folder = os.path.join(current_script_dir, "supervised", "v1_pol_mvplayed_val_result_nsearch", "logs")

# Regex patterns
train_pattern = re.compile(
    r"Epoch (\d+), Batch (\d+): .*?T_Loss=([0-9.]+)"
)
val_pattern = re.compile(
    r"--- Epoch (\d+) Val Summary ---.*?Average Total Loss: ([0-9.]+)", re.DOTALL
)

# Data containers
train_losses = defaultdict(list)  # epoch -> list of T_Loss
val_losses = {}

# Read and parse logs
for filename in os.listdir(log_folder):
    if filename.endswith(".log"):
        filepath = os.path.join(log_folder, filename)
        with open(filepath, "r") as f:
            content = f.read()

            # Collect all training losses per epoch
            for match in train_pattern.finditer(content):
                epoch = int(match.group(1))
                t_loss = float(match.group(3))
                train_losses[epoch].append(t_loss)

            # Extract validation summary losses
            val_summaries = re.findall(r"--- Epoch (\d+) Val Summary ---.*?Average Total Loss: ([0-9.]+)", content, re.DOTALL)
            for epoch_str, val_loss_str in val_summaries:
                epoch = int(epoch_str)
                val_loss = float(val_loss_str)
                val_losses[epoch] = val_loss

# Compute average training loss per epoch
avg_train_losses = {epoch: sum(losses)/len(losses) for epoch, losses in train_losses.items()}

# Sort epochs
epochs = sorted(set(avg_train_losses.keys()) | set(val_losses.keys()))

# Prepare data for plotting
train_y = [avg_train_losses.get(e, None) for e in epochs]
val_y = [val_losses.get(e, None) for e in epochs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_y, label="Avg Training T_Loss", marker='o')
plt.plot(epochs, val_y, label="Validation T_Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("T_Loss")
plt.title("Average Training Loss and Validation Loss per Epoch")
plt.legend()
plt.grid(True)

output_path = os.path.join(log_folder, "training_vs_validation_loss.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

plt.show()