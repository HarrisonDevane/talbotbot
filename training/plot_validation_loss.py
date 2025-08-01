import os
import re
import matplotlib.pyplot as plt

current_script_dir = os.path.dirname(os.path.abspath(__file__))
log_folder = os.path.join(current_script_dir, "supervised", "v5_historic_input_channels", "logs")

# Regex patterns for train and validation summaries
train_summary_pattern = re.compile(
    r"--- Epoch (\d+) Train Summary ---.*?Average Total Loss: ([0-9.]+)", re.DOTALL
)
val_summary_pattern = re.compile(
    r"--- Epoch (\d+) Validation Summary ---.*?Average Total Loss: ([0-9.]+)", re.DOTALL
)

# Containers for average losses per epoch
train_losses = {}
val_losses = {}

# Read and parse logs
for filename in os.listdir(log_folder):
    if filename.endswith(".log"):
        filepath = os.path.join(log_folder, filename)
        with open(filepath, "r") as f:
            content = f.read()

            # Extract train summaries
            for match in train_summary_pattern.finditer(content):
                epoch = int(match.group(1))
                avg_train_loss = float(match.group(2))
                train_losses[epoch] = avg_train_loss

            # Extract validation summaries
            for match in val_summary_pattern.finditer(content):
                epoch = int(match.group(1))
                avg_val_loss = float(match.group(2))
                val_losses[epoch] = avg_val_loss

# Get all epochs with data
epochs = sorted(set(train_losses.keys()) | set(val_losses.keys()))

# Prepare data for plotting
train_y = [train_losses.get(e, None) for e in epochs]
val_y = [val_losses.get(e, None) for e in epochs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_y, label="Avg Training Total Loss", marker='o')
plt.plot(epochs, val_y, label="Avg Validation Total Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Average Total Loss")
plt.title("Average Training and Validation Total Loss per Epoch")
plt.legend()
plt.grid(True)

output_path = os.path.join(log_folder, "training_vs_validation_loss.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

plt.show()
