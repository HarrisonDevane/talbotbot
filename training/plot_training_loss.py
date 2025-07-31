import os
import re
import matplotlib.pyplot as plt

current_script_dir = os.path.dirname(os.path.abspath(__file__))
log_folder = os.path.join(current_script_dir, "supervised", "v1_pol_mvplayed_val_result_nsearch", "logs")
print("Reading from:", log_folder)

# Total batches per epoch
batches_per_epoch = 9500

# Regex pattern to extract Epoch, Batch, P_Loss, V_Loss, and T_Loss
pattern = re.compile(r"Epoch (\d+), Batch (\d+): P_Loss=([0-9.]+), V_Loss=([0-9.]+), T_Loss=([0-9.]+)")

all_data = []

# Read logs
for filename in os.listdir(log_folder):
    if filename.endswith(".log"):
        filepath = os.path.join(log_folder, filename)
        with open(filepath, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    batch = int(match.group(2))
                    p_loss = float(match.group(3))
                    v_loss = float(match.group(4))
                    t_loss = float(match.group(5))
                    # Continuous X value: epoch + fractional batch progress
                    epoch_float = epoch + batch / batches_per_epoch
                    all_data.append((epoch_float, p_loss, v_loss, t_loss))

# Sort by continuous epoch
all_data.sort(key=lambda x: x[0])

# Prepare for plotting
x = [entry[0] for entry in all_data]
p_y = [entry[1] for entry in all_data]
v_y = [entry[2] for entry in all_data]
t_y = [entry[3] for entry in all_data]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x, p_y, label="P_Loss", alpha=0.5)
plt.plot(x, v_y, label="V_Loss", alpha=0.5)
plt.plot(x, t_y, label="T_Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_path = os.path.join(log_folder, "training_loss.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

plt.show()