import os
import re
import matplotlib.pyplot as plt

def rolling_average(data, window_size=10):
    """Compute rolling average with given window size."""
    if len(data) < window_size:
        return data  # Not enough data to smooth
    averaged = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        averaged.append(sum(window) / len(window))
    return averaged

current_script_dir = os.path.dirname(os.path.abspath(__file__))
log_folder = os.path.join(current_script_dir, "supervised", "v5_historic_input_channels", "logs")
print("Reading from:", log_folder)

batches_per_epoch = 54700

pattern = re.compile(r"Epoch (\d+), Batch (\d+): P_Loss=([0-9.]+), V_Loss=([0-9.]+), T_Loss=([0-9.]+)")

all_data = []

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
                    epoch_float = epoch + batch / batches_per_epoch
                    all_data.append((epoch_float, p_loss, v_loss, t_loss))

all_data.sort(key=lambda x: x[0])

x = [entry[0] for entry in all_data]
p_y = [entry[1] for entry in all_data]
v_y = [entry[2] for entry in all_data]
t_y = [entry[3] for entry in all_data]


# Apply rolling average smoothing
window_size = 50
p_y_smoothed = rolling_average(p_y, window_size)
v_y_smoothed = rolling_average(v_y, window_size)
t_y_smoothed = rolling_average(t_y, window_size)

x_plot = x[::10]
p_plot = p_y_smoothed[::10]
v_plot = v_y_smoothed[::10]
t_plot = t_y_smoothed[::10]

plt.figure(figsize=(12, 6))
plt.plot(x_plot, p_plot, label="P_Loss", alpha=0.7)
plt.plot(x_plot, v_plot, label="V_Loss", alpha=0.7)
plt.plot(x_plot, t_plot, label="T_Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Smoothed Losses over Epochs")
plt.legend()
plt.grid(True)
plt.ylim(1, 3)
plt.tight_layout()

output_path = os.path.join(log_folder, "training_loss.png")
plt.savefig(output_path)
print(f"Smoothed plot saved to {output_path}")

plt.show()
