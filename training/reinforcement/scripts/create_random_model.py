import os
import sys
import torch
import random
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_root = os.path.abspath(os.path.join(current_script_dir, ".."))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))

sys.path.insert(0, rl_root)
sys.path.insert(0, project_root)

from src_shared.model import ChessAIModel

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Set a fixed random seed for reproducibility
    set_seed(42)

    # Create a new, seeded instance of the model
    model = ChessAIModel(
        num_input_planes=68,
        num_residual_blocks=20,
        num_filters=128,
        dropout_rate_conv=0,
        dropout_rate_fc=0,
        dropout_conv_start_block=0
    )

    print("Successfully created a new instance of the ChessAIModel class.")
    print("The model's initial state is reproducible due to fixed seed.")
    torch.save(model.state_dict(), os.path.join(rl_root, 'rl_cycles', 'initial_model.pth'))