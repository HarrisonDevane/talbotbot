import torch
import os
import sys

# Adjust sys.path to ensure src_shared is found
# This assumes the script is run from a directory relative to your 'talbot' project root
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming 'src_shared' is two directories up and then into 'src_shared'
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

# Import your ChessAIModel class
# Ensure this path is correct for your project structure
from src_shared.model import ChessAIModel 

def create_and_save_new_model(
    output_path: str = "/Users/User/Projects/talbot/training/reinforcement/best_models/initial_random_model.pth",
    input_planes: int = 68,
    filters: int = 128,
    resblocks: int = 20,
    dropout_rate_conv: float = 0.1,
    dropout_rate_fc: float = 0.25,
    dropout_conv_start_block: int = 10
):
    """
    Creates a new ChessAIModel with random weights and saves its state_dict.

    Args:
        output_path (str): The full path including filename where the model will be saved.
        input_planes (int): Number of input planes for the ChessAIModel.
        filters (int): Number of filters for convolutional layers.
        resblocks (int): Number of residual blocks in the model.
        dropout_rate_conv (float): Dropout rate for convolutional layers.
        dropout_rate_fc (float): Dropout rate for fully connected layers.
        dropout_conv_start_block (int): The residual block index from which dropout is applied.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Instantiate the model. By default, PyTorch models are initialized with random weights.
    model = ChessAIModel(
        num_input_planes=input_planes,
        num_residual_blocks=resblocks,
        num_filters=filters,
        dropout_rate_conv=dropout_rate_conv,
        dropout_rate_fc=dropout_rate_fc,
        dropout_conv_start_block=dropout_conv_start_block
    )

    # Move model to CPU before saving to ensure compatibility across devices
    model.to('cpu') 

    # Save the model's state_dict
    torch.save(model.state_dict(), output_path)

    print(f"New ChessAIModel with random weights created and saved to: {output_path}")

if __name__ == "__main__":
    # You can customize these parameters or pass them from a config file
    # Using parameters from your provided config for consistency
    model_params = {
        "output_path": "/Users/User/Projects/talbot/training/reinforcement/best_models/best_model.pth",
        "input_planes": 68,
        "filters": 128,
        "resblocks": 20,
        "dropout_rate_conv": 0.1,
        "dropout_rate_fc": 0.25,
        "dropout_conv_start_block": 10
    }
    
    create_and_save_new_model(**model_params)