import torch
import torch.nn as nn
import torch.nn.functional as F

# Define constants (optional, but good for consistency)
# Assume these are available, e.g., imported from utils or defined globally
BOARD_DIM = 8
POLICY_CHANNELS = 73 
TOTAL_POLICY_MOVES = BOARD_DIM * BOARD_DIM * POLICY_CHANNELS

class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # Store the input for the skip connection

        # First convolutional layer, batch norm, and ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Second convolutional layer and batch norm
        out = self.bn2(self.conv2(out))

        # Add the input (identity) to the output of the two convolutional layersl
        # This is the core of the residual connection
        out += identity 
        
        # Apply ReLU activation after the addition
        return F.relu(out)
    


class ChessAIModel(nn.Module):
    def __init__(self, num_input_planes: int, num_residual_blocks: int = 10, num_filters: int = 128):
        super().__init__()

        # --- Common Backbone ---
        # Initial convolution to expand channels to num_filters (e.g., 128 or 256)
        self.initial_conv = nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # Stack multiple Residual Blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # --- Policy Head ---
        # Policy convolution: reduces channels for policy prediction
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False) # 1x1 conv
        self.policy_bn = nn.BatchNorm2d(2)
        # Policy FC: flattens and maps to 8*8*73 possible moves
        self.policy_fc = nn.Linear(2 * BOARD_DIM * BOARD_DIM, TOTAL_POLICY_MOVES)

        # --- Value Head ---
        # Value convolution: reduces channels for value prediction
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False) # 1x1 conv
        self.value_bn = nn.BatchNorm2d(1)
        # Value FC1: flattens and processes features for value
        self.value_fc1 = nn.Linear(1 * BOARD_DIM * BOARD_DIM, 64)
        # Value FC2: outputs a single scalar value
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Common Backbone Forward ---
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        for res_block in self.residual_blocks:
            x = res_block(x)

        # --- Policy Head Forward ---
        policy_x = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_x = policy_x.view(policy_x.size(0), -1) # Flatten for linear layer
        policy_logits = self.policy_fc(policy_x) # Output logits
        
        # Apply log_softmax (often done in the loss function or as last step)
        # policy_output = F.log_softmax(policy_logits, dim=1) 
        # For training with nn.CrossEntropyLoss, output logits directly.
        # For inference, apply F.softmax(policy_logits, dim=1) to get probabilities.

        # --- Value Head Forward ---
        value_x = F.relu(self.value_bn(self.value_conv(x)))
        value_x = value_x.view(value_x.size(0), -1) # Flatten for linear layer
        value_x = F.relu(self.value_fc1(value_x))
        value_output = torch.tanh(self.value_fc2(value_x)) # Output value between -1 and 1

        return policy_logits, value_output # Return logits for policy