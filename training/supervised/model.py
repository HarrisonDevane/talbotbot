import torch
import torch.nn as nn
import torch.nn.functional as F

# Define constants (optional, but good for consistency)
# Assume these are available, e.g., imported from utils or defined globally
BOARD_DIM = 8
POLICY_CHANNELS = 73
TOTAL_POLICY_MOVES = BOARD_DIM * BOARD_DIM * POLICY_CHANNELS

class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, dropout_rate_conv: float = 0.0): # Default to 0.0 for internal dropout
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Dropout after the first ReLU in the residual block
        # Using Dropout2d for convolutional layers. It will be nn.Identity() if rate is 0.
        self.dropout_conv = nn.Dropout2d(dropout_rate_conv) if dropout_rate_conv > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # Store the input for the skip connection

        # First convolutional layer, batch norm, ReLU, and then convolutional dropout
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout_conv(out) # Apply Dropout2d here (this is the dropout *within* the res block)

        # Second convolutional layer and batch norm
        out = self.bn2(self.conv2(out))

        # Add the input (identity) to the output of the two convolutional layers
        # This is the core of the residual connection
        out += identity

        # Apply ReLU activation after the addition
        return F.relu(out)


class ChessAIModel(nn.Module):
    def __init__(self, num_input_planes: int, num_residual_blocks: int = 20, num_filters: int = 128, 
                     dropout_rate_conv: float = 0.1, # This rate will be used for blocks 5-20
                     dropout_rate_fc: float = 0.2,
                     dropout_conv_start_block: int = 5): # New parameter: 1-indexed block number to start applying conv dropout
        super().__init__()

        # --- Common Backbone ---
        # Initial convolution to expand channels to num_filters (e.g., 128 or 256)
        self.initial_conv = nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # Store the start block for conv dropout
        # We subtract 1 to convert from 1-indexed (e.g., "start at 5th block") to 0-indexed (for enumerate)
        self.dropout_conv_start_idx = dropout_conv_start_block - 1 if dropout_conv_start_block > 0 else 0

        # Stack multiple Residual Blocks, controlling their internal dropout rate based on their index
        residual_blocks_list = []
        for i in range(num_residual_blocks):
            if i >= self.dropout_conv_start_idx:
                # For blocks from 'dropout_conv_start_block' onwards, apply the specified dropout_rate_conv
                residual_blocks_list.append(ResidualBlock(num_filters, dropout_rate_conv=dropout_rate_conv))
            else:
                # For blocks before 'dropout_conv_start_block', apply 0 dropout (Identity layer)
                residual_blocks_list.append(ResidualBlock(num_filters, dropout_rate_conv=0.0))
        self.residual_blocks = nn.ModuleList(residual_blocks_list)


        # --- Policy Head ---
        # Policy convolution: reduces channels for policy prediction
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False) # 1x1 conv
        self.policy_bn = nn.BatchNorm2d(2)
        
        # Policy FC: flattens and maps to 8*8*73 possible moves
        self.policy_fc = nn.Linear(2 * BOARD_DIM * BOARD_DIM, TOTAL_POLICY_MOVES)
        # Dropout is generally not applied directly before the final output layer of a policy/classification head.

        # --- Value Head ---
        # Value convolution: reduces channels for value prediction
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False) # 1x1 conv
        self.value_bn = nn.BatchNorm2d(1)
        
        # Value FC1: flattens and processes features for value
        self.value_fc1 = nn.Linear(1 * BOARD_DIM * BOARD_DIM, 64)
        # Dropout after the first FC layer in the value head
        # Using standard Dropout for fully connected layers
        self.dropout_fc = nn.Dropout(dropout_rate_fc) if dropout_rate_fc > 0 else nn.Identity()
        
        # Value FC2: outputs a single scalar value
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Common Backbone Forward ---
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Iterate through residual blocks. Dropout is now handled by the individual ResidualBlock instances
        for res_block in self.residual_blocks:
            x = res_block(x)

        # --- Policy Head Forward ---
        policy_x = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_x = policy_x.view(policy_x.size(0), -1) # Flatten for linear layer
        policy_logits = self.policy_fc(policy_x) # Output logits
        
        # --- Value Head Forward ---
        value_x = F.relu(self.value_bn(self.value_conv(x)))
        value_x = value_x.view(value_x.size(0), -1) # Flatten for linear layer
        value_x = F.relu(self.value_fc1(value_x))
        value_x = self.dropout_fc(value_x) # Apply dropout here
        value_output = torch.tanh(self.value_fc2(value_x)) # Output value between -1 and 1

        return policy_logits, value_output # Return logits for policy