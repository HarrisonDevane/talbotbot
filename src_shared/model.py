import torch
import torch.nn as nn
import torch.nn.functional as F
import src_shared.utils as u

class BottleneckBlock(nn.Module):
    """
    Squeezes the channel depth, performs the spatial convolution, and expands it back.
    Significantly reduces FLOPs compared to a standard dense block.
    """
    def __init__(self, num_channels: int, bottleneck_channels: int):
        super().__init__()
        # 1x1 Squeeze
        self.conv1 = nn.Conv2d(num_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3x3 Spatial Process
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 1x1 Expand
        self.conv3 = nn.Conv2d(bottleneck_channels, num_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)

class GlobalBroadcastingBlock(nn.Module):
    """
    Squeeze-and-Excitation block. Squashes the entire board state into a single 
    global context vector, then broadcasts it back to every spatial square.
    """
    def __init__(self, num_channels: int, reduction_ratio: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        
        # Broadcast the global context by multiplying it across the spatial dimensions
        return x * y.expand_as(x)

class ChessAIModel(nn.Module):
    def __init__(self, num_input_planes: int, num_residual_blocks: int, num_filters: int, 
                 bottleneck_channels: int, broadcast_reduction_ratio: int, broadcast_interval: int):
        super().__init__()

        # --- Common Backbone ---
        self.initial_conv = nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # Stack Blocks: Bottleneck by default, Global Broadcasting based on interval parameter
        residual_blocks_list = []
        for i in range(num_residual_blocks):
            # Parameterized interval check. If set to 0, no broadcasting blocks are added.
            if broadcast_interval > 0 and (i + 1) % broadcast_interval == 0:
                residual_blocks_list.append(GlobalBroadcastingBlock(num_filters, broadcast_reduction_ratio))
            else:
                residual_blocks_list.append(BottleneckBlock(num_filters, bottleneck_channels))
        
        self.residual_blocks = nn.ModuleList(residual_blocks_list)

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * u.BOARD_DIM * u.BOARD_DIM, u.TOTAL_POLICY_MOVES)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * u.BOARD_DIM * u.BOARD_DIM, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Common Backbone Forward ---
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        for res_block in self.residual_blocks:
            x = res_block(x)

        # --- Policy Head Forward ---
        policy_x = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_x = policy_x.view(policy_x.size(0), -1)
        policy_logits = self.policy_fc(policy_x)
        
        # --- Value Head Forward ---
        value_x = F.relu(self.value_bn(self.value_conv(x)))
        value_x = value_x.view(value_x.size(0), -1)
        value_x = F.relu(self.value_fc1(value_x))
        value_output = torch.tanh(self.value_fc2(value_x))

        return policy_logits, value_output