import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BroadcastLayer(nn.Module):
    """
    Mixes information within each channel across all spatial dimensions.
    Equivalent to the 'broadcast' function in the Gumbel AlphaZero paper.
    """
    def __init__(self, board_dim: int):
        super().__init__()
        self.hw = board_dim * board_dim
        # Applies an (H*W, H*W) linear transformation. 
        # By PyTorch broadcasting, the identical spatial weights are applied to all channels.
        self.linear = nn.Linear(self.hw, self.hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = x.view(b, c, self.hw)
        x = F.relu(self.linear(x))
        return x.view(b, c, h, w)


class BroadcastBlock(nn.Module):
    """
    Replaces the bottleneck block every N layers.
    1x1 mix channels -> Global Spatial Broadcast -> 1x1 mix channels.
    """
    def __init__(self, num_channels: int, board_dim: int):
        super().__init__()
        # 1x1 Mix Channels
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Spatial Broadcast
        self.broadcast = BroadcastLayer(board_dim)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        # 1x1 Mix Channels
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.broadcast(out)))
        out = self.bn3(self.conv2(out))
        
        out += identity
        return F.relu(out)


class BottleneckBlock(nn.Module):
    """
    DeepMind Gumbel architecture: 1x1 squeeze -> 3x3 spatial -> 3x3 spatial -> 1x1 expand.
    """
    def __init__(self, num_channels: int, bottleneck_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv4 = nn.Conv2d(bottleneck_channels, num_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.bn4(self.conv4(out))
        
        out += identity
        return F.relu(out)


class ChessAIModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        m_cfg = config['model']
        c_cfg = config['chess']

        self.num_filters = m_cfg['filters']
        self.board_dim = c_cfg['board_dim']
        broadcast_interval = m_cfg['broadcast_interval']

        # --- Initial Representation ---
        self.initial_conv = nn.Conv2d(c_cfg['input_planes'], self.num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(self.num_filters)

        # --- Residual Backbone ---
        res_blocks = []
        bottleneck_channels = m_cfg['bottleneck_channels']

        for i in range(m_cfg['resblocks']):
            if broadcast_interval > 0 and (i % broadcast_interval) == (broadcast_interval - 1):
                res_blocks.append(BroadcastBlock(self.num_filters, self.board_dim))
            else:
                res_blocks.append(BottleneckBlock(self.num_filters, bottleneck_channels))
        
        self.residual_blocks = nn.ModuleList(res_blocks)

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(self.num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_dim * self.board_dim, c_cfg['total_policy_moves'])

        # --- Value Head (WDL) ---
        self.value_conv = nn.Conv2d(self.num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_dim * self.board_dim, 64)
        self.value_fc2 = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)
        
        # Value (WDL)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value_out = F.softmax(self.value_fc2(v), dim=1)

        return policy_logits, value_out
    

def fuse_bn_for_export(model: ChessAIModel) -> ChessAIModel:
    def fuse_conv_bn(conv, bn):
        bn_mean = bn.running_mean
        bn_var = bn.running_var
        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_eps = bn.eps

        scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        conv.weight.data *= scale[:, None, None, None]
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype))
        conv.bias.data = (conv.bias.data - bn_mean) * scale + bn_bias
        return conv

    model.initial_conv = fuse_conv_bn(model.initial_conv, model.initial_bn)
    model.initial_bn = nn.Identity()

    for block in model.residual_blocks:
        if isinstance(block, BottleneckBlock):
            block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
            block.bn2 = nn.Identity()
            block.conv3 = fuse_conv_bn(block.conv3, block.bn3)
            block.bn3 = nn.Identity()
            block.conv4 = fuse_conv_bn(block.conv4, block.bn4)
            block.bn4 = nn.Identity()
        elif isinstance(block, BroadcastBlock):
            block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = fuse_conv_bn(block.conv2, block.bn3)
            block.bn3 = nn.Identity()

    model.policy_conv = fuse_conv_bn(model.policy_conv, model.policy_bn)
    model.policy_bn = nn.Identity()
    model.value_conv = fuse_conv_bn(model.value_conv, model.value_bn)
    model.value_bn = nn.Identity()

    return model