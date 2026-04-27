import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BottleneckBlock(nn.Module):
    """
    1x1 squeeze -> 3x3 spatial -> 1x1 expand with residual skip.
    No SE - global coordination is handled by BroadcastResBlock at intervals.
    """
    def __init__(self, num_channels: int, bottleneck_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, num_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)


class BroadcastResBlock(nn.Module):
    """
    DeepMind-style broadcasting residual block.
    1x1 mix channels -> broadcast (pool to global vector, linear, broadcast back) -> 1x1 mix channels.
    Full residual skip connection.
    """
    def __init__(self, num_channels: int, board_dim: int):
        super().__init__()
        spatial_size = board_dim * board_dim

        # First 1x1 to mix channels
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Broadcast: operates on each channel independently across spatial dims
        self.broadcast_fc = nn.Linear(spatial_size, spatial_size, bias=True)

        # Second 1x1 to mix channels
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape

        # 1x1 channel mix
        out = F.relu(self.bn1(self.conv1(x)))

        # Broadcast: reshape to (B, C, H*W), apply same linear to each channel
        out = out.reshape(b, c, h * w)
        out = self.broadcast_fc(out)
        out = F.relu(out)
        out = out.reshape(b, c, h, w)

        # 1x1 channel mix (no relu before skip)
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)


class ChessAIModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        m_cfg = config['model']
        c_cfg = config['chess']

        self.num_filters = m_cfg['filters']
        self.board_dim = c_cfg['board_dim']

        # --- Initial Representation ---
        self.initial_conv = nn.Conv2d(m_cfg['input_planes'], self.num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(self.num_filters)

        # --- Residual Backbone ---
        res_blocks = []
        broadcast_interval = m_cfg.get('broadcast_interval', 0)
        bottleneck_channels = m_cfg['bottleneck_channels']

        for i in range(m_cfg['resblocks']):
            if broadcast_interval > 0 and (i + 1) % broadcast_interval == 0:
                res_blocks.append(BroadcastResBlock(self.num_filters, self.board_dim))
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

    # Fuse initial conv+bn
    model.initial_conv = fuse_conv_bn(model.initial_conv, model.initial_bn)
    model.initial_bn = nn.Identity()

    # Fuse inside each block
    for block in model.residual_blocks:
        if isinstance(block, BottleneckBlock):
            block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
            block.bn2 = nn.Identity()
            block.conv3 = fuse_conv_bn(block.conv3, block.bn3)
            block.bn3 = nn.Identity()
        elif isinstance(block, BroadcastResBlock):
            block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
            block.bn2 = nn.Identity()

    # Fuse heads
    model.policy_conv = fuse_conv_bn(model.policy_conv, model.policy_bn)
    model.policy_bn = nn.Identity()
    model.value_conv = fuse_conv_bn(model.value_conv, model.value_bn)
    model.value_bn = nn.Identity()

    return model