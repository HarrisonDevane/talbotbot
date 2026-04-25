import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GlobalBroadcastingBlock(nn.Module):
    """
    Squeeze-and-Excitation: Pools the board into a global vector 
    to provide long-range coordination.
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
        return x * y

class BasicBlock(nn.Module):
    """
    Standard AlphaZero/Lc0 block: 3x3 -> 3x3 -> SE.
    Maintains full spatial capacity without bottlenecking.
    """
    def __init__(self, num_channels: int, se_reduction_ratio: int = 4):
        super().__init__()
        # First 3x3 Spatial Processing
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Second 3x3 Spatial Processing
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

        # Unconditional SE applied to the residual branch
        self.se = GlobalBroadcastingBlock(num_channels, se_reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE seamlessly
        out = self.se(out)
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
        for _ in range(m_cfg['resblocks']):
            res_blocks.append(
                BasicBlock(
                    self.num_filters, 
                    se_reduction_ratio=m_cfg['broadcast_reduction_ratio']
                )
            )
        
        self.residual_blocks = nn.ModuleList(res_blocks)

        # --- Policy Head (Where to move) ---
        self.policy_conv = nn.Conv2d(self.num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_dim * self.board_dim, c_cfg['total_policy_moves'])

        # --- Value Head (Who is winning) ---
        self.value_conv = nn.Conv2d(self.num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_dim * self.board_dim, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value_out = torch.tanh(self.value_fc2(v))

        return policy_logits, value_out
    
def fuse_bn_for_export(model: ChessAIModel) -> ChessAIModel:
    def fuse_conv_bn(conv, bn):
        # Fold BN into conv weights and bias
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

    # Fuse inside each BasicBlock (SE has no BN, so no changes needed there)
    for block in model.residual_blocks:
        if isinstance(block, BasicBlock):
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