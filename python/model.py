import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SEBlock(nn.Module):
    """
    Leela Chess Zero-style residual block with Squeeze-and-Excitation.

    Two 3x3 convs (num_filters -> num_filters), then an SE layer producing
    per-channel scale (Z=sigmoid(W)) and bias (B), applied as (Z * conv_out)+B,
    then residual skip and final ReLU.
    """
    def __init__(self, num_filters: int, se_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.se_fc1 = nn.Linear(num_filters, se_channels)
        self.se_fc2 = nn.Linear(se_channels, 2 * num_filters)
        self.num_filters = num_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        n, c, h, w = out.size()
        pooled = out.mean(dim=(2, 3))
        s = F.relu(self.se_fc1(pooled))
        s = self.se_fc2(s)
        w_gate, b_gate = torch.split(s, self.num_filters, dim=1)
        z = torch.sigmoid(w_gate).view(n, c, 1, 1)
        b = b_gate.view(n, c, 1, 1)
        out = out * z + b

        out = out + identity
        return F.relu(out)


class ChessAIModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        m_cfg = config['model']

        self.num_filters = m_cfg['filters']
        self.board_dim = m_cfg['board_dim']
        se_channels = m_cfg['se_channels']

        self.initial_conv = nn.Conv2d(m_cfg['input_planes'], self.num_filters,
                                      kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(self.num_filters)

        res_blocks = []
        for _ in range(m_cfg['resblocks']):
            res_blocks.append(SEBlock(self.num_filters, se_channels))
        self.residual_blocks = nn.ModuleList(res_blocks)

        # Convolutional policy head (no dense FC). Outputs POLICY_CHANNELS (73)
        # planes of 8x8; these ARE the move logits. A 3x3 conv gives each move
        # its from-square neighbourhood context. Far fewer params than the
        # flatten->FC(->4672) head.
        self.policy_channels = m_cfg['policy_channels']
        self.policy_conv1 = nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(self.num_filters)
        self.policy_conv2 = nn.Conv2d(self.num_filters, self.policy_channels, kernel_size=3, padding=1, bias=True)

        value_channels = m_cfg['value_channels']
        value_fc_hidden = m_cfg['value_fc_hidden']
        self.value_conv = nn.Conv2d(self.num_filters, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * self.board_dim * self.board_dim, value_fc_hidden)
        self.value_fc2 = nn.Linear(value_fc_hidden, 3)

        # Moves-left head (MLH). Predicts normalized plies remaining until game
        # end (plies_left / MLH_SCALE, MLH_SCALE=100 by convention -- must match
        # the C++ target writer and the C++ consumer). Softplus keeps the output
        # non-negative. Auxiliary head: trained at low weight, consumed only by
        # action selection (never by search) as a tiebreak in decided positions.
        mlh_channels = m_cfg['mlh_channels']
        mlh_fc_hidden = m_cfg['mlh_fc_hidden']
        self.mlh_conv = nn.Conv2d(self.num_filters, mlh_channels, kernel_size=1, bias=False)
        self.mlh_bn = nn.BatchNorm2d(mlh_channels)
        self.mlh_fc1 = nn.Linear(mlh_channels * self.board_dim * self.board_dim, mlh_fc_hidden)
        self.mlh_fc2 = nn.Linear(mlh_fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        # Policy: conv head -> (N, 73, 8, 8) -> flatten. C++ index layout is
        # channel-major (channel*64 + row*8 + col; see
        # policy_components_to_flat_index), which matches a plain flatten
        # exactly -- no permute needed.
        p = F.relu(self.policy_bn(self.policy_conv1(x)))
        p = self.policy_conv2(p)                       # (N, 73, 8, 8)
        policy_logits = p.flatten(1)                    # (N, 4672), channel-major

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value_out = F.softmax(self.value_fc2(v), dim=1)

        m = F.relu(self.mlh_bn(self.mlh_conv(x)))
        m = m.flatten(1)
        m = F.relu(self.mlh_fc1(m))
        mlh_out = F.softplus(self.mlh_fc2(m)).squeeze(-1)   # (N,), >= 0, units of MLH_SCALE plies

        return policy_logits, value_out, mlh_out


def fuse_bn_for_export(model: ChessAIModel) -> ChessAIModel:
    def fuse_conv_bn(conv, bn):
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                          stride=conv.stride, padding=conv.padding, bias=True)
        w_conv = conv.weight.clone()
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        std = torch.sqrt(var + eps)
        w_fused = w_conv * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta - gamma * mean / std
        fused.weight.data = w_fused
        fused.bias.data = b_fused
        return fused

    model.initial_conv = fuse_conv_bn(model.initial_conv, model.initial_bn)
    model.initial_bn = nn.Identity()

    for block in model.residual_blocks:
        block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
        block.bn1 = nn.Identity()
        block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
        block.bn2 = nn.Identity()

    model.policy_conv1 = fuse_conv_bn(model.policy_conv1, model.policy_bn)
    model.policy_bn = nn.Identity()
    # policy_conv2 has its own bias and no BN -> nothing to fuse.
    model.value_conv = fuse_conv_bn(model.value_conv, model.value_bn)
    model.value_bn = nn.Identity()
    model.mlh_conv = fuse_conv_bn(model.mlh_conv, model.mlh_bn)
    model.mlh_bn = nn.Identity()

    return model