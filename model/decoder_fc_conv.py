import torch
import torch.nn as nn
from .util import \
    spatial_map, MLPConfig, create_mlp, vec, points_to_grid, grid_to_points


class FCDecoder(nn.Module):

    def __init__(
        self,
        code_gen_cfg=MLPConfig(
            input_size=7*7*512,
            hidden_size=512,
            hidden_layers=1,
            output_size=None
        ),
        output_cfg=MLPConfig(
            input_size=512, hidden_size=512,
            hidden_layers=1, output_size=3*256
        )
    ):
        super().__init__()
        self.code_gen = create_mlp(code_gen_cfg)
        self.pc_gen = create_mlp(output_cfg)
        self.n_points = output_cfg.output_size // 3

    def forward(self, encodings):
        x = vec(encodings[-2])
        code = self.code_gen(x)
        pcs = self.pc_gen(code).view(-1, self.n_points, 3)
        return pcs, code


class DeformBlock(nn.Module):

    def __init__(
        self, input_channels, output_channels,
        factor=1, reflection=True, filter_size=3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                input_channels + 2, output_channels, filter_size,
                padding=filter_size//2, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
            nn.Conv2d(
                output_channels, 3, filter_size,
                padding=filter_size//2
            )
        )
        self.upsample = nn.Upsample(factor, mode='nearest') if factor > 1 \
            else lambda x: x

    def forward(self, x):
        x = self.interpolate(x)
        coords = spatial_map(*x.size()[-2:], device=x.device)\
            .unsqueeze(0).expand(x.size(0), -1, -1, -1)
        x = torch.cat([x, coords], dim=1)
        return self.net(x) + x[:, :3, :, :]


class DeformCNN(nn.Module):

    def __init__(
        self,
        input_channels=[256, 128],
        hidden_channels=128,
        upsamples=[1, 2],
        num_layers=2,
        reflection=True,  # makes input 2^n
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                hidden_channels = 3
            self.convs.add_module(DeformBlock(
                input_channels[i], hidden_channels, factor=upsamples[i]))
            self.pad = nn.ReflectionPad2d(1) if reflection \
                else lambda x: x

    def forward(self, encodings, points, code_vec=None):
        inputs = [self.pad(encodings[-3-i]) for i in range(len(self.convs))]
        points = points_to_grid(points, *inputs[0].size())
        for block, input in zip(self.convs, inputs):
            points = block(torch.cat([points, input], dim=1))
        return grid_to_points(points)
