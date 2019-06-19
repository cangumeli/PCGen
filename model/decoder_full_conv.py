import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import \
    spatial_map, CNNConfig, create_cnn, normalize_point_grid, max_pool_select


class ResBlock(nn.Module):

    def __init__(self, cfg, upsample=2):
        super().__init__()
        self.net = create_cnn(cfg, last_activation=False)
        self.upsample = upsample

    def forward(self, x, encoding):
        H, W = x.size()[-2:]
        if self.upsample > 1:
            x = F.interpolate(x, (H * self.upsample, W * self.upsample))
        return F.relu_(self.net(torch.cat([x, encoding], 1)) + x)


class DepthDecoder(nn.Module):

    def __init__(
        self, out_channels=128, encoding_channels=[64, 128], spat_channels=256,
        pred_hidden_channels=128, output_res=(16, 16)
    ):
        super().__init__()
        self.first_block = create_cnn(CNNConfig(
            input_channels=spat_channels, hidden_channels=out_channels,
            filter_size=3, num_layers=2))
        self.blocks = nn.ModuleList()
        for ec in reversed(encoding_channels):
            cfg = CNNConfig(
                input_channels=ec+out_channels,
                hidden_channels=out_channels,
                num_layers=2,
                filter_size=3)
            self.blocks.append(ResBlock(cfg))
        depth_pred_cfg = CNNConfig(
            input_channels=out_channels,
            hidden_channels=pred_hidden_channels,
            num_layers=1, output_channels=1, filter_size=1)
        xy_deform_pred_cfg = CNNConfig(
            input_channels=out_channels*3,
            hidden_channels=pred_hidden_channels,
            num_layers=1, output_channels=2, filter_size=3)
        self.pool = nn.AdaptiveMaxPool2d(output_res, return_indices=True)
        self.depth_pred = create_cnn(depth_pred_cfg)
        self.xy_deform_pred = create_cnn(xy_deform_pred_cfg)

    def forward(self, encodings):
        first_encoding = encodings[-1]
        spat_encodings = encodings[:-1]
        x = self.first_block(first_encoding)
        for block, enc in zip(self.blocks, reversed(spat_encodings)):
            x = block(x, enc)
        # Full resolution smaps
        smaps = spatial_map(*x.size()[-2:], device=x.device)\
            .repeat(x.size(1), 1, 1).expand(x.size(0), -1, -1, -1)
        # predict the depth map
        x, i = self.pool(x)
        depth_map = self.depth_pred(x)
        # Pool the smaps and use them for xy deformation
        i = i.repeat_interleave(2, 1)
        smaps = max_pool_select(smaps, i).view(x.size(0), -1, *x.size()[-2:])
        xy_deform = self.xy_deform_pred(torch.cat([x, smaps], 1))
        xy_grid = spatial_map(*xy_deform.size()[-2:], device=xy_deform.device)
        xy_map = xy_grid + xy_deform  # broadcast
        # return point grid and features
        return torch.cat([xy_map, depth_map], 1), x


class GridDeformer(nn.Module):

    def __init__(self, deform_dims=(1, 4), cfg=CNNConfig(
            input_channels=128 + 3,  # Spat feat + points
            hidden_channels=128, output_channels=3,
            filter_size=(1, 9), num_layers=2)):
        super().__init__()
        self.net = create_cnn(cfg)
        self.deform_dims = deform_dims

    def forward(self, depth_map, encodings, features):
        output_res = (
            depth_map.size(-2) * self.deform_dims[0],
            depth_map.size(-1) * self.deform_dims[1]
        )
        input = torch.cat([depth_map, features], 1)
        folded = F.interpolate(input, output_res)
        deform_grid = self.net(folded)
        deformed = deform_grid + folded[:, :3, :, :]
        return normalize_point_grid(deformed)


def test():
    from .encoder import Encoder
    cnn = Encoder(last_layer=3)
    x = torch.rand(4, 3, 224, 224)
    encs = cnn(x)
    dd = DepthDecoder()
    gd = GridDeformer()
    print(gd)
    coords, features = dd(encs)
    return gd(coords, features)
