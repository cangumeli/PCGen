import torch
import torch.nn as nn
from math import sqrt
from .util import \
    spatial_grid, normalize_points, create_mlp, vec, batch, MLPConfig, \
    points_from_sphere

default_course_res = 1024

point_gen_defaut_cfg = MLPConfig(
    input_size=512, hidden_size=512,
    hidden_layers=2, output_size=default_course_res*3)

transform_gen_default_cfg = MLPConfig(
    input_size=14*14*256, hidden_size=512,
    hidden_layers=2, output_size=12*default_course_res)

gate_attn_default_cfg = MLPConfig(
    input_size=512+3+12, hidden_size=128, hidden_layers=1, output_size=128)

spat_attn_default_cfg = MLPConfig(
    input_size=3+12, hidden_size=128, hidden_layers=1, output_size=28*28)

fold_mlp_default_cfg = MLPConfig(
    input_size=128+512+3+2+12, hidden_size=128,
    hidden_layers=2, output_size=3)

spat_attn_default_cfgs = [
    MLPConfig(
        input_size=3+12, hidden_size=128,
        hidden_layers=1, output_size=os[0]*os[1])
    for os in [(14, 14), (28, 28), (56, 56)]
]

gate_attn_default_cfgs = [
    MLPConfig(
        input_size=512+3+12, hidden_size=128, hidden_layers=1, output_size=os)
    for os in [256, 128, 64]
]

fold_mlp_default_cfgs = [
    MLPConfig(
        input_size=f+512+3+2+12, hidden_size=128, hidden_layers=2,
        output_size=3)
    for f in [256, 128, 64]
]


class CoarseDecoder(nn.Module):
    def __init__(
            self,
            model_view_encoding=-3,
            point_gen_cfg=point_gen_defaut_cfg,
            transform_gen_cfg=transform_gen_default_cfg,
            sphere_deform=False):
        super().__init__()
        self.model_view_encoding = model_view_encoding
        self.point_gen = create_mlp(point_gen_cfg)
        self.transform_gen = create_mlp(transform_gen_cfg)
        self.sphere_deform = sphere_deform

    def _transform_points(self, points, transform):
        N = transform.size(0)
        P = points.size(-2)
        w, b = transform.split([9, 3], -1)
        points = points.view(points.size(0), P, 1, 3)
        w = w.view(N, P, 3, 3)
        b = b.view(N, P, 1, 3)
        return (points @ w + b).view(N, P, 3)

    def _deform_sphere(self, points, transform=None):
        # print('Deforming sphere...')
        points = normalize_points(points, scale=0.5)
        sphere = points_from_sphere(
            int(sqrt(points.size(-2))), device=points.device)
        sphere = sphere.unsqueeze(0)
        if transform is not None:
            points = self._transform_points(points, transform)
            sphere = self._transform_points(sphere, transform)
        points = points + sphere
        return points

    def forward(self, encodings):
        spat = vec(encodings[self.model_view_encoding])
        glob = vec(encodings[-1])
        N = spat.size(0)
        points = self.point_gen(glob).view(N, -1, 3)
        '''
        if self.sphere_deform:
            points = self._deform_sphere(points)
        '''
        transform = self.transform_gen(spat).view(N, -1, 12)
        if self.sphere_deform:
            points = self._deform_sphere(points, transform)
        else:
            points = self._transform_points(points, transform)
        return normalize_points(points), transform


class GatedSpatialAttention(nn.Module):

    def __init__(self, spat_attn_cfg, gate_attn_cfg):
        super().__init__()
        self.spat_mlp = create_mlp(spat_attn_cfg)
        self.gate_mlp = create_mlp(gate_attn_cfg)
        self.gate_mlp[-1].bias.data.fill_(1)  # open all the gates by default

    def forward(self, spat_attn_input, gate_attn_input, filters):
        N, C, H, W = filters.size()
        sattn = self.spat_mlp(batch(spat_attn_input)).view(N, -1, 1, H*W)\
            .softmax(-1)
        gattn = self.gate_mlp(batch(gate_attn_input)).view(N, -1, C, 1)\
            .sigmoid()
        spat_map = (filters.view(N, 1, C, H*W) * sattn).sum(-1)
        attn_map = gattn * spat_map.view_as(gattn)
        return attn_map.view(N, -1, C)


class AttentionFold(nn.Module):

    def __init__(
            self,
            spat_attn_cfg=spat_attn_default_cfg,
            gate_attn_cfg=gate_attn_default_cfg,
            mlp_cfg=fold_mlp_default_cfg,
            factor=1, encoding_level=1, upsample=False):
        super().__init__()
        self.attn = GatedSpatialAttention(spat_attn_cfg, gate_attn_cfg)
        self.mlp = create_mlp(mlp_cfg)
        self.factor = factor
        self.encoding_level = encoding_level
        self.upsample = upsample

    def forward(self, points, encodings, transform):
        N, P, _ = points.size()
        globs = encodings[-1].view(N, 1, -1).expand(-1, P, -1)
        coords = spatial_grid(int(sqrt(P * self.factor)), device=globs.device)\
            .unsqueeze(0).expand(N, -1, -1)
        code_vecs = torch.cat([points, transform, globs], -1)
        code_vecs = code_vecs.repeat_interleave(self.factor, dim=1)
        fold_feats = torch.cat([code_vecs, coords], dim=-1)
        spat_attn_input = fold_feats[:, :, :(3+12)]
        gate_attn_input = fold_feats[:, :, :(3+12+globs.size(-1))]
        spat_feats = self.attn(
            spat_attn_input, gate_attn_input, encodings[self.encoding_level])
        inputs = torch.cat([fold_feats, spat_feats], dim=-1)
        deform = self.mlp(batch(inputs)).view(N, -1, 3)
        return normalize_points(deform + inputs[:, :, :3])


class AttentionFoldNet(nn.Module):

    def __init__(
        self,
        fold_cfgs=fold_mlp_default_cfgs,
        spat_cfgs=spat_attn_default_cfgs,
        gate_cfgs=gate_attn_default_cfgs
    ):
        super().__init__()
        self.folds = nn.ModuleList()
        for i, fcfg, scfg, gcfg in zip(
                reversed(range(len(fold_cfgs))), fold_cfgs, 
                spat_cfgs, gate_cfgs):
            self.folds.append(
                AttentionFold(scfg, gcfg, fcfg, encoding_level=i))

    def forward(self, points, encodings, transform):
        for fold in self.folds:
            points = fold(points, encodings, transform)
        return points
