import torch
import torch.nn as nn
from collections import namedtuple
from math import pi


MLPConfig = namedtuple(
    'MLPConfig',
    ['input_size', 'hidden_size', 'hidden_layers', 'output_size'])


CNNConfig = namedtuple(
    'CNNConfig', [
        'input_channels', 'hidden_channels', 'filter_size',
        'num_layers', 'output_channels'])
# https://stackoverflow.com/questions/11351032/namedtuple-and-default-values-for-optional-keyword-arguments
CNNConfig.__new__.__defaults__ = (None,) * len(CNNConfig._fields)


def create_mlp(cfg):
    input_size, hidden_size, hidden_layers, output_size = cfg
    layers = []
    for i in range(hidden_layers):
        layers += [
            nn.Linear(input_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True)
        ]
        input_size = hidden_size
    if output_size is not None:
        layers.append(nn.Linear(input_size, output_size))
    return nn.Sequential(*layers)


def create_cnn(cfg, last_activation=True):
    layers = []
    filter_size = (cfg.filter_size,) * 2 if isinstance(cfg.filter_size, int) \
        else cfg.filter_size
    padding = [filter_dim//2 for filter_dim in filter_size]
    for i in range(cfg.num_layers):
        layers += [
            nn.Conv2d(
                cfg.input_channels if i == 0 else cfg.hidden_channels,
                cfg.hidden_channels,
                filter_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(cfg.hidden_channels),
            nn.ReLU(True)
        ]
    if not last_activation:
        layers.pop()
    if cfg.output_channels is not None:
        layers.append(nn.Conv2d(
            cfg.hidden_channels, cfg.output_channels, cfg.filter_size,
            padding=padding))
    return nn.Sequential(*layers)


def normalize_points(points, scale=None):
    centered = points - points.mean(1, True)
    result = centered / centered.abs().max(1, True)[0]
    if scale is not None:
        result = result * scale
    return result


def points_from_sphere(grid_size, radius=0.5, device=None):
    phi = torch.linspace(0, pi, grid_size, device=device).view(-1, 1)
    theta = torch.linspace(0, 2 * pi, grid_size, device=device).view(1, -1)
    # phi, theta = torch.meshgrid([phi, theta])
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi) * torch.ones_like(theta, device=device)
    return torch.stack([x.view(-1), y.view(-1), z.view(-1)], dim=-1)


def normalize_point_grid(points):
    points3d = points.flatten(2)
    centered = points3d - points3d.mean(-1, True)
    normalized = centered / centered.abs().max(-1, True)[0]
    normalized = normalized.view_as(points)
    return normalized


def spatial_map(dimx, dimy, scale=1, device=None, grid_dim=0):
    # Ref: https://github.com/TonythePlaneswalker/pcn/blob/master/models/folding.py
    x = torch.linspace(-scale, scale, dimx, device=device)
    y = x if dimy == dimx else \
        torch.linspace(-scale, scale, dimy, device=device)
    grid = torch.meshgrid([x, y])
    grid = torch.stack(grid, grid_dim)
    return grid


def spatial_grid(dim, scale=1, device=None):
    return spatial_map(dim, dim, scale, device, -1).view(-1, 2)


def vec(x):
    return x.view(x.size(0), -1)


def batch(x):
    return x.view(-1, x.size(-1))


def points_to_grid(points, H, W):
    assert H * W == points.size(-2)
    return points.view(-1, H, W, 3).permute(0, 3, 1, 2)


def grid_to_points(points):
    return points.permute(0, 3, 1, 2).contiguous().view(points.size(0), -1, 3)


def max_pool_select(grid, idx):
    return torch.gather(grid.flatten(2), 2, idx.flatten(2))
