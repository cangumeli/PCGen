import torch
from math import cos, sin


def rotatex(points, angle):
    if angle == 0:
        return points
    mat = torch.tensor([
        [1, 0, 0],
        [0, cos(angle), -sin(angle)],
        [0, sin(angle), cos(angle)]
    ])
    return torch.matmul(points, mat)


def rotatey(points, angle):
    if angle == 0:
        return points
    mat = torch.tensor([
        [cos(angle), 0, sin(angle)],
        [0, 1, 0],
        [-sin(angle), 0, cos(angle)]
    ])
    return torch.matmul(points, mat)


def rotatez(points, angle):
    if angle == 0:
        return points
    mat = torch.tensor([
        [cos(angle), -sin(angle), 0],
        [sin(angle), cos(angle), 0],
        [0, 0, 1]
    ])
    return torch.matmul(points, mat)


def normalize(points, scale=None):
    points = points - points.mean(-2, keepdim=True)
    points = points / points.abs().max(-2, keepdim=True)[0]
    return points if scale is None else scale * points


def sample(points, faces, num):
    # Taken from https://rusty1s.github.io/pytorch_geometric/build/html/_modules/torch_geometric/transforms/sample_points.html#SamplePoints
    pos_max = points.max()
    points = points / pos_max
    area = (points[faces[1]] - points[faces[0]]).cross(
        points[faces[2]] - points[faces[0]])
    area = area.norm(p=2, dim=1).abs() / 2

    prob = area / area.sum()
    sample = torch.multinomial(prob, num, replacement=True)
    faces = faces[:, sample]

    frac = torch.rand(num, 2)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = points[faces[1]] - points[faces[0]]
    vec2 = points[faces[2]] - points[faces[0]]
    pos_sampled = points[faces[0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2

    return pos_sampled * pos_max


def points_to_grid(points, grid_size, view_distance=0.2):
    '''
    Simple perspective projection + sorting
    Assumes points are normalized to [-1, 1]
    '''
    H, W = grid_size
    assert H*W == points.size(0)
    ortho = points[:, :2]
    depth = points[:, 2] + 1 + view_distance
    persp = ortho / depth.unsqueeze(-1)
    # Sort w.r.t y-axis
    grid = points[persp[:, 1].argsort()].view(H, W, 3)
    # Sort each bucket w.r.t x-axis
    idx = grid[:, :, 0].argsort(-1).unsqueeze(-1).expand_as(grid)
    grid = torch.gather(grid, 1, idx)
    # Make coords channel
    return grid.permute(2, 0, 1)
