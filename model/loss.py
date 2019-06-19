import torch
import torch.nn.functional as F
from numpy import prod
from .util import normalize_points, max_pool_select, normalize_point_grid


def chamfer_distance(
        pred, targets, batch_average=True,
        normalize_pred=False, normalize_target=False, sqrt=False):
    '''pred: NxPx3, targets: NxPx3'''
    # (y_ - y)**2 = <y_, y_> - 2 * <y, y_> + <y, y>
    if normalize_pred:
        pred = normalize_points(pred)
    if normalize_target:
        targets = normalize_points(targets)
    dists = (pred**2).sum(-1, keepdim=True) \
        - 2 * pred @ targets.transpose(-2, -1) \
        + (targets**2).sum(-1).unsqueeze(-2)
    # print(dists.min())
    if sqrt:
        dists = dists.abs().sqrt()  # abs is for numbers close to 0
    mdims = {} if batch_average else {'dim': (1, 2)}
    return (dists.min(-2)[0].mean(**mdims) + dists.min(-1)[0].mean(**mdims))


def std_loss(pred):
    dists = (pred**2).sum(-1).sqrt()
    return -dists.std(1).mean()


def grid_chamfer_distance(
        preds, targets, grid_size, stride=None, batch_average=True):
    if stride is None:
        stride = grid_size
    unfoldeds = []
    for grid in (preds, targets):
        unfolded = F.unfold(grid, grid_size, stride=stride)\
            .transpose(2, 1).contiguous().view(grid.size(0), -1, *grid_size)\
            .transpose(2, 1).contiguous().view(-1, prod(grid_size), 3)
        unfoldeds.append(unfolded)
    return chamfer_distance(*unfoldeds, batch_average=batch_average)


def depth_loss(preds, targets, size=None, stride=None, normalize_grid=True):
    if normalize_grid:
        preds = normalize_point_grid(preds)
    if size is None:
        size = (
            targets.size(-2) // preds.size(-2),
            targets.size(-1) // preds.size(-1)
        )
    idx = F.max_pool2d(
        -targets[:, 2:3, :, :], size, stride=stride, return_indices=True)[1]
    idx = idx.expand(-1, 3, -1, -1)
    targets = max_pool_select(targets, idx).view_as(preds)
    return torch.mean((preds - targets)**2)


def test():
    # x = torch.randn(32, 1024, 3)
    # x = normalize_points(x)
    # print(chamfer_distance(x, x))
    x = normalize_point_grid(torch.randn(32, 3, 32, 32))
    y = normalize_point_grid(torch.randn(32, 3, 32, 32))
    print(grid_chamfer_distance(x, y, (1, 4), 2))
