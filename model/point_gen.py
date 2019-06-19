import torch
import torch.nn as nn
from .decoder_fold import CoarseDecoder, AttentionFold
from .decoder_full_conv import DepthDecoder, GridDeformer
from .encoder import Encoder


class PointGen(nn.Module):

    def __init__(
            self, cnn=None, coarse_decoder=None, fold=None):
        super().__init__()
        self.encoder = cnn if cnn is not None else Encoder()
        self.coarse_decoder = coarse_decoder if coarse_decoder is not None \
            else CoarseDecoder()
        self.fold = fold if fold is not None else AttentionFold()

    def forward(self, img):
        encodings = self.encoder(img)
        points, transform = self.coarse_decoder(encodings)
        return points, self.fold(points, encodings, transform)


def full_conv_default():
    return PointGen(
        cnn=Encoder(last_layer=3),
        coarse_decoder=DepthDecoder(),
        fold=GridDeformer())


def test():
    from timeit import default_timer as timer
    model = PointGen().eval()
    x = torch.rand(1, 3, 224, 224)
    model(x)
    print(
        sum(p.numel() for p in model.parameters()) -
        sum(p.numel() for p in model.encoder.parameters()))
    start = timer()
    y = model(x)
    end = timer()
    print(y.size())
    print(end - start)
