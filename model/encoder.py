import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):

    def __init__(
            self, model=18, pretrained=True, train_from=None, last_layer=5):
        super().__init__()
        cnn = getattr(models, 'resnet'+str(model))(pretrained=pretrained)
        input = nn.Sequential(
           cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool)
        self.layers = nn.ModuleList([input])
        del cnn.fc
        for i in range(1, 5):
            layer = getattr(cnn, 'layer'+str(i))
            if i <= last_layer:
                self.layers.append(layer)
            else:
                del layer
        self.last_layer = last_layer
        self.train_from = train_from
        self.pretrained = pretrained
        if last_layer == 5:
            self.layers.append(cnn.avgpool)

    def _run_layer(self, idx, x):
        layer = self.layers[idx]
        if not self.pretrained or \
                (self.train_from is not None and self.train_from <= idx):
            x = layer(x)
        else:
            with torch.no_grad():
                layer.eval()
                x = layer(x)
        return x

    def forward(self, x):
        x = self._run_layer(0, x)
        outputs = []
        for i in range(1, len(self.layers)):
            x = self._run_layer(i, x)
            outputs.append(x)
        return outputs


def test():
    x = torch.rand(32, 3, 224, 224)
    encoder = Encoder(last_layer=3)
    y = encoder(x)
    print([e.size() for e in y])
