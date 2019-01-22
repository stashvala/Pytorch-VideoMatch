import torch
from torch import nn
from torchvision.models import resnet101


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.feat_ext = resnet101(pretrained=True)

    # only takes first two layers from resnet101, output size is 8 times smaller than original
    def forward(self, x):
        x = self.feat_ext.conv1(x)
        x = self.feat_ext.bn1(x)
        x = self.feat_ext.relu(x)
        x = self.feat_ext.maxpool(x)

        x = self.feat_ext.layer1(x)
        x = self.feat_ext.layer2(x)

        return x

    def save_weights(self, path):
        torch.save(self.feat_ext.state_dict(), path)

    def load_weights(self, path):
        torch.load(self.feat_ext.state_dict(), path)

    def weight_MB(self):
        """
        :return: Model size in MegaBytes
        """
        return sum(p.numel() for p in self.feat_ext.parameters() if p.requires_grad) * 32 / 4 / 2**20

