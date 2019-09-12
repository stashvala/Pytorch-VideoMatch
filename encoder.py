import torch
from torch import nn
from torchvision.models import resnet101, vgg19
from torch.nn.functional import interpolate

from log import logger


class Encoder(nn.Module):

    ENCODER_TYPES = {"resnet": 0, "vgg": 1}

    def __init__(self, encoder_type="vgg", upsample_fac=1.8):
        super(Encoder, self).__init__()

        if encoder_type not in self.ENCODER_TYPES.keys():
            raise ValueError("Encoder type {} uknown, should be one of the following: {}"
                             .format(encoder_type, self.ENCODER_TYPES))

        self.upsample_fac = upsample_fac
        self.encoder_type = self.ENCODER_TYPES[encoder_type]
        if self.encoder_type == 0:
            self.feat_ext = resnet101(pretrained=True)
        else:
            self.feat_ext = self.load_vgg(19)
        logger.info("Using {} as encoder".format(encoder_type))

        # self.freeze_all_layers_except(self.feat_ext.layer2)

    @staticmethod
    def load_vgg(layers):
        """
        Loads VGG with weights using model from pytorch.
        Using only feature part of VGG (without FC layer) and removes last few layers.
        :param layers: number of layers (16 or 19)
        :param use_bn: load VGG with batch normalization
        :return: pretrained VGG
        """
        if layers not in [16, 19]:
            raise ValueError("Vgg{} not implemented!".format(layers))

        if layers == 19:
            vgg = vgg19(pretrained=True)
        else:
            raise ValueError("currently only vgg 19")

        feats = vgg.features

        # remove last layers two maxpools:
        pools = 0
        while pools < 2:
            _, layer = feats._modules.popitem()
            if isinstance(layer, nn.MaxPool2d):
                pools += 1

        # delete another layer, because openpose does so also
        to_delete = 2  # Only Relu and Conv2d if no batch normalization
        for _ in range(to_delete):
            feats._modules.popitem()

        # This is legacy code, we screwed up when reading the article and accidentally returned 11 layers instead of 10
        # Problem is only with models using vgg16. So far we won't delete 11th layer to avoid breaking the backward compatibility
        # with trained models.
        if layers == 19:
            for _ in range(to_delete):
                feats._modules.popitem()

        # return first 10 feature layers of VGG
        return feats

    # only takes first two layers from resnet101, output size is 8 times smaller than original
    def forward(self, x):
        # resnet
        if self.encoder_type == 0:
            x = self.feat_ext.conv1(x)
            x = self.feat_ext.bn1(x)
            x = self.feat_ext.relu(x)
            x = self.feat_ext.maxpool(x)

            x = self.feat_ext.layer1(x)
            x = self.feat_ext.layer2(x)
        # vgg
        else:
            x = self.feat_ext(x)

        # TODO: could also use CNN upsampling (bilinear + convolution)
        # upsample since we multiply element-wise features with reference mask if the latter is small
        upsample_size = int(x.shape[-2] * self.upsample_fac), int(x.shape[-1] * self.upsample_fac)
        x = interpolate(x, size=upsample_size, mode='bilinear', align_corners=False)

        return x

    def freeze_all_layers_except(self, lay):
        used_layers = self.feat_ext.conv1, self.feat_ext.bn1, self.feat_ext.relu, \
                      self.feat_ext.maxpool, self.feat_ext.layer1, self.feat_ext.layer2
        for l in used_layers:
            if l is lay:
                continue

            for param in l.parameters():
                param.requires_grad = False

    def save_weights(self, path):
        torch.save(self.feat_ext.state_dict(), path)

    def load_weights(self, path):
        self.feat_ext.load_state_dict(torch.load(path))

    def weight_MB(self):
        """
        :return: Model size in MegaBytes
        """
        return sum(p.numel() for p in self.feat_ext.parameters() if p.requires_grad) * 32 / 4 / 2**20

