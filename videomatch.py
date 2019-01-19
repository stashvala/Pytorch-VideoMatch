import torch
from torch import nn
from torchvision.models import resnet101
from torch.nn.functional import interpolate, softmax

from utils import l2_normalization


class VideoMatch:
    def __init__(self, ref_t, mask_t, k=20, d=100, out_shape=None, cuda_dev=None):

        self.device = "cpu" if cuda_dev is None else "cuda:{:d}".format(cuda_dev)
        self.k = k

        self.ref_feat = None
        self.mask_fg = None
        self.mask_bg = None
        self.feat_shape = (0, 0)
        self.out_shape = tuple(ref_t.shape[-2:]) if out_shape is None else out_shape

        self.features = self.to_device(Encoder())
        self.init_vm(ref_t, mask_t)

    def init_vm(self, ref_t, mask_t):
        self.ref_feat = self.extract_features(ref_t)
        self.feat_shape = self.ref_feat.shape[2:4]

        # added two dimensions to mask since 4D is needed for bilinear interpolation
        self.mask_fg = mask_t.unsqueeze(0).unsqueeze(0).float()
        self.mask_bg = (~ self.mask_fg.byte()).float()

        # downsample to ref_feat shape, add dim because sim_mat is 5D
        self.mask_fg = interpolate(self.mask_fg, size=self.feat_shape, mode='bilinear', align_corners=False).unsqueeze(0)
        self.mask_bg = interpolate(self.mask_bg, size=self.feat_shape, mode='bilinear', align_corners=False).unsqueeze(0)

        self.mask_fg, self.mask_bg = self.to_device(self.mask_fg, self.mask_bg)

    def to_device(self, *tensors):
        t = tuple(t.cuda(self.device) for t in tensors)
        return t if len(t) > 1 else t[0]

    def extract_features(self, img_t):
        img_t = img_t if len(img_t.shape) == 4 else img_t.unsqueeze(0)
        return self.features(self.to_device(img_t))

    def soft_match(self, test_t):

        test_feats = self.extract_features(test_t)
        sim_mat = self.cos_similarity(self.ref_feat, test_feats)
        h, w = sim_mat.shape[-2:]

        sim_fg, _ = torch.topk((sim_mat * self.mask_fg).view(-1, h, w, h * w), k=self.k)
        sim_bg, _ = torch.topk((sim_mat * self.mask_bg).view(-1, h, w, h * w), k=self.k)

        sim_fg = torch.mean(sim_fg, dim=-1)
        sim_bg = torch.mean(sim_bg, dim=-1)

        return sim_fg, sim_bg

    def predict_fg_bg(self, test_t):
        sim_fg, sim_bg = self.soft_match(test_t)

        # upsample similarity response
        sim_fg = interpolate(sim_fg.unsqueeze(0), size=self.out_shape, mode='bilinear', align_corners=False)
        sim_bg = interpolate(sim_bg.unsqueeze(0), size=self.out_shape, mode='bilinear', align_corners=False)

        fg_prob, bg_prob = self.softmax(sim_fg.squeeze(0), sim_bg.squeeze(0))

        return fg_prob, bg_prob

    def segment(self, test_t):
        fgs, bgs = self.predict_fg_bg(test_t)

        return fgs > bgs

    def outlier_removal(self, prev_fg_pred, curr_fg_pred):
        pass

    @staticmethod
    def cos_similarity(X, Y):
        assert (X.shape[0] == 1)
        assert (X.shape[1:] == Y.shape[1:])

        # normalize along channels
        Xnorm = l2_normalization(X, dim=1)
        Ynorm = l2_normalization(Y, dim=1)

        # compute pairwise similarity between all pairs of features
        return torch.einsum("xijk, bilm -> bjklm", Xnorm, Ynorm)

    @staticmethod
    def softmax(*args):
        stacked = torch.stack(args)
        res = softmax(stacked, dim=-1)
        return res.unbind(0)


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

    def weight_MB(self):
        """
        :return: Model size in MegaBytes
        """
        return sum(p.numel() for p in self.feat_ext.parameters() if p.requires_grad) * 32 / 4 / 2**20


if __name__ == '__main__':
    import sys
    from time import time

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    from utils import preprocess
    from visualize import plot_result

    if len(sys.argv) < 4:
        raise ValueError("Expected at least three arguments: "
                         "path to reference image, path to mask, path to test image(s). "
                         "\nI got: {}".format(sys.argv))

    ref_img = Image.open(sys.argv[1])
    ref_tensor = preprocess(ref_img)

    mask = np.array(Image.open(sys.argv[2]))
    mask_tensor = torch.from_numpy(mask)

    test_imgs = [Image.open(arg) for arg in sys.argv[3:]]
    img_names = ["/".join(arg.split("/")[-2:]) for arg in sys.argv[3:]]
    test_tensors = preprocess(*test_imgs)

    vm = VideoMatch(ref_tensor, mask_tensor, out_shape=ref_img.size[::-1], cuda_dev=0)

    start = time()
    fgs, bgs = vm.predict_fg_bg(test_tensors)
    print("Prediction for {} images took {:.2f} ms".format(len(test_imgs), (time() - start) * 1000))

    for name, test_img, fg, bg in zip(img_names, test_imgs, fgs, bgs):
        plot_result(np.array(ref_img), mask, np.array(test_img), fg.data.cpu().numpy(), bg.data.cpu().numpy(), name)
        plt.show()
