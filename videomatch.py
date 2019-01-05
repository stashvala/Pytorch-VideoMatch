import numpy as np

import torch
from torchvision.models import resnet101
from torch.nn.functional import interpolate, softmax


class VideoMatch:
    def __init__(self, reference_img, mask, k=20, out_shape=None, device=None):

        self.device = "cpu" if device is None else "cuda:{:d}".format(device)
        self.k = k

        self.ref_feat = None
        self.mask_fg = None
        self.mask_bg = None
        self.feat_shape = (0, 0)
        self.out_shape = reference_img.shape if out_shape is None else out_shape

        self.init_vm(reference_img, mask)

        self.features = resnet101(pretrained=True)

    def init_vm(self, reference_img, mask):
        self.ref_feat = self.extract_features((reference_img, ))
        self.feat_shape = self.ref_feat.shape[2:4]

        # added two dimensions to mask since 4D is needed for bilinear interpolation
        self.mask_fg = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        self.mask_bg = (~ self.mask_fg.byte()).float()

        # downsample to ref_feat shape, add dim because sim_mat is 5D
        self.mask_fg = interpolate(self.mask_fg, size=self.feat_shape, mode='bilinear', align_corners=False).unsqueeze(0)
        self.mask_bg = interpolate(self.mask_bg, size=self.feat_shape, mode='bilinear', align_corners=False).unsqueeze(0)

        self.mask_fg, self.mask_bg = self.to_device(self.mask_fg, self.mask_bg)

    def to_device(self, *tensors):
        return tuple(t.cuda(self.device) for t in tensors)

    def extract_features(self, imgs):
        batch = torch.stack([torch.from_numpy(img) for img in imgs])
        batch = self.prepare_tensor(batch)
        return self.features(batch).unbind(0)

    def soft_match(self, test_imgs):

        test_feats = self.extract_features(test_imgs)
        sim_mat = self.cos_similarity(self.ref_feat, test_feats)
        h, w = sim_mat.shape[-2:]

        sim_fg, _ = torch.topk((sim_mat * self.mask_fg).view(-1, h, w, h * w), k=self.k)
        sim_bg, _ = torch.topk((sim_mat * self.mask_bg).view(-1, h, w, h * w), k=self.k)

        sim_fg = torch.mean(sim_fg, dim=-1)
        sim_bg = torch.mean(sim_bg, dim=-1)

        return sim_fg, sim_bg

    def softmax(self, *args):
        stacked = torch.stack(args)
        res = softmax(stacked, dim=-1)
        return res.unbind(0)

    def predict(self, test_imgs):
        sim_fg, sim_bg = self.soft_match(test_imgs)

        # upsample similarity response
        sim_fg = interpolate(sim_fg, size=self.out_shape, mode='bilinear', align_corners=False)
        sim_bg = interpolate(sim_bg, size=self.out_shape, mode='bilinear', align_corners=False)

        probs = softmax(sim_fg, sim_bg)

        return probs

    @staticmethod
    def cos_similarity(X, Y):
        assert (X.shape[0] == 1)
        assert (X.shape[1:] == Y.shape[1:])

        # l2 normalization
        Xnorm = X / torch.norm(X, p=2, dim=-1, keepdim=True)
        Ynorm = Y / torch.norm(Y, p=2, dim=-1, keepdim=True)

        # compute pairwise similarity between all pairs of features
        # return torch.einsum("bijk, bilm -> bjklm", Xnorm, Ynorm) # TODO: should work...

        # fastest alternative to einsum

        _, c, h, w = Xnorm.shape
        Xnorm = Xnorm.squeeze().view(c, h * w)

        cos_sims = []
        for batch in Ynorm:
            batch = batch.view(c, h * w)
            # computes batched outer product
            outer = torch.stack([torch.ger(i, j) for i, j in zip(Xnorm, batch)])
            cs = torch.sum(outer, dim=0).view(h, w, h, w)
            cos_sims.append(cs)

        return torch.stack(cos_sims)

