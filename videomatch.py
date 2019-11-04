import torch
from torch.nn.functional import interpolate, softmax, conv2d

from encoder import Encoder


class VideoMatch:
    def __init__(self, k=20, d=51, out_shape=None, device=None, encoder="vgg", upsample_fac=1):

        self.device = device
        self.k = k
        self.d = d

        self.ref_feat = None
        self.mask_fg = None
        self.mask_bg = None
        self.dilate_kernel = None
        self.feat_shape = (0, 0)
        self.out_shape = out_shape

        self.feat_net = Encoder(encoder, upsample_fac).cuda(self.device)

    def seq_init(self, ref_t, mask_t):
        assert(len(mask_t.shape) <= 4)

        self.ref_feat = self.extract_features(ref_t)
        self.feat_shape = self.ref_feat.shape[2:4]

        self.out_shape = tuple(ref_t.shape[-2:]) if self.out_shape is None else self.out_shape

        # added dimensions to mask since 4D is needed for bilinear interpolation
        self.mask_fg = mask_t.float()  # cast to float just in case it's byte tensor
        while len(self.mask_fg.shape) < 4:
            self.mask_fg = self.mask_fg.unsqueeze(0)

        # downsample to ref_feat shape, add dim because sim_mat is 5D
        self.mask_fg = interpolate(self.mask_fg, size=self.feat_shape, mode='bilinear', align_corners=False).unsqueeze(0)
        self.mask_bg = -1 * (self.mask_fg - 1)

        self.mask_fg, self.mask_bg = self.to_device(self.mask_fg, self.mask_bg)

        self.dilate_kernel = torch.ones(1, 1, self.d, self.d).cuda(self.device)

    def online_update(self, prev_segm, prev_feat):
        pass

    def to_device(self, *tensors):
        t = tuple(t.cuda(self.device) for t in tensors)
        return t if len(t) > 1 else t[0]

    def extract_features(self, img_t):
        img_t = img_t if len(img_t.shape) == 4 else img_t.unsqueeze(0)
        return self.feat_net(self.to_device(img_t))

    def soft_match(self, test_t):

        test_feats = self.extract_features(test_t)
        sim_mat = self.cos_similarity(test_feats, self.ref_feat)
        # sanity check
        assert(sim_mat.shape[0] == test_feats.shape[0])
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

    def segment(self, test_t, thresh=0.5):
        fgs, bgs = self.predict_fg_bg(test_t)

        return fgs > thresh

    def dilate_mask(self, mask_t):
        assert(self.d % 2 != 0)

        # no need to use mask_t.clone().detach() here
        # mask with 4d shape needed, because conv2d requires NCHW shape
        mask_4d = mask_t
        while len(mask_4d.shape) < 4:
            mask_4d = mask_4d.unsqueeze(0)

        pad = (self.d - 1) // 2

        return conv2d(mask_4d.float(), self.dilate_kernel, padding=pad) > 0.

    def outlier_removal(self, prev_segm, curr_segm):
        assert(prev_segm.shape == curr_segm.shape and len(curr_segm.shape) == 2)
        # TODO: should this be handled elsewhere?
        prev_segm, curr_segm = self.to_device(prev_segm), self.to_device(curr_segm)

        prev_segm_dil = self.dilate_mask(prev_segm)
        # TODO: should the squeeze be left out?
        # extrusion
        return (prev_segm_dil.byte() * curr_segm.byte()).squeeze()

    @staticmethod
    def cos_similarity(X, Y):
        assert (Y.shape[0] == 1)  # TODO: batch cossim for ref
        assert (X.shape[1:] == Y.shape[1:])

        # normalize along channels
        Xnorm = VideoMatch.l2_normalization(X, dim=1)
        Ynorm = VideoMatch.l2_normalization(Y, dim=1)

        # compute pairwise similarity between all pairs of features
        # TODO: Batch is broadcastable dimension so we can use 'x' instead of 'b' for now to avoid an error,
        # see https://github.com/pytorch/pytorch/issues/15671
        return torch.einsum("bijk, xilm -> bjklm", Xnorm, Ynorm)

    @staticmethod
    def softmax(*args):
        stacked = torch.stack(args, dim=1)  # stack along channel dim
        res = softmax(stacked, dim=1)  # compute softmax along channel dim
        return res.unbind(1)

    @staticmethod
    def l2_normalization(X, dim, eps=1e-12):
        return X / (torch.norm(X, p=2, dim=dim, keepdim=True) + eps)

    def save_model(self, path):
        self.feat_net.save_weights(path)

    def load_model(self, path):
        self.feat_net.load_weights(path)


if __name__ == '__main__':
    import sys
    from time import time

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    from preprocess import basic_img_transform, basic_ann_transform
    from visualize import plot_fg_bg

    if len(sys.argv) < 4:
        raise ValueError("Expected at least three arguments: "
                         "path to reference image, path to mask, path to test image(s). "
                         "\nI got: {}".format(sys.argv))

    ref_img = Image.open(sys.argv[1])
    img_shape = ref_img.size[::-1]
    ref_tensor = basic_img_transform(ref_img, img_shape)

    mask = Image.open(sys.argv[2])
    mask_tensor = basic_ann_transform(mask, img_shape)

    test_imgs = [Image.open(arg) for arg in sys.argv[3:]]
    img_names = ["/".join(arg.split("/")[-2:]) for arg in sys.argv[3:]]
    test_tensors = torch.stack([basic_img_transform(t, img_shape) for t in test_imgs])

    vm = VideoMatch(out_shape=img_shape, device="cuda:0")
    vm.seq_init(ref_tensor, mask_tensor)

    start = time()
    fgs, bgs = vm.predict_fg_bg(test_tensors)
    print("Prediction for {} images took {:.2f} ms".format(len(test_imgs), (time() - start) * 1000))

    for name, test_img, fg, bg in zip(img_names, test_imgs, fgs, bgs):
        plot_fg_bg(np.array(ref_img), np.array(mask), np.array(test_img), fg.detach().cpu().numpy(),
                   bg.detach().cpu().numpy(), (fg > bg).cpu().numpy(), name)
        plt.show()
