from random import uniform, shuffle, random

import numpy as np

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import RandomApply


class FramePreprocessor:

    MAX_CROP_PERCENT = 0.70
    MIN_CROP_PERCENT = 0.95
    MAX_SCALE = 1.15
    MIN_SCALE = 0.85

    def __init__(self, img_shape, augment=True, aug_oneof=False, aug_probs=0.5, custom_transforms=None):
        self.augment = augment
        self.one_of = aug_oneof
        self.probs = aug_probs
        self.img_shape = img_shape
        self.custom_transforms = custom_transforms

        self.aug_transform_list = [self.hflip, self.random_crop, self.random_scale]

        if self.one_of:
            self.aug_transform = transforms.RandomChoice(self.aug_transform_list)
        else:
            self.aug_transform = RandomAugment(self.aug_transform_list, self.probs)

    @staticmethod
    def basic_img_transform(img, img_shape):
        return transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor(),  # normalizes image to 0-1 values
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])(img)

    @staticmethod
    def basic_ann_transform(ann, img_shape):
        return transforms.Compose([
            transforms.Resize(img_shape),
            # convert to tensor manually, since transform.ToTensor() expects values from 0-255
            transforms.Lambda(lambda ann: torch.from_numpy(np.array(ann)))
        ])(ann)

    def hflip(self, frame):
        img, ann = frame
        return TF.hflip(img), TF.hflip(ann)

    def random_crop(self, frame):
        img, ann = frame
        w, h = img.size
        # we don't want to crop images that are smaller than expected size
        if h <= self.img_shape[0] or w <= self.img_shape[1]:
            print("Warning: input image size {} is equal or smaller than expected output size {}, "
                  "skipping random cropping".format((h, w), self.img_shape))
            return img, ann

        # crop to random size (with same aspect ratio) by clamping it between max_crop_percent and 1.0
        min_ratio = min(self.img_shape[0] / h, self.img_shape[1] / w)
        max_crop = max(self.MAX_CROP_PERCENT, min(min_ratio, self.MIN_CROP_PERCENT))  # clamp
        crop_percent = uniform(max_crop, self.MIN_CROP_PERCENT)
        out_size = int(h * crop_percent), int(w * crop_percent)

        crop_params = transforms.RandomCrop.get_params(img, output_size=out_size)

        return TF.crop(img, *crop_params), TF.crop(ann, *crop_params)

    def random_scale(self, frame):
        img, ann = frame
        scale_params = transforms.RandomAffine.get_params((0, 0), None, (self.MIN_SCALE, self.MAX_SCALE), None, img.size)
        return TF.affine(img, *scale_params), TF.affine(ann, *scale_params)

    def __call__(self, *frames):
        ret = []
        for f in frames:
            img_t, ann_t = f
            if self.augment is not None:
                img_t, ann_t = self.aug_transform((img_t, ann_t))

            if self.custom_transforms is not None:
                img_t, ann_t = self.custom_transforms(img_t, ann_t)

            img_t = self.basic_img_transform(img_t, self.img_shape)
            ann_t = self.basic_ann_transform(ann_t, self.img_shape)

            ret.append((img_t, ann_t))

        return ret


class RandomAugment(RandomApply):
    def __init__(self, transf, p=0.5, random_order=True):
        super(RandomAugment, self).__init__(transf, p)
        self.random_order = random_order

    def __call__(self, img):
        if self.random_order:
            shuffle(self.transforms)

        img_a = img
        for t in self.transforms:
            if random() > self.p:
                img_a = t(img_a)

        return img_a


if __name__ == '__main__':
    import sys

    from PIL import Image
    import matplotlib.pyplot as plt

    from visualize import blend_img_segmentation

    if len(sys.argv) < 3:
        raise ValueError("expected at least two arguments, "
                         "path to image and it's annotation! \nI got {}".format(sys.argv))

    ref_img = Image.open(sys.argv[1])
    ref_mask = Image.open(sys.argv[2])
    img_shape = (256, 456)

    fp = FramePreprocessor(img_shape, augment=True)

    while True:
        img_a, mask_a = fp.aug_transform((ref_img, ref_mask))
        blended = blend_img_segmentation(np.array(img_a), np.array(mask_a))
        plt.imshow(blended)
        plt.title("Shape: {}".format(img_a.size[::-1]))
        plt.show()
