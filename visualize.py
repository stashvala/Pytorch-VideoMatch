import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from log import logger


def plot_fg_bg(ref_img, mask, test_img, fg, bg, test_segm, title="", axes=None):
    assert(len(fg.shape) == len(bg.shape) == 2)

    _, axes = plt.subplots(2, 3) if axes is None else axes
    assert(axes.shape == (2, 3))

    # we get fig like this to avoid extra function parameter
    fig = plt.gcf()
    fig.suptitle(title)

    axes[0, 0].imshow(ref_img)
    axes[0, 0].set_title("Reference image")

    axes[0, 1].imshow(mask, cmap=plt.get_cmap('binary_r'))
    axes[0, 1].set_title("Reference mask")

    axes[0, 2].imshow(test_img)
    axes[0, 2].set_title("Test image")

    axes[1, 0].imshow(fg)
    axes[1, 0].set_title("Foreground")

    axes[1, 1].imshow(bg)
    axes[1, 1].set_title("Background")

    blended = blend_img_segmentation(test_img, test_segm)
    axes[1, 2].imshow(blended)
    axes[1, 2].set_title("Result")

    # turn off axis labels
    [x.axis('off') for x in axes.ravel()]

    return axes


def blend_img_segmentation(img, seg, color='r', alpha=0.4):
    assert(type(seg) == np.ndarray and len(seg.shape) == 2)

    if img.shape != seg.shape:
        logger.warning("Image {} and segmentation {} are not of the same shape, resizing!".format(img.shape, seg.shape))
        img = np.array(Image.fromarray(img).resize(seg.shape[::-1], resample=Image.BILINEAR))

    c = (np.array(mcolors.to_rgb(color)) * 255).astype(np.uint8)
    seg3d = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
    segcolor = seg3d * c

    ref_img_pil = Image.fromarray(img).convert("RGBA")
    segcolor_pil = Image.fromarray(segcolor).convert("RGBA")
    blended = Image.blend(ref_img_pil, segcolor_pil, alpha=alpha)

    return np.array(blended)


def plot_sequence_result(seq, segmentations, ax=None):
    assert ((len(seq) - 1) == len(segmentations))

    ax = plt.gca() if ax is None else ax
    ax.set_title(seq.name)
    ax.set_axis_off()

    ref_img = np.array(Image.open(seq[0][0]))
    ref_mask = np.array(Image.open(seq[0][1]))
    blended = blend_img_segmentation(ref_img, ref_mask)
    artist = ax.imshow(blended)

    # skip first (reference) frame
    for frame, seg in zip(seq[1:], segmentations):
        img = Image.open(frame[0])
        h, w = seg.shape
        if img.size != (w, h):  # PIL flips h and w in .size
            img = img.resize((w, h), Image.ANTIALIAS)

        blended = blend_img_segmentation(np.array(img), seg)
        artist.set_data(blended)

        plt.pause(0.0001)
        plt.draw()


def plot_dilation(mask_orig, mask_dil, title="", axes=None):
    assert(mask_orig.shape == mask_dil.shape)

    _, axes = plt.subplots(1, 2) if axes is None else axes
    assert(len(axes) == 2)

    fig = plt.gcf()
    fig.suptitle(title)

    axes[0].imshow(mask_orig)
    axes[0].set_title("Original")

    axes[1].imshow(mask_dil)
    axes[1].set_title("Dilated")

    return axes
