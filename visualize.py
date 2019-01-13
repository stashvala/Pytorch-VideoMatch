import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def plot_result(ref_img, mask, test_img, fg, bg, name="", axes=None):
    _, axes = plt.subplots(2, 3) if axes is None else axes
    assert(axes.shape == (2, 3))

    # we get fig like this to avoid extra function parameter
    fig = plt.gcf()
    fig.suptitle(name)

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

    plot_segmentation(ref_img, fg, bg, ax=axes[1, 2])
    axes[1, 2].set_title("Result")

    # turn off axis labels
    [x.axis('off') for x in axes.ravel()]

    return axes


def plot_segmentation(ref_img, fg, bg, color='r', alpha=0.4, ax=None):
    ax = plt.gca() if ax is None else ax

    seg = (fg > bg).astype(np.uint8)
    c = (np.array(mcolors.to_rgb(color)) * 255).astype(np.uint8)
    seg3d = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
    segcolor = seg3d * c

    ref_img_pil = Image.fromarray(ref_img).convert("RGBA")
    segcolor_pil = Image.fromarray(segcolor).convert("RGBA")
    blended = Image.blend(ref_img_pil, segcolor_pil, alpha=alpha)

    ax.imshow(np.array(blended))

    return ax
