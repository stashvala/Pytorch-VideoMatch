import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import animation

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

    if img.shape[:2] != seg.shape:
        logger.warning("Image {} and segmentation {} are not of the same shape, resizing!".format(img.shape, seg.shape))
        img = np.array(Image.fromarray(img).resize(seg.shape[::-1], resample=Image.BILINEAR))

    c = (np.array(mcolors.to_rgb(color)) * 255).astype(np.uint8)
    seg3d = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
    segcolor = seg3d * c

    ref_img_pil = Image.fromarray(img).convert("RGBA")
    segcolor_pil = Image.fromarray(segcolor).convert("RGBA")
    blended = Image.blend(ref_img_pil, segcolor_pil, alpha=alpha)

    return np.array(blended)


def plot_sequence_result(seq, segmentations, out_file=None, fig=None):
    assert len(seq) == len(segmentations)

    fig = plt.figure() if fig is None else fig
    ax = plt.gca()
    ax.set_title(seq.name)
    ax.set_axis_off()

    ref_img = np.array(Image.open(seq[0][0]))
    ref_mask = np.array(Image.open(seq[0][1]))
    blended = blend_img_segmentation(ref_img, ref_mask)
    img_artist = plt.imshow(blended, animated=True)

    def animate(frame):
        nonlocal img_artist
        img, seg = Image.open(frame[0][0]), frame[1]
        h, w = seg.shape
        if img.size != (w, h):  # PIL flips h and w in .size
            img = img.resize((w, h), Image.ANTIALIAS)

        blended = blend_img_segmentation(np.array(img), seg)
        img_artist.set_array(blended)
        return img_artist,

    anim = animation.FuncAnimation(fig, animate, frames=list(zip(seq[1:], segmentations)),
                                   interval=50, blit=True, repeat=False)

    if out_file is not None:
        writer = "imagemagick" if out_file.endswith('.jpg') else "ffmpeg"
        anim.save(out_file, writer=writer)
        logger.debug("Result video sequence saved to {}".format(out_file))

    # TODO: for now you have to close figure manually to continue with the program, find solution!
    plt.show()
    logger.debug("Finished plotting sequence {}".format(seq.name))


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


def plot_loss(loss_list, val_score_list, report_iters, bins=100, clip_max=1.0, ax=None):
    if bins == 0:
        loss_bins, val_bins = loss_list, val_score_list
    else:
        loss_bins, val_bins = [], []
        loss_tmp, val_tmp = [], []
        for i, (l, v) in enumerate(zip(loss_list, val_score_list)):
            loss_tmp.append(l)
            val_tmp.append(v)
            if (i + 1) % bins == 0:
                running_loss_avg = min(sum(loss_tmp) / len(loss_tmp), clip_max)
                running_valacc_avg = min(sum(val_tmp) / len(val_tmp), clip_max)
                loss_bins.append(running_loss_avg)
                val_bins.append(running_valacc_avg)
                loss_tmp, val_tmp = [], []

    ax1 = plt.gca() if ax is None else ax

    ax2 = ax1.twinx()

    x_iters = np.arange(report_iters, (len(loss_bins) + 1) * report_iters, report_iters)
    ax1.plot(x_iters, loss_bins, 'b')
    ax2.plot(x_iters, val_bins, 'r')

    ax1.set_xlabel('Število iteracij')
    ax1.set_ylabel('Vrednost kriterijske funkcije', color='b')
    ax2.set_ylabel('IOU', color='r')

    # save fig because of weird crop
    plot_name = "vm_dice1e-4"
    plt.savefig("{}.png".format(plot_name), bbox_inches='tight')

    return ax
