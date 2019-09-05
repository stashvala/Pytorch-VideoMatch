import argparse
import signal
from time import time
from os.path import isdir
from os import mkdir

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim

from videomatch import VideoMatch
from davis import Davis, PairSampler, MultiFrameSampler, collate_pairs, collate_multiframes
from visualize import plot_sequence_result, plot_loss
from preprocess import FrameAugmentor, basic_img_transform, basic_ann_transform
from log import init_logging, logger


def parse_args():
    parser = argparse.ArgumentParser(description="Real time video object segmentation with VideoMatch")

    parser.add_argument("--dataset", '-d', metavar='PATH', default="./DAVIS", type=str,
                        help="Path to DAVIS dataset (default: ./DAVIS)")
    parser.add_argument("--year", '-y', default='2016', choices=['2016', '2017', 'all'], type=str,
                        help="DAVIS challenge year (default: 2016)")
    parser.add_argument("--set", '-t', default='train', choices=['train', 'val', 'trainval'], type=str,
                        help="Construct dataset from DAVIS train, val or all sequences (default: train)")
    parser.add_argument("--sequences", '-q', metavar="SEQ_NAME", nargs='+',
                        help="List of sequence names to include in the dataset. "
                             "Don't use this flag to choose all. (default: None)")
    parser.add_argument("--no-shuffle", '-u', default=False, action='store_true',
                        help="Don't shuffle the dataset (default: False)")

    parser.add_argument("--mode", '-m', default='train', choices=['train', 'eval'], type=str,
                        help="Model train or evaluation/test mode (default: train)")
    parser.add_argument("--cuda_device", '-c', type=int, help="Id of cuda device. Leave empty for CPU (default: CPU)")
    parser.add_argument("--model_save", '-s', metavar='PATH', type=str,
                        help="Path to save trained model (default: None)")
    parser.add_argument("--model_load", '-o', metavar='PATH', type=str,
                        help="Path to load trained model (default: None)")

    parser.add_argument("--batch_size", '-b', default=1, type=int, help="Batch size for eval mode (default: 1)")
    parser.add_argument("--epochs", '-e', default=1, type=int,
                        help="Number of epochs to iterate through whole dataset when training (default: 1)")
    parser.add_argument("--iters", '-i', default=10000, type=int,
                        help="Number of image pairs to iterate through when training. "
                             "Use value -1 to use all dataset pairs (default: 10000)")
    parser.add_argument("--no-augment", '-a', default=False, action='store_true',
                        help="Don't augment frames during training (default: False)")
    parser.add_argument("--learning_rate", '-l', default=1e-5, type=float,
                        help="Learning rate for Adam (default: 0.00001)")
    parser.add_argument("--weight_decay", '-w', default=5e-4, type=float,
                        help="Weight decay for Adam (default: 0.0005)")
    parser.add_argument("--validation_size", default=0.0025, type=float, help="Validation set size (default: 0.0025)")

    parser.add_argument("--input_image_shape", '-p', metavar=('HEIGHT', 'WIDTH'), default=(256, 456), nargs=2, type=int,
                        help="Input image shape (default: 256 456)")
    parser.add_argument("--segmentation_shape", '-g', metavar=('HEIGHT', 'WIDTH'), nargs=2, type=int,
                        help="Segmentation output shape (default: input image size)")

    parser.add_argument("--val_report", '-r', metavar='ITER', default=50, type=int,
                        help="Report validation score on every n-th iteration. Set to -1 to turn it off (default: 50)")
    parser.add_argument("--visualize", '-v', default=False, action='store_true',
                        help="Visualize results in eval mode (default: False)")
    parser.add_argument("--results_dir", '-j',
                        help="Save segmented videosequences to folder. Visualization flag (-v) required!")
    parser.add_argument("--logger", '-f', metavar="LEVEL", choices=['debug', 'info', 'warn', 'fatal'], default='debug',
                        help="Choose logger output level (default: debug)")
    parser.add_argument("--loss_visualization", '-x', default=False, action='store_true',
                        help="Plot binary cross entropy loss when finished training (default: False)")

    return parser.parse_args()


def main():

    parsed_args = parse_args()
    init_logging(parsed_args.logger)

    # dataset related
    davis_dir = parsed_args.dataset
    year = parsed_args.year
    dataset_mode = parsed_args.set
    seq_names = parsed_args.sequences
    shuffle = not parsed_args.no_shuffle

    # model related
    mode = parsed_args.mode
    cuda_dev = parsed_args.cuda_device
    model_save_path = parsed_args.model_save
    model_load_path = parsed_args.model_load

    # training related
    batch_size = parsed_args.batch_size
    epochs = parsed_args.epochs
    iters = parsed_args.iters
    augment = not parsed_args.no_augment
    lr = parsed_args.learning_rate
    weight_decay = parsed_args.weight_decay
    validation_size = parsed_args.validation_size

    # videomatch related
    img_shape = parsed_args.input_image_shape
    seg_shape = parsed_args.segmentation_shape

    # misc
    val_report_iter = parsed_args.val_report
    visualize = parsed_args.visualize
    results_dir = parsed_args.results_dir
    loss_visualize = parsed_args.loss_visualization

    # args checks
    if mode == 'train' and batch_size != 1:
        logger.warning("Batch size > 1 is only applicable to 'eval' mode.")

    if iters != -1 and epochs > 1:
        logger.warning("Iters is set to {} and not to -1 (full dataset), but epoch is > 1".format(iters))

    if mode == 'eval' and shuffle:
        logger.warning("Dataset shuffle can't be set to True in 'eval' mode, setting it to False!")
        shuffle = False
    if mode == 'train' and not shuffle:
        logger.warning("Dataset shuffle is off, consider turning it on when training, "
                       "to avoid overfitting on starting sequences")

    if mode != 'eval' and visualize:
        logger.warning("Visualize is set to True, but mode isn't 'eval'")

    if results_dir is not None and not visualize:
        logger.warning("Visualization has to be enabled to save the results")

    device = None if cuda_dev is None else "cuda:{:d}".format(cuda_dev)

    dataset = Davis(davis_dir, year, dataset_mode, seq_names)

    vm = VideoMatch(out_shape=seg_shape, device=device)
    if model_load_path is not None:
        logger.info("Loading model from path {}".format(model_load_path))
        vm.load_model(model_load_path)

    if mode == 'train':
        pair_sampler = PairSampler(dataset, randomize=shuffle)
        indices = np.arange(len(pair_sampler))
        if shuffle:
            np.random.shuffle(indices)

        split = int(np.floor(validation_size * len(pair_sampler)))
        val_indices = indices[:split]
        train_indices = indices[split:]
        train_loader = DataLoader(dataset, batch_sampler=SubsetRandomSampler(pair_sampler.get_indexes(train_indices)),
                                  collate_fn=collate_pairs)
        val_loader = DataLoader(dataset, batch_sampler=SubsetRandomSampler(pair_sampler.get_indexes(val_indices)),
                                collate_fn=collate_pairs)

        logger.debug("Train set size: {}, Validation set size: {}".format(len(pair_sampler) - split, split))

        iters = len(train_loader) if iters == -1 else iters
        fp = FrameAugmentor(img_shape, augment)

        train_vm(train_loader, val_loader, vm, fp, device, lr, weight_decay, iters, epochs,
                 val_report_iter, model_save_path, loss_visualize)

    elif mode == 'eval':
        multiframe_sampler = MultiFrameSampler(dataset)
        data_loader = DataLoader(dataset, sampler=multiframe_sampler, collate_fn=collate_multiframes,
                                 batch_size=batch_size, num_workers=batch_size)

        if results_dir is not None and not isdir(results_dir):
            mkdir(results_dir)

        eval_vm(data_loader, vm, img_shape, visualize, results_dir)


def train_vm(data_loader, val_loader, vm, fp, device, lr, weight_decay, iters, epochs=1,
             val_report_iter=10, model_save_path=None, loss_visualize=False):

    # set model to train mode
    vm.feat_net.train()

    weight_num = sum(p.numel() for p in vm.feat_net.parameters() if p.requires_grad)
    logger.debug("Number of trainable parameters in VideoMatch: {}".format(weight_num))

    optimizer = optim.Adam(vm.feat_net.parameters(), lr=lr, weight_decay=weight_decay)

    stop_training = False

    # save model on SIGINT (Ctrl + c)
    def sigint_handler(signal, frame):
        logger.info("Ctrl+c caught, stopping the training and saving the model...")
        nonlocal stop_training
        stop_training = True

    signal.signal(signal.SIGINT, sigint_handler)

    # check videomatch avg val accuracy
    vm_avg_val_acc = 0.
    for val_ref_frame, val_test_frame in val_loader:
        (ref_img, ref_mask), (test_img, test_mask) = fp(val_ref_frame, val_test_frame)

        vm.seq_init(ref_img, ref_mask)
        fg_prob, _ = vm.predict_fg_bg(test_img)
        # vm_avg_val_acc += segmentation_accuracy(fg_prob, test_mask.cuda(device))
        vm_avg_val_acc += segmentation_IOU(fg_prob.cpu(), test_mask)

    logger.debug("Untrained Videomatch IOU on validation set: {:.3f}".format(vm_avg_val_acc / len(val_loader)))

    criterion = torch.nn.BCELoss()

    loss_list = []
    val_acc_list = []
    for epoch in range(epochs):
        logger.debug("Epoch: \t[{}/{}]".format(epoch + 1, epochs))

        avg_loss = 0.
        for i, (ref_frame, test_frame) in enumerate(data_loader):
            if i >= iters or stop_training:
                break

            # preprocess
            (ref_img, ref_mask), (test_img, test_mask) = fp(ref_frame, test_frame)
            test_mask = test_mask.unsqueeze(0).cuda(device).float()

            # initialize every time since reference image keeps changing
            vm.seq_init(ref_img, ref_mask)

            # Use softmaxed foreground probability and groundtruth to compute BCE loss
            fg_prob, _ = vm.predict_fg_bg(test_img)

            # loss = balanced_CE_loss(fg_prob, test_mask)
            # loss = criterion(input=fg_prob, target=test_mask)
            loss = dice_loss(fg_prob, test_mask)
            avg_loss += loss.data.mean().cpu().numpy()

            if ((i + 1) % val_report_iter == 0 or i + 1 == iters) and i > 0:
                vm_avg_val_acc = 0.
                val_cnt = 0
                for val_ref_frame, val_test_frame in val_loader:
                    (ref_img, ref_mask), (test_img, test_mask) = fp(val_ref_frame, val_test_frame)

                    vm.seq_init(ref_img, ref_mask)
                    fg_prob, _ = vm.predict_fg_bg(test_img)
                    # vm_avg_val_acc += segmentation_accuracy(fg_prob, test_mask.cuda(device))
                    vm_avg_val_acc += segmentation_IOU(fg_prob.cpu(), test_mask)
                    val_cnt += 1

                logger.debug("Iter [{:5d}/{}]:\tavg loss = {:.4f},\tavg val IOU = {:.3f}"
                             .format(i + 1, iters, avg_loss / val_report_iter, vm_avg_val_acc / val_cnt))

                val_acc_list.append(vm_avg_val_acc / val_cnt)
                loss_list.append(avg_loss / val_report_iter)
                avg_loss = 0.

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if stop_training:
            break

    if model_save_path is not None:
        logger.info("Saving model to path {}".format(model_save_path))
        vm.save_model(model_save_path)

    if loss_visualize:
        if not loss_list:
            logger.info("Loss list is empty, omitting loss visualization!")
        else:
            bins = 0 if len(loss_list) < 500 else 50
            plot_loss(loss_list, val_acc_list, val_report_iter, bins=bins)
            plt.show()


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def balanced_CE_loss(y_pred, y_true, size_average=True):
    assert len(y_pred.shape) == len(y_true.shape)

    mask_size = (y_true.shape[-1] * y_true.shape[-2]) # H * W
    fg_num = torch.sum(y_true).float()
    bg_num = mask_size - fg_num
    fg_share = fg_num / mask_size
    bg_share = bg_num / mask_size

    y_true_neg = -1 * (y_true - 1)

    # tensorflow solution (see https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)
    loss_mat = torch.max(y_pred, 0)[0] - y_pred * y_true + torch.log(1 + torch.exp(-torch.abs(y_pred)))

    loss_fg = torch.sum(torch.mul(y_true, loss_mat))
    loss_bg = torch.sum(torch.mul(y_true_neg, loss_mat))

    final_loss = bg_share * loss_fg + fg_share * loss_bg

    if size_average:
        final_loss /= mask_size

    return final_loss


def eval_vm(data_loader, vm, img_shape, visualize=True, results_dir=None):

    # set model to eval mode
    vm.feat_net.eval()

    # complex loop that calls visualization at the end of each sequence and also handles
    # sequences with number of frames that isn't divisible with batch size
    curr_seq = None
    segm_list = []
    next_seq_buff = []
    for frames in data_loader:
        if next_seq_buff:
            frames = next_seq_buff + frames
            next_seq_buff = []

        if curr_seq != frames[0].seq:
            if curr_seq is not None:
                process_results(curr_seq, segm_list, visualize, results_dir)

            curr_seq = frames[0].seq
            ref_frame = frames[0]
            test_frames = frames[1:]
            segm_list = [np.array(ref_frame.ann)]

            ref_img = basic_img_transform(ref_frame.img, img_shape)
            ref_mask = basic_ann_transform(ref_frame.ann, img_shape)

            vm.seq_init(ref_img, ref_mask)

        # presumes that batch size is smaller than min number of frames in smallest sequence
        else:
            seq_names = [f.seq.name for f in frames]
            comp = [i for i, x in enumerate(seq_names) if curr_seq.name != x]
            # next sequence
            if comp:
                next_seq_buff = frames[comp[0]:]
                test_frames = frames[:comp[0]]
            # same sequence
            else:
                test_frames = frames

        # for batch_size of 1
        if not test_frames:
            continue

        test_imgs = [basic_img_transform(f.img, img_shape) for f in test_frames]
        test_ts = torch.stack(test_imgs)
        vm_out = vm.segment(test_ts)
        segm_list.extend([x.data.cpu().numpy() for x in vm_out.unbind(0)])

    # process for last sequence in dataset
    process_results(curr_seq, segm_list, visualize, results_dir)


def process_results(curr_seq, segm_list, visualize=False, results_dir=None):
    if visualize:
        out_file = None if results_dir is None else "{}/{}.mp4".format(results_dir, curr_seq.name)
        plot_sequence_result(curr_seq, segm_list, out_file=out_file)
    if results_dir is not None:
        save_eval_results(segm_list, "{}/{}".format(results_dir, curr_seq.name))


def save_eval_results(segmentations, sequence_dir, leading_zeros=5, img_format="png"):
    if not isdir(sequence_dir):
        mkdir(sequence_dir)

    for i, seg in enumerate(segmentations):
        seg_name = "{}/{:0{}d}.{}".format(sequence_dir, i, leading_zeros, img_format)
        Image.fromarray(seg).save(seg_name)
        logger.debug("Saved {}".format(seg_name))


def segmentation_accuracy(mask_pred, mask_true, fg_thresh=0.5):
    # check same height and width
    assert mask_pred.shape[-2:] == mask_true.shape[-2:]

    # TODO: should the size of segmentation also affect accuracy?
    comp_arr = (mask_pred >= fg_thresh) == mask_true.byte()
    acc = torch.sum(comp_arr).cpu().numpy() / (comp_arr.shape[-1] * comp_arr.shape[-2])

    return acc


def segmentation_IOU(y_pred, y_true, fg_thresh=0.5):
    mask_pred = y_pred >= fg_thresh
    mask_true = y_true.byte()

    if np.isclose(torch.sum(mask_pred), 0) and np.isclose(torch.sum(mask_true), 0):
        return 1
    else:
        return torch.sum(mask_true & mask_pred) / torch.sum((mask_true | mask_pred)).float()


if __name__ == '__main__':
    main()
