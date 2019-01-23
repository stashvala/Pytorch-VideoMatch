import argparse
import signal
from time import time

from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from videomatch import VideoMatch
from davis import Davis, PairSampler, MultiFrameSampler, collate_pairs, collate_multiframes


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
    parser.add_argument("--shuffle", '-u', default=True, action='store_true', help="Shuffle dataset (default: True)")

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
                        help="Number of image pairs to iterate through when training (default: 10000)")
    parser.add_argument("--learning_rate", '-l', default=1e-5, type=float,
                        help="Learning rate for Adam (default: 0.00001)")
    parser.add_argument("--weight_decay", '-w', default=5e-4, type=float,
                        help="Weight decay for Adam (default: 0.0005)")

    parser.add_argument("--input_image_shape", metavar=('HEIGHT', 'WIDTH'), default=(256, 456), nargs=2, type=int,
                        help="Input image shape (default: 256 456)")
    parser.add_argument("--segmentation_shape", '-g', metavar=('HEIGHT', 'WIDTH'), nargs=2, type=int,
                        help="Segmentation output shape (default: input image size)")

    parser.add_argument("--loss_report", '-r', metavar='ITER', default=50, type=int,
                        help="Report loss on every n-th iteration. Set to -1 to turn it off (default: 50)")
    parser.add_argument("--visualize", '-v', default=True, action='store_true',
                        help="Visualize results in eval mode (default: True)")

    return parser.parse_args()


def main():

    parsed_args = parse_args()

    # dataset related
    davis_dir = parsed_args.dataset
    year = parsed_args.year
    dataset_mode = parsed_args.set
    seq_names = parsed_args.sequences
    shuffle = parsed_args.shuffle

    # model related
    mode = parsed_args.mode
    cuda_dev = parsed_args.cuda_device
    model_save_path = parsed_args.model_save
    model_load_path = parsed_args.model_load

    # training related
    batch_size = parsed_args.batch_size
    epochs = parsed_args.epochs
    iters = parsed_args.iters
    lr = parsed_args.learning_rate
    weight_decay = parsed_args.weight_decay

    # videomatch related
    img_shape = parsed_args.input_image_shape
    seg_shape = parsed_args.segmentation_shape

    # misc
    loss_report_iter = parsed_args.loss_report
    visualize = parsed_args.visualize

    # TODO: add logger
    # args checks
    if mode == 'train' and batch_size != 1:
        print("Warning: Batch size > 1 is only applicable to 'val' mode.")

    if iters != -1 and epochs > 1:
        print("Warning: iters is set to {} and not to -1 (full dataset), but epoch is > 1".format(iters))

    if mode == 'eval' and shuffle:
        print("Warning: dataset shuffle can't be set to True in 'eval' mode, setting it to False!")
        shuffle = False

    if mode != 'eval' and visualize:
        print("Warning: visualize is set to True, but mode isn't 'eval'")

    device = None if cuda_dev is None else "cuda:{:d}".format(cuda_dev)

    # TODO: create class for transforms that applies equal transformation to both img and mask
    transforms = None
    dataset = Davis(davis_dir, year, dataset_mode, seq_names, transforms)

    vm = VideoMatch(out_shape=seg_shape, device=device)
    if model_load_path is not None:
        vm.load_model(model_load_path)

    if mode == 'train':
        pair_sampler = PairSampler(dataset, randomize=shuffle)
        data_loader = DataLoader(dataset, batch_sampler=pair_sampler, collate_fn=collate_pairs)
        iters = len(data_loader) if iters == -1 else iters

        train_vm(data_loader, vm, device, lr, weight_decay, iters, epochs, loss_report_iter, model_save_path)
    elif mode == 'eval':
        multiframe_sampler = MultiFrameSampler(dataset)
        data_loader = DataLoader(dataset, sampler=multiframe_sampler, collate_fn=collate_multiframes,
                                 batch_size=batch_size, num_workers=batch_size)

        eval_vm(data_loader, vm, visualize)


def train_vm(data_loader, vm, device, lr, weight_decay, iters, epochs=1, loss_report_iter=10, model_save_path=None):

    # set model to train mode
    vm.feat_net.train()

    params = filter(lambda p: p.requires_grad, vm.feat_net.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    stop_training = False

    # save model on SIGINT (Ctrl + c)
    def sigint_handler(signal, frame):
        print("Ctrl+c caught, stopping the training and saving the model...")
        nonlocal stop_training
        stop_training = True

    signal.signal(signal.SIGINT, sigint_handler)

    for epoch in range(epochs):
        print("Epoch: \t[{}/{}]".format(epoch + 1, epochs))

        start = time()
        for i, (ref_img, ref_mask, test_img, test_mask) in enumerate(data_loader):
            if i >= iters or stop_training:
                break

            # initialize every time since reference image keeps changing
            vm.seq_init(ref_img, ref_mask)
            out_mask = vm.segment(test_img).float()
            out_mask.requires_grad = True

            test_mask = test_mask.cuda(device).float()

            loss = criterion(input=out_mask, target=test_mask)

            if i % loss_report_iter == 0 and i > 0:
                end = time() - start
                print("Loss for iter [{:5d}/{}]:\t {:.2f}, \t it took {:.2f} s".format(i, iters, loss.data.mean(), end))
                start = time()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if stop_training:
            break

    if model_save_path is not None:
        vm.save_model(model_save_path)


def eval_vm(data_loader, vm, visualize=True):

    # set model to eval mode
    vm.feat_net.eval()

    ref_frame = None
    prev_seq_idx = -1
    for frames in data_loader:
        if prev_seq_idx != frames[0].seq_idx:
            ref_frame = frames[0]
            test_frames = frames[1:]
            prev_seq_idx = ref_frame.seq_idx
        else:
            test_frames = frames


if __name__ == '__main__':
    main()
