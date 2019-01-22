import argparse

from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from videomatch import VideoMatch
from davis import Davis, PairSampler, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Real time video object segmentation with VideoMatch")

    parser.add_argument("--dataset_path", '-d', default="./DAVIS", type=str,
                        help="Path to DAVIS dataset (default: ./DAVIS)")
    parser.add_argument("--year", '-y', default='2016', choices=['2016', '2017', 'all'], type=str,
                        help="DAVIS challenge year (default: 2016)")
    parser.add_argument("--set", '-t', default='train', choices=['train', 'val', 'trainval'], type=str,
                        help="Construct dataset from DAVIS train, val or all sequences (default: train)")
    parser.add_argument("--sequences", '-q', default=('-1',), nargs='+', metavar="SEQ_NAME",
                        help="List of sequence names to include in the dataset. Set to -1 to choose all. (default: -1)")
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

    parser.add_argument("--image_shape", default=(256, 456), metavar=('HEIGHT', 'WIDTH'), nargs=2, type=int,
                        help="Input image shape (default: 256 456)")
    parser.add_argument("--segmentation_shape", '-g', metavar=('HEIGHT', 'WIDTH'), nargs=2, type=int,
                        help="Segmentation output shape (default: input image size)")

    parser.add_argument("--loss_report", '-r', default=50, metavar='ITER', type=int,
                        help="Report loss on every n-th iteration. Set to -1 to turn it off (default: 50)")

    return parser.parse_args()


def main():

    parsed_args = parse_args()
    exit(0)

    # dataset related
    davis_dir = "./DAVIS"
    year = '2016'
    mode = 'train'
    transforms = None
    shuffle = True

    # cnn related
    cuda_dev = 0
    batch_size = 1
    epochs = 1
    iters = -1
    lr = 1e-5
    weight_decay = 5e-4

    model_checkpoint_path = None
    model_save_path = None
    loss_report_iter = 50

    # videomatch related
    out_shape = None

    # TODO: add logger
    # args checks
    if mode == 'train' and batch_size != 1:
        print("Warning: Batch size > 1 is only applicable to 'val' mode.")

    if iters != -1 and epochs > 1:
        print("Warning: iters is set to {} and not to -1 (full dataset), but epoch is > 1".format(iters))

    device = "cpu" if cuda_dev is None else "cuda:{:d}".format(cuda_dev)

    model = None
    if model_checkpoint_path is not None:
        pass  # load model
    else:
        pass  # create model

    dataset = Davis(davis_dir, year, mode, transforms)
    pair_sampler = PairSampler(dataset, randomize=shuffle)
    data_loader = DataLoader(dataset, batch_sampler=pair_sampler, collate_fn=collate_fn)

    if iters == -1:
        iters = len(data_loader)

    vm = VideoMatch(out_shape=out_shape, device=device)

    if mode == 'train':
        train(data_loader, vm, device, lr, weight_decay, iters, epochs, loss_report_iter)


def train(data_loader, vm, device, lr, weight_decay, iters, epochs=1, loss_report_iter=10, model_save_path=None):

    # set model to train mode
    vm.feat_net.train()

    params = filter(lambda p: p.requires_grad, vm.feat_net.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print("Epoch: \t[{}/{}]".format(epoch, epochs))

        for i, (ref_img, ref_mask, test_img, test_mask) in enumerate(data_loader):
            if i >= iters:
                break

            # initialize every time since reference image keeps changing
            vm.seq_init(ref_img, ref_mask)
            out_mask = vm.segment(test_img).float()
            out_mask.requires_grad = True

            test_mask = test_mask.cuda(device).float()

            loss = criterion(input=out_mask, target=test_mask)

            if i % loss_report_iter == 0:
                print("Loss for iter [{:5d}/{}]:\t {:.2f}".format(i, iters, loss.data.mean()))

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if model_save_path is not None:
        vm.save_model(model_save_path)


if __name__ == '__main__':
    main()
