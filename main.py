from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from videomatch import VideoMatch
from davis import Davis, PairSampler, collate_fn

def main():

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
    loss_report_iter = 10

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


def train(data_loader, vm, device, lr, weight_decay, iters, epochs=1, loss_report_iter=10):

    vm.feat_net.train()
    params = filter(lambda p: p.requires_grad, vm.feat_net.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print("Epoch: \t{}/{}".format(epoch, epochs))

        for i, (ref_img, ref_mask, test_img, test_mask) in enumerate(data_loader):
            if i >= iters:
                break

            # initialize every time since reference image keeps changing
            vm.seq_init(ref_img, ref_mask)
            out_mask = vm.segment(test_img)

            test_mask = test_mask.cuda(device)
            
            loss = criterion(out_mask.float(), test_mask.float())

            if i % loss_report_iter == 0:
                print("Loss for iter {}/{}: {}".format(i, iters, loss.data.mean()))

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()
