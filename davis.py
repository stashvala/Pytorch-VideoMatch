import os
from random import shuffle
from itertools import combinations

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class Davis(Dataset):

    years = '2016', '2017', 'all'
    modes = 'train', 'val', 'trainval'

    def __init__(self, base_dir, year='2016', mode='train', transorms=None):
        assert(year in self.years)
        assert(mode in self.modes)

        self._base_dir = base_dir
        self._annotations_dir = os.path.join(self._base_dir, "Annotations", "480p")
        self._images_dir = os.path.join(self._base_dir, "JPEGImages", "480p")
        self._image_sets_dir = os.path.join(self._base_dir, "ImageSets")

        alldirs = self._annotations_dir, self._images_dir, self._image_sets_dir
        checkdirs = [os.path.isdir(d) for d in alldirs]
        if not all(checkdirs):
            raise ValueError("This is not expected DAVIS dataset dir structure, "
                             "you need the following dirs: \n\t{}".format("\n\t".join(alldirs)))

        self.year = [year] if year in self.years[:-1] else self.years[:-1]
        self.mode = [mode] if mode in self.modes[:-1] else self.modes[:-1]
        self.transforms = transorms if transorms is not None else self.basic_transform

        self.seq_names = []
        for y in self.year:
            for m in self.mode:
                filepath = os.path.join(self._image_sets_dir, y, m + ".txt")
                with open(filepath) as file:
                    self.seq_names.extend(s.strip() for s in file.readlines())

        # 2016 and 2017 davis datasets contain duplicates
        self.seq_names = set(self.seq_names)

        self.sequences = [Sequence(name, self._annotations_dir, self._images_dir) for name in sorted(self.seq_names)]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, img_pos):
        seq_idx, frame_idx = img_pos
        img_path, ann_path = self.sequences[seq_idx][frame_idx]

        # load image and annotation with PIL
        img = Image.open(img_path)
        ann = Image.open(ann_path)

        # do the necessary transformation and possible image augmentation
        img_t, ann_t = self.transforms(img, ann)

        return img_t, ann_t

    @staticmethod
    def basic_transform(img, ann):
        from torchvision import transforms

        # TODO: move this elsewhere!
        img_transform = transforms.Compose([
             transforms.Resize((256, 456)),
             transforms.ToTensor(),  # normalizes image to 0-1 values
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

        ann_transform = transforms.Compose([
             transforms.Resize((256, 456)),  # TODO: is this necessary?
             transforms.ToTensor(),  # normalizes image to 0-1 values
            ])

        return img_transform(img), ann_transform(ann)


class Sequence:
    def __init__(self, name, base_ann_dir, base_img_dir):
        self.name = name
        self._ann_dir = os.path.join(base_ann_dir, name)
        self._img_dir = os.path.join(base_img_dir, name)

        frame_imgs = [os.path.join(self._img_dir, frame_no) for frame_no in sorted(os.listdir(self._img_dir))]
        frame_anns = [os.path.join(self._ann_dir, frame_no) for frame_no in sorted(os.listdir(self._ann_dir))]

        # TODO: simple check, maybe check also matching filenames...
        assert(len(frame_imgs) == len(frame_anns))

        self.frames = [(img, ann) for img, ann in zip(frame_imgs, frame_anns)]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class PairSampler(Sampler):
    def __init__(self, dataset, seq_names=None, randomize=True):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._randomize = randomize

        if seq_names and all(sn in self._dataset.seq_names for sn in seq_names):
            self._sequences = [self._dataset.sequences[sn] for sn in seq_names]
        else:
            self._sequences = self._dataset.sequences

        self._all_pairs = []
        for seq_idx, seq in enumerate(self._sequences):
            idx_pairs = [(seq_idx, frame_idx) for frame_idx in range(len(seq))]
            self._all_pairs.extend(list(combinations(idx_pairs, 2)))

        if self._randomize:
            shuffle(self._all_pairs)

        self._num_samples = len(self._all_pairs)

    def __iter__(self):
        for first_idx, second_idx in self._all_pairs:
            first_frame = self._dataset[first_idx]
            second_frame = self._dataset[second_idx]
            yield first_frame, second_frame

    def __len__(self):
        return self._num_samples


if __name__ == '__main__':
    davis = Davis("./DAVIS")

    ps = PairSampler(davis)
    print("Number of pairs in pair sampler = ", len(ps))

    ps = iter(ps)
    first_pair = next(ps)
    print("First frames shapes = ", first_pair[0][0].shape, first_pair[0][1].shape)
    print("Second frames shapes = ", first_pair[1][0].shape, first_pair[1][1].shape)
