import os
from random import shuffle
from itertools import combinations

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from log import logger


class Davis(Dataset):

    years = '2016', '2017', 'all'
    modes = 'train', 'val', 'trainval'

    def __init__(self, base_dir, year='2016', mode='train', use_seq=None):
        super().__init__()

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

        self.seq_names = []
        for y in self.year:
            for m in self.mode:
                filepath = os.path.join(self._image_sets_dir, y, m + ".txt")
                with open(filepath) as file:
                    self.seq_names.extend(s.strip() for s in file.readlines())

        # 2016 and 2017 davis datasets contain duplicates
        self.seq_names = set(self.seq_names)

        if use_seq is not None:
            # check if specified sequences are valid
            use_seq = set(use_seq)
            if not use_seq.issubset(self.seq_names):
                raise RuntimeError("Specified set of sequence names isn't subset of loaded DAVIS dataset (year: {}),"
                                   "\ngiven: {},\nvalid: {}".format(self.year, use_seq, self.seq_names))

            # only use specified sequences
            self.seq_names = use_seq & self.seq_names

        self.sequences = [Sequence(name, self._annotations_dir, self._images_dir) for name in sorted(self.seq_names)]

        logger.debug("Number of specified sequences in davis dataset is {}".format(self.__len__()))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, img_pos):
        seq_idx, frame_idx = img_pos
        img_path, ann_path = self.sequences[seq_idx][frame_idx]

        # load image and annotation with PIL
        img = Image.open(img_path)
        ann = Image.open(ann_path)

        frame = Frame(self.sequences[seq_idx], frame_idx, img, ann)

        return frame


class Frame:
    def __init__(self, seq, frame_idx, img, ann):
        self.seq = seq
        self.frame_idx = frame_idx
        self.img = img
        self.ann = ann


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

    # TODO: this compares sequences by name (string), is there a performance hit?
    def __eq__(self, other):
        return other is not None and self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class PairSampler(Sampler):
    def __init__(self, dataset, randomize=True):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._randomize = randomize
        self._sequences = self._dataset.sequences

        self._all_pairs = []
        for seq_idx, seq in enumerate(self._sequences):
            idx_pairs = [(seq_idx, frame_idx) for frame_idx in range(len(seq))]
            self._all_pairs.extend(list(combinations(idx_pairs, 2)))

        if self._randomize:
            shuffle(self._all_pairs)

        logger.debug("Number of all pairs {}".format(self.__len__()))

    def __iter__(self):
        for pair_idx in self._all_pairs:
            yield pair_idx

    def __len__(self):
        return len(self._all_pairs)


def collate_pairs(data):
    ref_frame, test_frame = data

    return (ref_frame.img, ref_frame.ann), (test_frame.img, test_frame.ann)


class MultiFrameSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._sequences = self._dataset.sequences

        self._samples = []
        for seq_idx, seq in enumerate(self._sequences):
            idx_pairs = [(seq_idx, frame_idx) for frame_idx in range(len(seq))]
            self._samples.extend(idx_pairs)

    def __iter__(self):
        for frame_idx in self._samples:
            yield frame_idx

    def __len__(self):
        return len(self._samples)


def collate_multiframes(data):
    return data


if __name__ == '__main__':
    davis = Davis("./DAVIS")

    ps = PairSampler(davis)

    ps = iter(ps)
    pair_i = next(ps)
    first_pair = davis[pair_i[0]], davis[pair_i[1]]
    print("First frames shapes = ", first_pair[0][0].shape, first_pair[0][1].shape)
    print("Second frames shapes = ", first_pair[1][0].shape, first_pair[1][1].shape)
