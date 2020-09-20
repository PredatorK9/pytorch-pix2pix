import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class RandomCropMap(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        sample = np.array(sample)
        _, width, _ = sample.shape
        sat_im, map_im = sample[:, :width//2, :], sample[:, width//2:, :]

        h, w = sat_im.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        side = np.random.randint(0, w - new_w)

        sat_im = sat_im[top: top+new_h, side: side+new_w]
        map_im = map_im[top: top+new_h, side: side+new_w]

        sample = np.concatenate([sat_im, map_im], axis=1)
        sample = Image.fromarray(sample)
        return sample


class MapDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.root,
            self.files[idx])).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)[:3, :, :]

        _, _, width = image.shape
        sample = (image[:, :, :width//2], image[:, :, width//2:])
        return sample
