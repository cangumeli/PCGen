import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
from torch.utils.data import Dataset
import os
from math import pi
import h5py
from .transforms import rotatex, rotatez, normalize, sample, points_to_grid
from .io import read_image


def filter_files(files):
    return [f for f in files if not f.startswith('.')]


def list_images(dir='modelnet40_images_new_12x', train=True, verbose=False):
    images = []
    classes = filter_files(os.listdir(dir))
    for (i, cls) in enumerate(sorted(classes)):
        if verbose:
            print('Processing', cls, '...')
        clsdir = os.path.join(dir, cls)
        filedir = os.path.join(clsdir, 'train' if train else 'test')
        files = sorted(filter_files(os.listdir(filedir)))
        images.extend([
            os.path.join(filedir, f) for f in files if f.endswith('.png')
        ])
    return images


def offs_by_images(image_list, dir='ModelNet40'):
    off_set = set()
    off_list = []
    for img in image_list:
        off = off_by_image(img, dir)
        if off in off_set:
            continue
        off_list.append(off)
        off_set.add(off)
    return off_list


def off_by_image(image_path, dir='ModelNet40'):
    img_parts = image_path.split(os.sep)
    img_file = img_parts[-1].split('.')[0]
    return dir + os.sep + img_parts[-3] + os.sep + \
        img_parts[-2] + os.sep + img_file + '.off'


def angle_by_image(image_path):
    view_idx = int(image_path.split('.')[-2][-3:]) - 1
    return view_idx * (pi / 6)


IMAGE_LINK = 'http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz'
OFF_LINK = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'


class ModelNet40(Dataset):

    def __init__(
            self,
            points_to_sample=1024,
            coarse_sample_fraction=4,
            img_root='modelnet40_images_new_12x',
            off_root='ModelNet40',
            train=True,
            img_transform=Compose([
                ToTensor(),
                Normalize([.5, .5, .5], [.25, .25, .25])
            ]),
            cache_offs=False,
            cache_imgs=False,
            data_file='modelnet40.hdf5',
            grid=False,
            grid_size=(16, 64)
    ):
        self.img_root = img_root
        self.off_root = off_root
        self.image_files = list_images(img_root, train=train)
        self.points_to_sample = points_to_sample
        self.coarse_sample_fraction = coarse_sample_fraction
        self.img_transform = img_transform
        self.off_cache = {}
        self.img_cache = {}
        self.cache_imgs = cache_imgs
        self.cache_offs = cache_offs
        self.data_file = h5py.File(data_file, 'r')
        self.grid = grid
        self.grid_size = grid_size
        if self.grid:
            self.points_to_sample = self.grid_size[0] * self.grid_size[1]

    def __len__(self):
        return len(self.image_files)

    def _read_off_data(self, off_file):
        if off_file in self.off_cache:
            return self.off_cache[off_file]
        points = np.array(self.data_file[off_file + '/' + 'points'])
        faces = np.array(self.data_file[off_file + '/' + 'faces'])
        return torch.from_numpy(points), torch.from_numpy(faces).long()

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img = read_image(img_file) if img_file not in self.img_cache \
            else self.img_cache(img_file)
        if self.cache_imgs:
            self.img_cache[img_file] = img
        if self.img_transform is not None:
            img = self.img_transform(img)
        off_file = off_by_image(img_file, self.off_root)
        points, faces = self._read_off_data(off_file)
        if self.cache_offs:
            self.off_cache[off_file] = (points, faces)
        points = sample(points, faces, self.points_to_sample)
        points = normalize(points)
        # points = rotatez(points, pi)
        points = rotatex(points, pi/3)
        points = rotatez(points, angle_by_image(img_file))
        if self.grid:
            points = points_to_grid(points, self.grid_size)
        elif self.coarse_sample_fraction > 1:
            n_coarse = self.points_to_sample // self.coarse_sample_fraction
            coarse_points = points[torch.randperm(points.size(0))[:n_coarse]]
            return img, coarse_points, points
        return img, points
