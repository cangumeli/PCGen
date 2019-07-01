import h5py
from data.modelnet import list_images, offs_by_images
from data.io import read_off
from data.transforms import sample
import os


sample_size = 1024
target_dir = 'modelnet10_images_new_12x'
source_dir = 'modelnet40_images_new_12x'
classes = [
     'bathtub',
     'bed',
     'chair',
     'desk',
     'dresser',
     'monitor',
     'night_stand',
     'sofa',
     'table',
     'toilet'
]
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
    for fld in classes:
        os.system('cp -r {}/{} {}/{}'.format(
            source_dir, fld, target_dir, fld))

off_files = offs_by_images(list_images(target_dir))
with open('val10.txt', 'r') as f:
    val_files = set(f.readlines())
    with h5py.File('modelnet10.hdf5', 'w') as f:
        val_grp = f.create_group('val')
        train_grp = f.create_group('train')
        for filename in off_files:
            print('Saving file {}...'.format(filename))
            points, faces = [a.numpy() for a in read_off(filename)]
            big_grp = val_grp if filename in val_files else train_grp
            grp = big_grp.create_group(filename)
            if filename in val_files:
                points = sample(points, faces, sample_size)
                grp.create_dataset(
                    'points', points.shape, data=points, dtype='f')
            else:
                grp.create_dataset(
                    'points', points.shape, data=points, dtype='f')
                grp.create_dataset('faces', faces.shape, data=faces, dtype='i')
