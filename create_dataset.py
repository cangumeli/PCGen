import h5py
from data.modelnet import list_images, offs_by_images
from data.io import read_off


off_files = offs_by_images(list_images())
with h5py.File('modelnet40.hdf5', 'w') as f:
    for filename in off_files:
        print('Saving file {}...'.format(filename))
        points, faces = [a.numpy() for a in read_off(filename)]
        grp = f.create_group(filename)
        grp.create_dataset('points', points.shape, data=points, dtype='f')
        grp.create_dataset('faces', faces.shape, data=faces, dtype='i')
