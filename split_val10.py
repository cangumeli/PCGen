from data.modelnet import list_images, off_by_image
import numpy as np
import sys
import os


filename = 'val10.txt' if len(sys.argv) == 1 else sys.argv[1]
off_filename = 'val10.off.txt' if len(sys.argv) == 1 else sys.argv[2]
if os.path.isfile(filename):
    print('{} already exist, explicitly run rm {}'.format(filename, filename))
    sys.exit(1)
if os.path.isfile(off_filename):
    print('{} already exist, explicitly run rm {}'.format(
        off_filename, off_filename))
    sys.exit(1)

val_ratio = 0.2
target_dir = 'modelnet10_images_new_12x'
val_off_set = set()
val_images = []
images = list_images(target_dir)
offs = set(off_by_image(img) for img in images)
val_offs = set()
train_offs = set()
for off in offs:
    if np.random.rand() <= val_ratio:
        val_offs.add(off)
    else:
        train_offs.add(off)
print('Off files ==> Train: {} / Val: {} -> Ratio: {}'.format(
    len(train_offs), len(val_offs),
    len(val_offs) / (len(train_offs) + len(val_offs))
))
with open(off_filename, 'w') as f:
    f.write(str.join('\n', val_offs))
for img in images:
    if off_by_image(img) in val_offs:
        val_images.append(img)
print('Image files ==> Train: {} / Val: {} -> Ratio: {}'.format(
    len(images) - len(val_images), len(val_images),
    len(val_images) / len(images)
))
print('Writing the file list...')
with open(filename, 'w') as f:
    f.write(str.join('\n', val_images))
