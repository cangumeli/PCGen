import torch
from PIL import Image
from threading import Thread


def read_off(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    line_iter = iter(lines)

    def _read_nums(dtype):
        line = next(line_iter).strip().split()
        return [dtype(w) for w in line]

    start = next(line_iter)
    if start.strip() != 'OFF':
        line = start[3:].strip().split()
        npoints, nfaces, _ = [int(w) for w in line]
    else:
        npoints, nfaces, _ = _read_nums(int)
    # Read points
    points = []
    for i in range(npoints):
        points.append(_read_nums(float))
    points = torch.tensor(points)
    # Read faces
    faces = []
    for i in range(nfaces):
        faces.append(_read_nums(int)[1:])
    faces = torch.tensor(faces, dtype=torch.long)
    return points, faces.t().contiguous()


def read_image(filename):
    return Image.open(filename).convert('RGB')


def read_offs(off_files, num_threads=1):
    chunks = {i: {} for i in range(num_threads)}
    elems_per_thread = len(off_files) // num_threads
    if num_threads > 0:
        def worker(chunk):
            start = chunk * elems_per_thread
            end = start + elems_per_thread
            for i in range(start, end):
                off_file = off_files[i]
                chunks[chunk][off_file] = read_off(off_file)
        threads = []
        for i in range(num_threads):
            threads.append(Thread(target=worker, args=(i,)))
            threads[-1].start()
        for t in threads:
            t.join()
        data = {}
        for i in range(len(chunks)):
            data.update(chunks[i])
        elems_rem = len(off_files) % num_threads
    else:
        elems_rem = len(off_files)
    for i in range(elems_rem, 0, -1):
        off_file = off_files[-i]
        data[off_file] = read_off(off_file)
    return data
