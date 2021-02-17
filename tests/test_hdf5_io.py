import multiprocessing
import os
import time

import h5py
import numpy as np


def time_this(func, pid=False):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        if pid:
            print(
                f'Process {os.getpid()}: '
                f'Evaluating {func.__name__} took {t2 - t1:.2f}s'
            )
        else:
            print(f'Evaluating {func.__name__} took {t2 - t1:.2f}s')
        return (result, t2 - t1)
    return wrapper


def on_queue(queue, func, *args, **kwargs):
    res, dt = time_this(func, pid=True)(*args, **kwargs)
    queue.put([os.getpid(), res, dt])


def generate_random_file():
    """Generate large-ish file with random data."""
    with h5py.File('test.asdf', 'a') as f:
        f.create_dataset(
            'PartType0/Coordinates', data=np.random.uniform(0, 400, (3, 1e7))
        )
        f.create_dataset(
            'PartType0/Masses', data=np.random.uniform(1e-2, 1e4, (1e7,))
        )
        f.create_dataset(
            'PartType1/Coordinates', data=np.random.uniform(0, 400, (3, 1e7))
        )
        f.create_dataset(
            'PartType1/Masses', data=np.random.uniform(1e-2, 1e4, (1e7,))
        )


def heavy_operations(h5file_name):
    with h5py.File(h5file_name, 'r') as f:
        coords = f['PartType0/Coordinates'][:]
        masses = f['PartType0/Masses'][:]

    sort_idx = np.argsort(coords)
    return np.log10(coords[sort_idx] / masses[sort_idx])

def test_multiple_read_single_file(n_cpus=10):
    out_q = multiprocessing.Queue()
    procs = []
    for _ in range(n_cpus):
        proc = multiprocessing.Process(
            target=on_queue,
            args=(out_q, heavy_operations),
            kwargs={
                'h5file_name': 'test.asdf',
            }
        )
        procs.append(proc)
        proc.start()

    results = []
    for _ in range(n_cpus):
        results.append(out_q.get()[0, 2])

    for proc in procs:
        proc.join()

    results.sort()
    times = [r[1] for r in results]
    return times
