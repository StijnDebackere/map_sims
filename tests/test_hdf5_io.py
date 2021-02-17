from functools import wraps
import multiprocessing
import os
from pathlib import Path
import time

import h5py
import numpy as np

TEST_DIR = Path(__file__).parent


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


def generate_random_file(i=0, size=int(1e7)):
    """Generate large-ish file with random data."""
    with h5py.File(TEST_DIR / f'test_{i:03d}.hdf5', 'a') as f:
        f.create_dataset(
            'PartType0/Coordinates', data=np.random.uniform(0, 400, (3, size))
        )
        f.create_dataset(
            'PartType0/Masses', data=np.random.uniform(1e-2, 1e4, (size,))
        )
        f.create_dataset(
            'PartType1/Coordinates', data=np.random.uniform(0, 400, (3, size))
        )
        f.create_dataset(
            'PartType1/Masses', data=np.random.uniform(1e-2, 1e4, (size,))
        )


def heavy_operations(h5file_name, sort_axis=0):
    with h5py.File(h5file_name, 'r') as f:
        coords = f['PartType0/Coordinates'][:]
        masses = f['PartType0/Masses'][:]

    sort_idx = np.argsort(coords[sort_axis])
    return np.log10(coords[:, sort_idx] / masses[sort_idx])

def multiple_read_single_file(n_cpus=10):
    out_q = multiprocessing.Queue()
    procs = []
    for _ in range(n_cpus):
        proc = multiprocessing.Process(
            target=on_queue,
            args=(out_q, heavy_operations),
            kwargs={
                'h5file_name': str(TEST_DIR / 'test_000.hdf5'),
            }
        )
        procs.append(proc)
        proc.start()

    results = []
    for _ in range(n_cpus):
        results.append(out_q.get())

    for proc in procs:
        proc.join()

    results.sort()
    times = np.array([r[2] for r in results])
    return times


def multiple_read_different_file(n_cpus=10):
    out_q = multiprocessing.Queue()
    procs = []
    for i in range(n_cpus):
        proc = multiprocessing.Process(
            target=on_queue,
            args=(out_q, heavy_operations),
            kwargs={
                'h5file_name': str(TEST_DIR / f'test_{i:03d}.hdf5'),
            }
        )
        procs.append(proc)
        proc.start()

    results = []
    for _ in range(n_cpus):
        results.append(out_q.get())

    for proc in procs:
        proc.join()

    results.sort()
    times = np.array([r[2] for r in results])
    return times


def test_multiple_read(mode='single', n_runs=20, n_cpus=32):
    times = np.empty((n_runs, n_cpus), dtype=float)
    if mode == 'single':
        for i in range(n_runs):
            times[i] = multiple_read_single_file(n_cpus=n_cpus)

    elif mode == 'different':
        for i in range(n_runs):
            times[i] = multiple_read_single_file(n_cpus=n_cpus)

    return times
