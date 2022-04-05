import os
import h5py
from queue import Empty
import multiprocessing


def get_hdf5_group_keys(file_path, data_path=None):
    """Note: segmentation faults seem to be caused only when iterating the HDF5 root.
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with h5py.File(file_path, mode="r") as group:
        if data_path:
            group = group[data_path]
        else:
            group = group["/"]  # to preserve the order
        return list(group.keys())


def safe_hdf5_group_keys(file_path, data_path=None):
    return run_in_subprocess(
        get_hdf5_group_keys, file_path, data_path=data_path, default=list()
    )


def run_in_subprocess(target, *args, context=None, default=None, **kwargs):
    ctx = multiprocessing.get_context(context)
    queue = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=subprocess_main,
        args=(queue, target) + args,
        kwargs=kwargs,
    )
    p.start()
    try:
        p.join()
        try:
            return queue.get(block=False)
        except Empty:
            return default
    finally:
        try:
            p.kill()
        except AttributeError:
            p.terminate()


def subprocess_main(queue, method, *args, **kwargs):
    queue.put(method(*args, **kwargs))
