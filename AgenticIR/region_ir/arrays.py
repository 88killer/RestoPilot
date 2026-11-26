"""Lossless sample arrays with explicit legacy NPY and compressed NPZ support."""
from pathlib import Path
import numpy as np


def sample_array_path(directory, name, required=True):
    directory = Path(directory)
    found = [directory / (name + suffix) for suffix in ('.npy', '.npz')
             if (directory / (name + suffix)).is_file()]
    if len(found) > 1:
        raise ValueError(f'Ambiguous sample array: {directory / name}')
    if not found:
        if required:
            raise FileNotFoundError(f'Missing sample array: {directory / name}')
        return None
    return found[0]


def load_array(path, key=None):
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        with value:
            selected = key or (value.files[0] if len(value.files) == 1 else None)
            if selected is None or value.files != [selected]:
                raise ValueError(f'Expected one array named {key!r}: {path}')
            return value[selected]
    return value


def load_sample_array(directory, name, required=True):
    path = sample_array_path(directory, name, required)
    return None if path is None else load_array(path, name)


def save_sample_array(directory, name, array, storage='npy'):
    if storage not in ('npy', 'npz'):
        raise ValueError('Array storage must be npy or npz')
    if sample_array_path(directory, name, required=False) is not None:
        raise FileExistsError(f'Sample array already exists: {Path(directory) / name}')
    path = Path(directory) / (name + '.' + storage)
    if storage == 'npz':
        np.savez_compressed(path, **{name: array})
    else:
        np.save(path, array, allow_pickle=False)
    return path
