from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.io import wavfile

from subroutine.util import waveform, timestamp
import h5py


def load_wav(
    path: Union[Path, str], sr: Optional[Union[int, float]] = None, **kargs
) -> tuple[float, np.ndarray]:
    """load_wav

    Args:
        path (Union[Path, str]): wav file's path
        sr (Optional[Union[int, float]], optional):
            specified sampling rate. Defaults to None.

    Returns:
        float: waveform sampling rate.
        np.ndarray: wavefrom numpy array
    """

    path = Path(path)
    assert path.exists(), f"Path not exist: {path}"
    assert path.suffix == ".wav", f"Path is not waveform: {path}"

    sr1, w = wavfile.read(path)
    sr2 = sr

    # resampling
    if sr2 is not None:
        w = waveform.resample(w, sr1, sr2)
    else:
        sr2 = sr1

    # type cast
    if "dtype" in kargs:
        w = w.astype(kargs["dtype"])

    return sr2, w


def rec_dump_h5(file: h5py.File, group: str, **h5data):
    for key, value in h5data.items():
        abskey = f"{group}{key}"
        if type(value) == dict:
            file.create_group(abskey)
            rec_dump_h5(file, f"{abskey}/", **value)
        elif type(value) == np.ndarray:
            file.create_dataset(abskey, data=value)
        else:
            print(f"[warning] value must be np.ndarray or dict: {key}({type(value)})")


def dump_h5(path: Union[str, Path], overwrite: bool = True, **h5data) -> Path:
    path = Path(path)

    if path.suffix != ".hdf5":
        print(f"[warning] Path is not hdf5 file: {path}")
        path = path.parent / f"{path.name}.hdf5"

    if path.exists() and not overwrite:
        ts = timestamp.now_datetime_text()
        altpath = path.parent / f"{path.stem}-{ts}{path.suffix}"
        print(f"[warning] Path already existed, converting path: {path} -> {altpath}")
        path = altpath

    with h5py.File(path, "w") as file:
        rec_dump_h5(file, "", **h5data)

    return path


def rec_load_h5(h5file) -> dict:
    data = {}

    for key, value in h5file.items():
        if type(value) == h5py._hl.dataset.Dataset:
            array = np.array(value)
            data |= {key: array}
        elif type(value) == h5py._hl.group.Group:
            data |= {key: rec_load_h5(value)}
        else:
            raise NotImplementedError(f"load_h5 - not implemented type: {type(value)}")

    return data


def load_h5(path: Union[str, Path]) -> dict:
    path = Path(path)

    if path.suffix != ".hdf5":
        print(f"[warning] Path is not hdf5 file: {path}")
    assert path.exists(), f"Path not exist: {path}"

    with h5py.File(path, "r") as file:
        data = rec_load_h5(file)

    return data
