import numpy as np
from scipy.io import wavfile
from typing import Union, Optional
from pathlib import Path
from subroutine.util import waveform


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
