from typing import Union

import numpy as np
import scipy


def resample(
    w: np.ndarray, sr1: Union[int, float], sr2: Union[int, float]
) -> np.ndarray:
    """resample

    Args:
        w (np.ndarray): waveform numpy array.
        sr1 (Union[int, float]): origin sampling rate.
        sr2 (Union[int, float]): resampling rate.

    Returns:
        np.ndarray: resampled waveform
    """

    assert sr1 > 0, "sr1 must be larger than 0."

    if sr1 != sr2:
        L = len(w)
        resampled_len = int(L * sr2 / sr1)
        w = scipy.signal.resample(w, resampled_len)

    return w
