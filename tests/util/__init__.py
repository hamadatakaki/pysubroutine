import unittest
from subroutine.util import waveform
import numpy as np


def generate_sin_function(length=220500, sr=44100, f=1600):
    sec = int(length / sr)
    t = np.arange(0, sec, 1 / sr)
    w = np.sin(2 * np.pi * f * t)
    return sr, w.astype(np.float64)


class TestWaveform(unittest.TestCase):
    def test_resample(self):
        L = 20000
        sr = 4000

        _, y1 = generate_sin_function(L, sr, f=1600)
        y2 = waveform.resample(y1, sr, sr / 2)
        y3 = waveform.resample(y1, sr, sr * 2)

        self.assertEqual(len(y1), len(y2) * 2)
        self.assertEqual(len(y1), len(y3) / 2)
