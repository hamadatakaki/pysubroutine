import dataclasses
from typing import Any, Optional, Union

import sys
import abc


import librosa
import numpy as np
import pysptk
import pyworld
from scipy import interpolate

FeatureObject = Union[np.ndarray, int, float]


@dataclasses.dataclass
class BaseSpeechFeature(abc.ABC):
    config: dict[str, Any]
    waveform: Optional[np.ndarray] = dataclasses.field(default=None)

    def __post_init__(self):
        if self.waveform is not None:
            self.waveform = self.waveform.astype(np.float64)

        self.assert_config()

    def assert_config(self):
        if self.config_keys is None:
            raise NotImplementedError("Self::config_keys (list[str])")

        exist_keys = [key in self.config for key in self.config_keys]

        err = False
        for res, key in zip(exist_keys, self.config_keys):
            if not res:
                err = True
                print(f"[warning] config not have key: {key}", file=sys.stderr)

        if err:
            raise RuntimeError(
                f"{self.__class__.__name__}::assert_config - some keys not exist."
            )


@dataclasses.dataclass
class LibrosaFeature(BaseSpeechFeature):
    features: list[str] = dataclasses.field(
        default_factory=lambda: ["spectrogram", "mel_spectrogram"], init=False
    )
    config_keys: list[str] = dataclasses.field(
        default_factory=lambda: [
            "fft_length",
            "hop_length",
            "sampling_rate",
            "win_length",
            "window",
            "num_mels",
        ],
        init=False,
    )

    _spec: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    _melspec: Optional[np.ndarray] = dataclasses.field(default=None, init=False)

    def spectrogram(self) -> np.ndarray:
        if self._spec is None:
            fft_length = self.config["fft_length"]
            hop_length = self.config["hop_length"]
            win_length = self.config["win_length"]

            self._spec = librosa.stft(
                self.waveform,
                n_fft=fft_length,
                hop_length=hop_length,
                win_length=win_length,
            )

        return self._spec

    def mel_spectrogram(self) -> np.ndarray:
        if self._melspec is None:
            sr = self.config["sampling_rate"]
            fft_length = self.config["fft_length"]
            n_mels = self.config["num_mels"]

            spec = self.spectrogram()
            melfb = librosa.filters.mel(sr=sr, n_fft=fft_length, n_mels=n_mels)
            self._melspec = librosa.amplitude_to_db(
                np.dot(melfb, np.abs(spec)), ref=np.max
            )

        return self._melspec

    def get_features(
        self, waveform: Optional[np.ndarray] = None
    ) -> dict[str, FeatureObject]:
        if waveform is not None:
            self.waveform = waveform.astype(np.float64)

        if self.waveform is None:
            raise RuntimeError("LibrosaFeature::fet_features - Waveform is None")

        return {
            "spectrogram": self.spectrogram(),
            "mel_spectrogram": self.mel_spectrogram(),
        }


@dataclasses.dataclass
class WorldFeature(BaseSpeechFeature):
    features: list[str] = dataclasses.field(
        default_factory=lambda: [
            "log_f0",
            "vuv_flag",
            "mel_cepstrum",
            "band_aperiodicity",
        ],
        init=False,
    )
    config_keys: list[str] = dataclasses.field(
        default_factory=lambda: ["frame_period", "sampling_rate", "mgc_order"],
        init=False,
    )

    _f0: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    _timeaxis: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    _lf0: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    _vuv: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    _spec: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    _mgc: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    _bap: Optional[np.ndarray] = dataclasses.field(default=None, init=False)

    def f0_and_timeaxis(self) -> tuple[np.ndarray, np.ndarray]:
        if self._f0 is None or self._timeaxis is None:
            sr = self.config["sampling_rate"]
            frame_period = self.config["frame_period"]

            f0, timeaxis = pyworld.dio(self.waveform, sr, frame_period=frame_period)
            self._f0 = f0
            self._timeaxis = timeaxis

        assert self._f0 is not None
        assert self._timeaxis is not None

        return (self._f0, self._timeaxis)

    def log_f0(self) -> np.ndarray:
        if self._lf0 is None:
            f0, _ = self.f0_and_timeaxis()

            f0[f0 < 0] = 0
            lf0 = f0.copy()
            nonzero = np.nonzero(f0)
            lf0[nonzero] = np.log(f0[nonzero])
            self._lf0 = self.__interp1d(lf0).reshape((-1, 1))

        return self._lf0

    def vuv_flag(self) -> np.ndarray:
        if self._vuv is None:
            f0, _ = self.f0_and_timeaxis()
            vuv = (f0 > 0).astype(np.float64)
            self._vuv = vuv.reshape((-1, 1))

        return self._vuv

    def spectrogram(self) -> np.ndarray:
        if self._spec is None:
            sr = self.config["sampling_rate"]

            f0, timeaxis = self.f0_and_timeaxis()
            self._spec = pyworld.cheaptrick(self.waveform, f0, timeaxis, sr)

        return self._spec

    def mel_cepstrum(self) -> np.ndarray:
        if self._mgc is None:
            sr = self.config["sampling_rate"]
            mgc_order = self.config["mgc_order"]
            alpha = pysptk.util.mcepalpha(sr)

            spec = self.spectrogram()
            self._mgc = pysptk.sp2mc(spec, mgc_order, alpha)

        return self._mgc

    def band_aperiodicity(self) -> np.ndarray:
        if self._bap is None:
            sr = self.config["sampling_rate"]

            f0, timeaxis = self.f0_and_timeaxis()
            ap = pyworld.d4c(self.waveform, f0, timeaxis, sr)
            self._bap = pyworld.code_aperiodicity(ap, sr)

        return self._bap

    def get_features(
        self, waveform: Optional[np.ndarray] = None
    ) -> dict[str, FeatureObject]:
        if waveform is not None:
            self.waveform = waveform.astype(np.float64)

        if self.waveform is None:
            raise RuntimeError("WorldFeature::fet_features - Waveform is None")

        return {
            "log_f0": self.log_f0(),
            "vuv_flag": self.vuv_flag(),
            "mel_cepstrum": self.mel_cepstrum(),
            "band_aperiodicity": self.band_aperiodicity(),
        }

    def __interp1d(self, f0: np.ndarray, kind: str = "slinear") -> np.ndarray:
        ndim = f0.ndim
        if len(f0) != f0.size:
            raise RuntimeError("1D Array is only supported.")

        continuous_f0 = f0.flatten()
        nonzero_indices = np.where(continuous_f0 > 0)[0]

        if len(nonzero_indices) <= 0:
            return f0

        continuous_f0[0] = continuous_f0[nonzero_indices[0]]
        continuous_f0[-1] = continuous_f0[nonzero_indices[-1]]

        nonzero_indices = np.where(continuous_f0 > 0)[0]
        interp_func = interpolate.interp1d(
            nonzero_indices, continuous_f0[continuous_f0 > 0], kind=kind
        )

        zero_indices = np.where(continuous_f0 <= 0)[0]
        continuous_f0[zero_indices] = interp_func(zero_indices)

        if ndim == 2:
            return continuous_f0[:, None]

        return continuous_f0
