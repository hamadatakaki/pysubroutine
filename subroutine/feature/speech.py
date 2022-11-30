import dataclasses
from typing import Any, Optional, Union

import librosa
import numpy as np
import pysptk
import pyworld
from scipy import interpolate

FeatureObject = Union[np.ndarray, int, float]


@dataclasses.dataclass
class LibrosaFeature(object):
    config: dict[str, Any]
    waveform: np.ndarray
    spec: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    melspec: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    # features: list[str] = dataclasses.field(default=["spec", "melspec"], init=False)

    def __post_init__(self):
        self.waveform = self.waveform.astype(np.float64)
        self.assert_config()

    def assert_config(self):
        config_keys = [
            "fft_length",
            "hop_length",
            "sampling_rate",
            "win_length",
            "window",
            "num_mels",
        ]
        exist_keys = [key in self.config for key in config_keys]

        err = False
        for res, key in zip(exist_keys, config_keys):
            if not res:
                err = True
                print(f"assert_config > key not exist: {key}")

        assert not err, "LibrosaFeature::assert_config - some keys not exist."

    def spectrogram(self) -> np.ndarray:
        if self.spec is None:
            fft_length = self.config["fft_length"]
            hop_length = self.config["hop_length"]
            win_length = self.config["win_length"]

            self.spec = librosa.stft(
                self.waveform,
                n_fft=fft_length,
                hop_length=hop_length,
                win_length=win_length,
            )

        return self.spec

    def mel_spectrogram(self) -> np.ndarray:
        if self.melspec is None:
            sr = self.config["sampling_rate"]
            fft_length = self.config["fft_length"]
            n_mels = self.config["num_mels"]

            spec = self.spectrogram()
            melfb = librosa.filters.mel(sr=sr, n_fft=fft_length, n_mels=n_mels)
            self.melspec = librosa.amplitude_to_db(
                np.dot(melfb, np.abs(spec)), ref=np.max
            )

        return self.melspec

    def to_dict(self) -> dict[str, FeatureObject]:
        return {"spec": self.spectrogram(), "melspec": self.mel_spectrogram()}


@dataclasses.dataclass
class WorldFeature(object):
    config: dict[str, Any]
    waveform: np.ndarray
    f0: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    timeaxis: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    lf0: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    vuv: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    spec: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    mgc: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    bap: Optional[np.ndarray] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.waveform = self.waveform.astype(np.float64)
        self.assert_config()

    def assert_config(self):
        config_keys = [
            "frame_period",
            "sampling_rate",
            "mgc_order",
        ]
        exist_keys = [key in self.config for key in config_keys]

        err = False
        for res, key in zip(exist_keys, config_keys):
            if not res:
                err = True
                print(f"assert_config > key not exist: {key}")

        assert not err, "WorldFeature::assert_config - some keys not exist."

    def f0_and_timeaxis(self) -> tuple[np.ndarray, np.ndarray]:
        if self.f0 is None or self.timeaxis is None:
            sr = self.config["sampling_rate"]
            frame_period = self.config["frame_period"]

            f0, timeaxis = pyworld.dio(self.waveform, sr, frame_period=frame_period)
            self.f0 = f0
            self.timeaxis = timeaxis

        assert self.f0 is not None
        assert self.timeaxis is not None

        return (self.f0, self.timeaxis)

    def continuous_logarithmic_f0(self) -> np.ndarray:
        if self.lf0 is None:
            f0, _ = self.f0_and_timeaxis()

            f0[f0 < 0] = 0
            lf0 = f0.copy()
            nonzero = np.nonzero(f0)
            lf0[nonzero] = np.log(f0[nonzero])
            clf0 = self.__interp1d(lf0)
            self.lf0 = clf0.reshape((-1, 1))

        return self.lf0

    def vuv_flag(self) -> np.ndarray:
        if self.vuv is None:
            f0, _ = self.f0_and_timeaxis()
            vuv = (f0 > 0).astype(np.float64)
            self.vuv = vuv.reshape((-1, 1))

        return self.vuv

    def spectrogram(self) -> np.ndarray:
        if self.spec is None:
            sr = self.config["sampling_rate"]

            f0, timeaxis = self.f0_and_timeaxis()
            self.spec = pyworld.cheaptrick(self.waveform, f0, timeaxis, sr)

        return self.spec

    def mel_gemeralized_cepstrum(self) -> np.ndarray:
        if self.mgc is None:
            sr = self.config["sampling_rate"]
            mgc_order = self.config["mgc_order"]
            alpha = pysptk.util.mcepalpha(sr)

            spec = self.spectrogram()
            self.mgc = pysptk.sp2mc(spec, mgc_order, alpha)

        return self.mgc

    def band_aperiodicity(self) -> np.ndarray:
        if self.bap is None:
            sr = self.config["sampling_rate"]

            f0, timeaxis = self.f0_and_timeaxis()
            ap = pyworld.d4c(self.waveform, f0, timeaxis, sr)
            self.bap = pyworld.code_aperiodicity(ap, sr)

        return self.bap

    def to_dict(self) -> dict[str, FeatureObject]:
        return {
            "lf0": self.continuous_logarithmic_f0(),
            "vuv": self.vuv_flag(),
            "mgc": self.mel_gemeralized_cepstrum(),
            "bap": self.band_aperiodicity(),
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
