import dataclasses
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from joblib import Parallel, delayed

from subroutine.feature.speech import LibrosaFeature, WorldFeature
from subroutine.jvs.path import jvspath, jvsspkr, jvsuttr
from subroutine.util.file import dump_h5, load_wav
from subroutine.util.timestamp import now_datetime_text

# import logging


@dataclasses.dataclass
class Preprocessor(object):
    input_dir: Path
    output_dir: Path
    config: dict[str, Any]

    features: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)
    stats: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)

    feat_config: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)
    process_config: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)

    n_job: int = dataclasses.field(default=-1, init=False)
    speakers: list[int] = dataclasses.field(default_factory=list, init=False)
    utterances: list[int] = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self):
        # check input_dir
        corpus_path = self.input_dir / "jvs_ver1"
        assert corpus_path.exists(), f"Corpus is not exist: {corpus_path}"

        # create output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # check config
        self.feat_config = self.config["feature"]
        self.process_config = self.config["preprocess"]
        self.preprocess_configures()

    def preprocess_configures(self):
        # n_job: parallel job number
        if "n_job" in self.process_config:
            self.n_job = self.process_config["n_job"]

        # speakers: speaker list
        if "speakers" in self.process_config:
            _speakers = self.process_config["speakers"]
            if type(_speakers) == list:
                self.speakers = _speakers
            elif type(_speakers) == str and _speakers == "all":
                self.speakers = list(range(1, 101))
            else:
                raise RuntimeError("Speakers setting is invalid.")

        # utterances: speaker list
        if "utterances" in self.process_config:
            _utterances = self.process_config["utterances"]
            if type(_utterances) == list:
                self.utterances = _utterances
            elif type(_utterances) == str and _utterances == "all":
                self.utterances = list(range(1, 101))
            else:
                raise RuntimeError("Utterances setting is invalid.")

        if "sampling_rate" in self.process_config:
            _sr = self.process_config["sampling_rate"]
            if type(_sr) == int:
                self.sampling_rate = _sr
            else:
                raise RuntimeError("sampling_rate must be int typed.")
        else:
            raise RuntimeError("Not found sampling_rate in config YAML.")

    def exec(self):
        results = Parallel(n_jobs=self.n_job)(
            delayed(Preprocessor.exec_spkr)(self, n_spkr) for n_spkr in self.speakers
        )

        for key, value in results:
            self.features[key] = value

        # TODO: stats info

        return self.features

    def exec_spkr(self, spkrnum: int) -> tuple[str, dict[str, Any]]:
        dict_spkr = {}

        results = Parallel(n_jobs=self.n_job)(
            delayed(Preprocessor.exec_uttr)(self, spkrnum, n_uttr)
            for n_uttr in self.utterances
        )

        for key, value in results:
            if value is not None:
                dict_spkr[key] = value

            # TODO: normalize

        return jvsspkr(spkrnum), dict_spkr

    def exec_uttr(
        self, spkrnum: int, uttrnum: int
    ) -> tuple[str, Optional[dict[str, Any]]]:
        waveform = self.load_uttr(spkrnum, uttrnum)

        if waveform is None:
            return jvsuttr(uttrnum), None

        librosa_feat = LibrosaFeature(self.feat_config["librosa"]).get_features(
            waveform
        )
        world_feat = WorldFeature(self.feat_config["world"]).get_features(waveform)

        return jvsuttr(uttrnum), {"librosa": librosa_feat, "world": world_feat}

    def load_uttr(self, spkrnum: int, uttrnum: int) -> Optional[np.ndarray]:
        path = jvspath(self.input_dir, spkrnum, uttrnum)

        if path.exists():
            _, waveform = load_wav(path, sr=self.sampling_rate)
        else:
            waveform = None

        return waveform

    def reset(self):
        self.features = {}

    def dump(self) -> Path:
        ts = now_datetime_text()
        name = f"jvs-preprocess-{ts}.hdf5"
        return dump_h5(self.output_dir / name, **self.features, overwrite=False)
