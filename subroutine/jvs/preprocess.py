import dataclasses
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from subroutine.feature.speech import LibrosaFeature, WorldFeature
from subroutine.jvs.path import jvspath, jvsspkr, jvsuttr
from subroutine.util.file import dump_h5, dump_yaml, load_wav
from subroutine.util.timestamp import now_datetime_text

# import logging


@dataclasses.dataclass
class Preprocessor(object):
    input_dir: Path
    output_dir: Path
    config: dict[str, Any]

    info: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)

    feat_config: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)
    process_config: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)

    n_job: int = dataclasses.field(default=-1, init=False)
    speakers: list[int] = dataclasses.field(default_factory=list, init=False)
    utterances: list[int] = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self):
        self.ts = now_datetime_text()

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

        self.info["general"] = {
            "speakers": str(self.process_config["speakers"]),
            "utterances": str(self.process_config["utterances"]),
        }

        if "sampling_rate" in self.process_config:
            _sr = self.process_config["sampling_rate"]
            if type(_sr) == int:
                self.sampling_rate = _sr
            else:
                raise RuntimeError("sampling_rate must be int typed.")
        else:
            raise RuntimeError("Not found sampling_rate in config YAML.")

    def exec(self):
        results = []

        for n_spkr in tqdm(self.speakers):
            results.append(self.exec_spkr(n_spkr))

        for key, n_uttrs, path in results:
            self.info[key] = {"n_uttrs": n_uttrs, "path": str(path)}

    def exec_spkr(self, spkrnum: int) -> tuple[str, int, Path]:
        feats = {}
        stats = None

        # results = Parallel(n_jobs=self.n_job)(
        #     delayed(Preprocessor.exec_uttr)(self, spkrnum, n_uttr)
        #     for n_uttr in self.utterances
        # )

        results = []
        for n_uttr in self.utterances:
            res = self.exec_uttr(spkrnum, n_uttr)
            results.append(res)

        n_uttrs = 0

        for key, value in results:
            if value is not None:
                feats[key] = value
                stats = self._update_stats(stats, value)
                n_uttrs += 1

        feats["statistics"] = self._calc_stats(stats)
        path = self.dump_spkr(spkrnum, feats)

        return jvsspkr(spkrnum), n_uttrs, path

    def _update_stats(self, stack, data):
        for key_corp in data:
            for key_feat in data[key_corp]:
                if stack is None:
                    stack = {}
                if key_corp not in stack:
                    stack[key_corp] = {}
                if key_feat not in stack[key_corp]:
                    stack[key_corp][key_feat] = None

                if stack[key_corp][key_feat] is None:
                    stack[key_corp][key_feat] = data[key_corp][key_feat].copy()
                else:
                    stack[key_corp][key_feat] = np.concatenate(
                        [stack[key_corp][key_feat], data[key_corp][key_feat]], axis=1
                    )

        return stack

    def _calc_stats(self, stack):
        stats = {}

        for key_corp in stack:
            if key_corp not in stats:
                stats[key_corp] = {}

            for key_feat in stack[key_corp]:
                mean = np.mean(stack[key_corp][key_feat], axis=1)
                var = np.var(stack[key_corp][key_feat], axis=1)

                stats[key_corp][key_feat] = np.stack([mean, var])

        return stats

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
        self.info = {}

    def dump(self) -> Path:
        return dump_yaml(
            self.output_dir / f"stats-jvs-preprocess-{self.ts}.yml", self.info
        )

    def dump_spkr(self, spkr, data):
        spkrdir = jvsspkr(spkr)
        os.makedirs(self.output_dir / "spkrs" / spkrdir, exist_ok=True)

        h5path = dump_h5(
            self.output_dir / "spkrs" / spkrdir / f"jvs-preprocess-{self.ts}.hdf5",
            **data,
            overwrite=False,
        )

        return h5path
