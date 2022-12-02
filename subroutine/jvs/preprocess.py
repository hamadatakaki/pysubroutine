import argparse
import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

from subroutine.feature.speech import LibrosaFeature, WorldFeature
from subroutine.jvs.path import jvspath
from subroutine.util.file import dump_h5, load_wav, load_yaml
from subroutine.util.timestamp import now_datetime_text


@dataclasses.dataclass
class Preprocessor(object):
    input_dir: Path
    output_dir: Path
    config: dict[str, Any]

    features: dict[str, Any] = dataclasses.field(default={}, init=False)

    feat_config: dict[str, Any] = dataclasses.field(default={}, init=False)
    process_config: dict[str, Any] = dataclasses.field(default={}, init=False)

    n_job: int = dataclasses.field(default=1, init=False)
    speakers: list[int] = dataclasses.field(default=[], init=False)
    utterances: list[int] = dataclasses.field(default=[], init=False)

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

    def exec(self):
        pass

    def reset(self):
        self.features = {}

    def dump(self) -> Path:
        ts = now_datetime_text()
        name = f"jvs-preprocess-{ts}.hdf5"
        return dump_h5(self.output_dir / name, **self.features, overwrite=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features of JVS corpus.")

    parser.add_argument(
        "-i", "--input_dir", required=True, help="jvs_ver1 が設置されているディレクトリ"
    )
    parser.add_argument("-o", "--output_dir", required=True, help="特徴量を保存するディレクトリ")
    parser.add_argument("-c", "--config", required=True, help="設定を記述したyamlのパス")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config = load_yaml(args.config)

    Preprocessor(input_dir, output_dir, config).exec()
