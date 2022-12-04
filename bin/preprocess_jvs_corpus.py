import argparse
import sys
from pathlib import Path

sys.path.append(".")

from subroutine.jvs.preprocess import Preprocessor  # noqa: E402
from subroutine.util.file import load_yaml  # noqa: E402

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

    prep = Preprocessor(input_dir, output_dir, config)
    prep.exec()
    prep.dump()
