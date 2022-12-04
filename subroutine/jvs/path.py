from pathlib import Path
from typing import Union


def jvsspkr(number: Union[int, str]) -> str:
    number = int(number)
    assert number >= 1 and number <= 100, f"spkr number is out of bounds: {number}"

    return f"jvs{number:03}"


def jvsuttr(number: Union[int, str]) -> str:
    number = int(number)
    assert number >= 1 and number <= 100, f"uttr number is out of bounds: {number}"

    return f"VOICEACTRESS100_{number:03}"


def jvspath(
    jvsdir: Union[str, Path], spkrnum: Union[int, str], uttrnum: Union[int, str]
) -> Path:
    spkr, uttr = jvsspkr(spkrnum), jvsuttr(uttrnum)
    path = Path(f"{jvsdir}/jvs_ver1/{spkr}/parallel100/wav24kHz16bit/{uttr}.wav")
    if not path.exists():
        print(f"[warning] jvs wav file not exist: {path}")

    return path
