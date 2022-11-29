from pathlib import Path
from typing import Union


def jvsspkr(number: int) -> str:
    assert number >= 1 and number <= 100, f"spkr number is out of bounds: {number}"
    return f"jvs{number:03}"


def jvsuttr(number: int) -> str:
    assert number >= 1 and number <= 100, f"uttr number is out of bounds: {number}"
    return f"VOICEACTRESS100_{number:03}"


def jvspath(jvsdir: Union[str, Path], spkrnum: int, uttrnum: int) -> Path:
    spkr = jvsspkr(spkrnum)
    uttr = jvsuttr(uttrnum)

    path = Path(f"{jvsdir}/jvs_ver1/{spkr}/parallel100/wav24kHz16bit/{uttr}.wav")
    assert path.exists(), f"jvs wav file not exist: {path}"
    return path
