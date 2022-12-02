import os
from pathlib import Path

from setuptools import setup

from subroutine import __VERSION__


def _requires_from_file(filename):
    return open(filename).read().splitlines()


def __find_rec_packages():
    directory = "./subroutine"
    package_dir = Path(directory)
    assert package_dir.exists()

    for root, dirs, _ in os.walk(package_dir):
        for dir in dirs:
            if "__pycache__" in dir:
                continue

            yield os.path.join(root, dir)


def _find_rec_packages():
    return list(__find_rec_packages())


setup(
    name="kurage-subroutine",
    version=__VERSION__,
    description="My Subroutine exporting modules.",
    author="jellyfish_rumble",
    packages=_find_rec_packages(),
    install_requires=_requires_from_file("requirements.txt"),
)
