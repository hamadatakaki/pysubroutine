import os
import unittest
from pathlib import Path

from subroutine.jvs import path


class TestPath(unittest.TestCase):
    def test_jvspath(self):
        jvsdir = Path(os.environ["HOME"]) / "dataset/speech"
        p = path.jvspath(jvsdir, 1, 1)
        self.assertTrue(p.exists())
