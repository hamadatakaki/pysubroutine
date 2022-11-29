import unittest
from subroutine.jvs import path
from pathlib import Path
import os


class TestPath(unittest.TestCase):
    def test_jvspath(self):
        jvsdir = Path(os.environ["HOME"]) / "dataset/speech"
        p = path.jvspath(jvsdir, 1, 1)
        self.assertTrue(p.exists())
