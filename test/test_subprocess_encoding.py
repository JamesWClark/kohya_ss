from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from kohya_gui.common_gui import setup_environment


class SubprocessEncodingTest(unittest.TestCase):
    def test_training_status_prints_as_utf8(self):
        with patch.dict("os.environ", {"PYTHONIOENCODING": "cp1252"}):
            env = setup_environment()
            result = subprocess.run(
                [sys.executable, "-c", "print('running training / 学習開始')"],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )

        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "running training / 学習開始")


if __name__ == "__main__":
    unittest.main()