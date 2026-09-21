import ast
from pathlib import Path
import unittest


GUI_MODULES = (
    "anima_lllite_gui.py",
    "dreambooth_gui.py",
    "finetune_gui.py",
    "leco_gui.py",
    "lora_gui.py",
    "textual_inversion_gui.py",
)


class ActivePythonLaunchTest(unittest.TestCase):
    def test_training_tabs_launch_accelerate_with_active_python(self):
        gui_root = Path(__file__).resolve().parents[1] / "kohya_gui"
        expected = ["sys.executable", "'-m'", "'accelerate.commands.launch'"]

        for module_name in GUI_MODULES:
            with self.subTest(module=module_name):
                source = (gui_root / module_name).read_text(encoding="utf-8")
                tree = ast.parse(source)
                launchers = []
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                        continue
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id == "run_cmd" and isinstance(node.value, ast.List):
                        launchers.append([ast.unparse(item) for item in node.value.elts])

                self.assertIn(expected, launchers)
                self.assertNotIn('get_executable_path("accelerate")', source)


if __name__ == "__main__":
    unittest.main()