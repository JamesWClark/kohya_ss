import ast
from pathlib import Path
import unittest

import torch


def load_regularizer(root):
    tree = ast.parse((root / "sd-scripts/networks/lora.py").read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "apply_max_norm_regularization")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(root), "exec"), namespace)
    return namespace[method.name]


def reference_regularizer(network, max_norm_value, device):
    downkeys = []
    upkeys = []
    alphakeys = []
    norms = []
    keys_scaled = 0
    state_dict = network.state_dict()
    for key in state_dict:
        if "lora_down" in key and "weight" in key:
            downkeys.append(key)
            upkeys.append(key.replace("lora_down", "lora_up"))
            alphakeys.append(key.replace("lora_down.weight", "alpha"))
    for index in range(len(downkeys)):
        down = state_dict[downkeys[index]].to(device)
        up = state_dict[upkeys[index]].to(device)
        alpha = state_dict[alphakeys[index]].to(device)
        scale = alpha / down.shape[0]
        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
        elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
            updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
        else:
            updown = up @ down
        updown *= scale
        norm = updown.norm().clamp(min=max_norm_value / 2)
        desired = torch.clamp(norm, max=max_norm_value)
        ratio = desired.cpu() / norm.cpu()
        sqrt_ratio = ratio**0.5
        if ratio != 1:
            keys_scaled += 1
            state_dict[upkeys[index]] *= sqrt_ratio
            state_dict[downkeys[index]] *= sqrt_ratio
        norms.append((updown.norm() * ratio).item())
    return keys_scaled, sum(norms) / len(norms), max(norms)


class Network:
    def __init__(self, state):
        self.state = {key: value.clone() for key, value in state.items()}

    def state_dict(self):
        return self.state


class MaxNormTest(unittest.TestCase):
    def test_original_equivalence(self):
        root = Path(__file__).resolve().parents[1]
        candidate = load_regularizer(root)
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for magnitude in [0.0, 0.001, 0.05, 1.0]:
                    with self.subTest(device=device, dtype=dtype, magnitude=magnitude):
                        torch.manual_seed(42)
                        state = {}
                        shapes = [((8, 32), (24, 8)), ((8, 32, 1, 1), (24, 8, 1, 1)), ((8, 32, 3, 3), (24, 8, 1, 1))]
                        for index, (down_shape, up_shape) in enumerate(shapes):
                            state[f"{index}.lora_down.weight"] = torch.randn(down_shape, device=device, dtype=dtype) * magnitude
                            state[f"{index}.lora_up.weight"] = torch.randn(up_shape, device=device, dtype=dtype) * magnitude
                            state[f"{index}.alpha"] = torch.tensor(4.0, device=device)
                        baseline, changed = Network(state), Network(state)
                        for step in range(3):
                            expected = reference_regularizer(baseline, 1.0, device)
                            actual = candidate(changed, 1.0, device)
                            self.assertEqual(expected[0], actual[0])
                            tolerance = 0.02 if dtype == torch.bfloat16 else 0.002
                            for key in state:
                                torch.testing.assert_close(baseline.state[key], changed.state[key], rtol=tolerance, atol=0.0001)
                            torch.testing.assert_close(torch.tensor(expected[1:]), torch.tensor(actual[1:]), rtol=tolerance, atol=0.0001)


if __name__ == "__main__":
    unittest.main()