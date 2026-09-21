# Local Windows Reproduction

This fork combines Kohya GUI upstream v26 work with locally qualified Windows training changes for an NVIDIA RTX 3090 Ti.

## Integration Sources

- Root upstream base: `45088f0`
- Root integration branch: `integration/upstream-2026-09-21`
- Preserved pre-integration branch: `preserve/pre-upstream-2026-09-21` at `b226446`
- `sd-scripts` upstream base: `6721028` (`v0.11.1`)
- `sd-scripts` integration branch: `integration/max-norm-upstream-2026-09-21` at `06c659c`
- Preserved original optimization: `preserve/max-norm-2026-09-21` at `1960156`

The preserved branch contains the original local presets and machine-specific scripts. Those files are not copied into the publishable integration branch because they contain personal names and absolute paths.

## Qualified Runtime

- Windows 10/11 x64
- Python 3.10.11 x64
- NVIDIA RTX 3090 Ti with 24 GB VRAM
- Torch `2.7.0+cu128`
- torchvision `0.22.0+cu128`
- xFormers `0.0.30`
- CUDA runtime `12.8` supplied by Torch
- cuDNN `9.7.1` supplied by Torch

Do not copy toolkit or cuDNN DLLs into the virtual environment. Do not add the obsolete `torchaudio 2.1.0+cu118`, `nvidia-cudnn-cu11`, or legacy Triton 2.1.0 packages to this environment.

## Installation

Clone recursively into a dedicated location and install through its local virtual environment:

```powershell
git clone --recurse-submodules https://github.com/JamesWClark/kohya_ss.git E:\Forge\kohya_ss
Set-Location E:\Forge\kohya_ss
py -3.10 -m venv venv
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\python.exe -m pip install -r requirements_pytorch_windows.txt
```

Start the GUI with `gui.bat`. It selects the checkout's Python directly, clears inherited Python path variables, and does not install or upgrade packages.

Training tabs invoke Accelerate through the active interpreter:

```text
<local-python> -m accelerate.commands.launch
```

This avoids stale `accelerate.exe` wrappers after moving or copying a virtual environment.

## Local Optimization

The LoRA max-norm implementation keeps ratio calculation and adapter scaling on the device, reuses the initial norm, and transfers only the final statistics to the host. The preserved equivalence regression covers FP32, FP16, and BF16 with linear, 1x1, and 3x3 adapters. The original synthetic rank-256 CUDA benchmark improved this phase from 10.316 ms to 8.138 ms.

## Validation

```powershell
& .\venv\Scripts\python.exe -m pip check
& .\venv\Scripts\python.exe -B -c "import torch, xformers; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), xformers.__version__)"
& .\venv\Scripts\python.exe .\test\test_active_python_launch.py -v
& .\venv\Scripts\python.exe .\test\test_max_norm_candidate.py -v
```

For training qualification, use the same model, dataset, buckets, preset, seed, and GPU workload. Compare warm steady-state step time, peak VRAM, loss behavior, generated output, and resume behavior.

## Publishing the Submodule

The optimized `sd-scripts` commit is published at `JamesWClark/kohya-sd-scripts` on branch `integration/max-norm-upstream-2026-09-21`. The root repository records that fork in `.gitmodules` and pins commit `06c659c`.

After changing the submodule, publish its commit before updating and pushing the root gitlink. Verify every release with a fresh recursive clone.

The original upstream remains `https://github.com/kohya-ss/sd-scripts.git`; use it when fetching future upstream updates into the submodule fork.