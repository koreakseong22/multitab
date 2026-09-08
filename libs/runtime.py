import hashlib
from pathlib import Path

import torch


def configure_device(gpu_id):
    if gpu_id < -1:
        raise ValueError("gpu_id must be -1 (CPU) or a visible CUDA device index")
    if gpu_id == -1 or not torch.cuda.is_available():
        return torch.device('cpu')
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {gpu_id} is not visible; use --gpu_id 0 or --gpu_id -1")
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    return device


def implementation_id():
    root = Path(__file__).resolve().parent.parent
    paths = list(root.glob('*.py')) + list((root / 'libs').glob('*.py'))
    paths += [root / 'dataset_id.json', root / 'requirements.txt']
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def check_implementation(metadata, expected, allow_legacy=False):
    actual = metadata.get('implementation_id')
    if actual is None and allow_legacy:
        return
    if actual != expected:
        raise ValueError(
            "The saved result was produced by a different or unrecorded implementation. "
            "Use a fresh --savepath. For verified official logs without metadata only, "
            "use --allow_legacy_logs."
        )


