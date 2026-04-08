"""Conftest: mock heavy dependencies so tests run without torch/GPU."""

from __future__ import annotations

import sys
from unittest import mock

# Stub out heavy or hardware-specific modules before any deploy.* imports.
# These are only needed at import time; the actual test logic mocks deeper.
_STUB_MODULES = [
    "torch",
    "torch.multiprocessing",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.parallel",
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "tqdm",
    "transformers",
    "piper_sdk",
]

for mod_name in _STUB_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock.MagicMock()
