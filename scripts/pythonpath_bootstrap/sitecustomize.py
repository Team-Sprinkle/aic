"""Optional Python path bootstrap for rootless AIC runtime smoke tests.

Pixi may prepend its installed site-packages before PYTHONPATH. Set
``AIC_CHECKOUT_PYTHONPATH`` to a path-separated list of checkout package roots
to force those paths ahead of installed package copies.
"""

from __future__ import annotations

import os
import sys


paths = [
    path
    for path in os.environ.get("AIC_CHECKOUT_PYTHONPATH", "").split(os.pathsep)
    if path
]
for path in reversed(paths):
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
