import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaugeguard.store import MemoryStore, load_assets  # noqa: E402
from gaugeguard.synth import SPECS, render_gauge  # noqa: E402

ASSET_ORDER = ["PG-101", "PG-202", "TG-303", "PG-404", "VG-505"]  # same order as synth.SPECS


@pytest.fixture
def store():
    return MemoryStore(load_assets())


@pytest.fixture
def gauge():
    """gauge(asset_id, value, seed=0) -> (image, truth) for a front-on synthetic gauge showing ``value``."""
    def make(asset_id, value, seed=0):
        return render_gauge(value, SPECS[ASSET_ORDER.index(asset_id)], rng=np.random.default_rng(seed))
    return make
