"""Regenerate the demo catalog in a temporary folder and compare MIDI checksums with data/demo.

    python scripts/verify_demo_checksums.py
"""

import json
import sys
import tempfile
from pathlib import Path

from melodyai.config import data_dir
from melodyai.demo.build import build_demo_files

with tempfile.TemporaryDirectory() as d:
    build_demo_files(Path(d), with_audio=False)
    fresh = json.loads((Path(d) / "checksums.json").read_text())
stored = json.loads((data_dir() / "demo" / "checksums.json").read_text())
diff = [k for k in fresh if fresh[k] != stored.get(k)]
print(f"{len(fresh)} songs, {len(diff)} mismatches")
sys.exit(1 if diff else 0)
