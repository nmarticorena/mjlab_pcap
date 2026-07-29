"""PCAP-style tree asset loading for mjlab scenes."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TREE_ASSET = PROJECT_ROOT / "pcap_example.urdf"


def get_tree_cfg(asset_path: str | Path = DEFAULT_TREE_ASSET) -> EntityCfg:
  """Build an mjlab entity config for a MuJoCo-loadable tree asset."""

  path = Path(asset_path).expanduser().resolve()
  return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_file(str(path)))

