"""Deterministic tree asset generation entrypoints.

The current implementation preserves the checked-in PCAP example as a deterministic
fixture. It gives the project a stable command surface while the full L-system
generator is ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from mjlab_pcap.trees.assets import DEFAULT_TREE_ASSET, PROJECT_ROOT


@dataclass(frozen=True)
class TreeGenerationConfig:
  """Configuration for generating a PCAP-style tree asset."""

  seed: int = 0
  source_asset: Path = DEFAULT_TREE_ASSET
  output_dir: Path = PROJECT_ROOT / "generated" / "trees"
  output_name: str | None = None

  @property
  def output_path(self) -> Path:
    name = self.output_name or f"pcap_tree_seed_{self.seed}.urdf"
    return self.output_dir / name


def generate_tree_asset(config: TreeGenerationConfig) -> Path:
  """Generate a deterministic tree asset and return its path.
  """

  source = config.source_asset.expanduser().resolve()
  output = config.output_path.expanduser().resolve()
  if not source.exists():
    raise FileNotFoundError(f"Tree source asset does not exist: {source}")

  output.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(source, output)
  return output

