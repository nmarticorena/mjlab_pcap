"""Franka Panda and PCAP-style tree scene builders."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab_pcap.robots.franka import get_franka_panda_cfg
from mjlab_pcap.trees.assets import DEFAULT_TREE_ASSET, get_tree_cfg


@dataclass(frozen=True)
class FrankaTreeSceneConfig:
  """Configuration for a vectorized Franka/tree scene."""

  num_envs: int = 20
  env_spacing: float = 2.0
  extent: float = 1.6
  tree_asset: Path = DEFAULT_TREE_ASSET
  gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
  nconmax: int = 512
  njmax: int = 2048


@dataclass
class FrankaTreeScene:
  """Compiled scene plus the lightweight env object expected by mjlab actions."""

  scene: Scene
  sim: Simulation
  env: SimpleNamespace
  robot: Entity
  device: str
  config: FrankaTreeSceneConfig


def make_grid_origins(num_envs: int, spacing: float, device: str) -> torch.Tensor:
  """Create deterministic grid origins for vectorized scenes."""

  cols = math.ceil(math.sqrt(num_envs))
  origins = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
  for env_id in range(num_envs):
    row, col = divmod(env_id, cols)
    origins[env_id, 0] = col * spacing
    origins[env_id, 1] = row * spacing
  return origins


def build_franka_tree_scene(
  config: FrankaTreeSceneConfig | None = None,
  *,
  device: str | None = None,
) -> FrankaTreeScene:
  """Compile and initialize a vectorized Franka/tree scene."""

  cfg = config or FrankaTreeSceneConfig()
  sim_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  scene = Scene(
    SceneCfg(
      num_envs=cfg.num_envs,
      env_spacing=cfg.env_spacing,
      extent=cfg.extent,
      entities={
        "robot": get_franka_panda_cfg(),
        "tree": get_tree_cfg(cfg.tree_asset),
      },
    ),
    device=sim_device,
  )
  model = scene.compile()
  sim_cfg = SimulationCfg(
    nconmax=cfg.nconmax,
    njmax=cfg.njmax,
    mujoco=MujocoCfg(gravity=cfg.gravity),
  )
  sim = Simulation(num_envs=cfg.num_envs, cfg=sim_cfg, model=model, device=sim_device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  scene._default_env_origins = make_grid_origins(
    cfg.num_envs, cfg.env_spacing, sim_device
  )

  env = SimpleNamespace(num_envs=cfg.num_envs, device=sim_device, scene=scene, sim=sim)
  reset_scene_to_default(env, None)
  sim.forward()

  return FrankaTreeScene(
    scene=scene,
    sim=sim,
    env=env,
    robot=scene["robot"],
    device=sim_device,
    config=cfg,
  )

