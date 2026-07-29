"""Differential IK helpers for the Franka Panda hand."""

from __future__ import annotations

from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab_pcap.robots.franka import (
  FRANKA_PANDA_ARM_JOINTS,
  FRANKA_PANDA_HAND_FRAME,
)

DEFAULT_IK_ITERATIONS = 10


def build_panda_ik_action_cfg() -> DifferentialIKActionCfg:
  """Build the default absolute-pose IK config used by the viewer demo."""

  return DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=FRANKA_PANDA_ARM_JOINTS,
    frame_name=FRANKA_PANDA_HAND_FRAME,
    frame_type="body",
    posture_weight=0.05,
    joint_limit_weight=0.1,
    damping=0.1,
    use_relative_mode=False,
  )


def build_panda_ik_action(env: object) -> DifferentialIKAction:
  """Build the default absolute-pose IK action used by the viewer demo."""

  cfg = build_panda_ik_action_cfg()
  return cfg.build(env)  # type: ignore[arg-type]


def get_frame_pose(
  ik_action: DifferentialIKAction,
  env_id: int = 0,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
  """Return the IK frame pose as Python tuples for viser controls."""

  pos, quat = ik_action._get_frame_pose()
  pos_np = pos[env_id].cpu().numpy()
  quat_np = quat[env_id].cpu().numpy()
  return (
    (float(pos_np[0]), float(pos_np[1]), float(pos_np[2])),
    (float(quat_np[0]), float(quat_np[1]), float(quat_np[2]), float(quat_np[3])),
  )
