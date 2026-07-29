"""Franka Panda entity configuration for mjlab scenes."""

from __future__ import annotations

import mujoco
from robot_descriptions import panda_mj_description

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

FRANKA_PANDA_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FRANKA_PANDA_GRIPPER_JOINTS = ("finger_joint1", "finger_joint2")
FRANKA_PANDA_HAND_FRAME = "hand"

FRANKA_PANDA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=FRANKA_PANDA_ARM_JOINTS),),
  soft_joint_pos_limit_factor=0.95,
)

FRANKA_PANDA_DEFAULT_STATE = EntityCfg.InitialStateCfg(
  pos=(-0.5, 0.5, 1.5),
  joint_pos={
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": -1.57079,
    "joint5": 0.0,
    "joint6": 1.57079,
    "joint7": -0.7853,
    "finger_joint1": 0.04,
    "finger_joint2": 0.04,
  },
  joint_vel={".*": 0.0},
)


def get_franka_panda_cfg() -> EntityCfg:
  """Build the Franka Panda mjlab entity config used by PCAP demos."""

  def _spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    while spec.keys:
      spec.delete(spec.keys[0])
    return spec

  return EntityCfg(
    init_state=FRANKA_PANDA_DEFAULT_STATE,
    spec_fn=_spec_fn,
    articulation=FRANKA_PANDA_ARTICULATION,
  )

