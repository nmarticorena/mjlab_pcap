"""

Drag the 3D transform control in the viser viewer to move the Panda hand.
The demo also spawns a short passive capsule-chain "rope" that the robot can
bump into. Contact points and contact forces are enabled in the viewer.

Run with:
  MJLAB_WARP_QUIET=1 pixi run python scripts/panda_ik.py
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import mujoco
import torch
import viser
from robot_descriptions import panda_mj_description

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.viewer.viser import ViserMujocoScene

PANDA_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
PANDA_GRIPPER_JOINTS = ("finger_joint1", "finger_joint2")
ENV_SPACING = 2.0
IK_ITERATIONS = 10
ROPE_LINKS = 6
ROPE_LINK_LENGTH = 0.09
ROPE_RADIUS = 0.012

PANDA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=PANDA_ARM_JOINTS),),
  soft_joint_pos_limit_factor=0.95,
)

DEMO_INIT_STATE = EntityCfg.InitialStateCfg(
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


def get_panda_robot_cfg() -> EntityCfg:
  def _spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    while spec.keys:
      spec.delete(spec.keys[0])
    return spec

  return EntityCfg(
    init_state=DEMO_INIT_STATE,
    spec_fn=_spec_fn,
    articulation=PANDA_ARTICULATION,
  )


def get_rope_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_file("pcap_example.urdf"))


def _make_grid_origins(num_envs: int, spacing: float, device: str) -> torch.Tensor:
  cols = math.ceil(math.sqrt(num_envs))
  origins = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
  for env_id in range(num_envs):
    row, col = divmod(env_id, cols)
    origins[env_id, 0] = col * spacing
    origins[env_id, 1] = row * spacing
  return origins


def _get_frame_pose(
  ik_action: DifferentialIKAction,
  env_id: int = 0,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
  pos, quat = ik_action._get_frame_pose()
  pos_np = pos[env_id].cpu().numpy()
  quat_np = quat[env_id].cpu().numpy()
  return (
    (float(pos_np[0]), float(pos_np[1]), float(pos_np[2])),
    (float(quat_np[0]), float(quat_np[1]), float(quat_np[2]), float(quat_np[3])),
  )


def main(n_envs:int = 1) -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  sim_scene = Scene(
    SceneCfg(
      num_envs=n_envs,
      env_spacing=ENV_SPACING,
      extent=1.6,
      entities={"robot": get_panda_robot_cfg(), "rope": get_rope_cfg()},
    ),
    device=device,
  )
  model = sim_scene.compile()
  sim_cfg = SimulationCfg(
    nconmax=512,
    njmax=1024 * 2,
    mujoco=MujocoCfg(gravity=(0, 0, -9.81)),
  )
  sim = Simulation(num_envs=n_envs, cfg=sim_cfg, model=model, device=device)
  sim_scene.initialize(sim.mj_model, sim.model, sim.data)
  sim_scene._default_env_origins = _make_grid_origins(n_envs, ENV_SPACING, device)

  robot: Entity = sim_scene["robot"]
  default_joint_pos = robot.data.default_joint_pos.to(
    device=device, dtype=robot.data.joint_pos.dtype
  )
  env = SimpleNamespace(num_envs=n_envs, device=device, scene=sim_scene, sim=sim)
  reset_scene_to_default(env, None)
  sim.forward()

  ik_cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=PANDA_ARM_JOINTS,
    frame_name="hand",
    frame_type="body",
    posture_weight=0.05,
    joint_limit_weight=0.1,
    damping=0.1,
    use_relative_mode=False,
  )
  ik_action: DifferentialIKAction = ik_cfg.build(env)  # type: ignore[arg-type]
  joint_ids = ik_action._joint_ids

  grip_ids, _ = robot.find_joints(PANDA_GRIPPER_JOINTS)
  grip_joint_ids = torch.tensor(grip_ids, device=device, dtype=torch.long)
  grip_open = default_joint_pos[:, grip_joint_ids].clone()
  zero_joint_vel = torch.zeros((n_envs, len(PANDA_ARM_JOINTS)), device=device)
  physics_dt = sim.cfg.mujoco.timestep

  server = viser.ViserServer(label="Panda IK Control Demo")
  viewer_scene = ViserMujocoScene(server, sim.mj_model, num_envs=n_envs)
  viewer_scene.camera_tracking_enabled = False
  viewer_scene.show_only_selected = True
  viewer_scene.show_contact_points = True
  viewer_scene.show_contact_forces = False
  viewer_scene.create_scene_gui(
    camera_distance=3.0,
    camera_azimuth=150.0,
    camera_elevation=18.0,
  )

  selected_env = viewer_scene.env_idx
  pos, quat = _get_frame_pose(ik_action, selected_env)
  transform_ctrl = server.scene.add_transform_controls(
    "/ik_target",
    position=pos,
    wxyz=quat,
    scale=0.18,
  )

  needs_reset = [False]

  with server.gui.add_folder("IK Control"):
    reset_button = server.gui.add_button("Reset")
    reset_button.on_click(lambda _: needs_reset.__setitem__(0, True))
    iterations_slider = server.gui.add_slider(
      "IK Iterations",
      min=1,
      max=50,
      step=1,
      initial_value=IK_ITERATIONS,
    )

  with server.gui.add_folder("IK Weights"):
    damping_slider = server.gui.add_slider(
      "Damping (lambda)",
      min=1e-2,
      max=1.0,
      step=1e-3,
      initial_value=ik_cfg.damping,
    )
    pos_w_slider = server.gui.add_slider(
      "Position Weight",
      min=0.0,
      max=10.0,
      step=0.1,
      initial_value=ik_cfg.position_weight,
    )
    ori_w_slider = server.gui.add_slider(
      "Orientation Weight",
      min=0.0,
      max=10.0,
      step=0.1,
      initial_value=ik_cfg.orientation_weight,
    )
    jlim_w_slider = server.gui.add_slider(
      "Joint Limit Weight",
      min=0.0,
      max=1.0,
      step=0.01,
      initial_value=ik_cfg.joint_limit_weight,
    )
    posture_w_slider = server.gui.add_slider(
      "Posture Weight",
      min=0.0,
      max=1.0,
      step=0.01,
      initial_value=ik_cfg.posture_weight,
    )

  print("=" * 60)
  print("Panda IK Control Demo")
  print("  Open the viser URL printed above")
  print("  Drag the 3D transform control to move the Panda hand")
  print("  Use the Environment selector to choose the controlled sim")
  print("  Contact points and forces are enabled")
  print("=" * 60)

  target_action = torch.zeros(n_envs, 7, device=device)

  def _reset() -> None:
    sim.reset()
    reset_scene_to_default(env, None)
    robot.write_joint_velocity_to_sim(zero_joint_vel, joint_ids=joint_ids)
    env.scene.reset()
    sim.forward()
    ik_action.reset()
    pos, quat = _get_frame_pose(ik_action, viewer_scene.env_idx)
    transform_ctrl.position = pos
    transform_ctrl.wxyz = quat

  try:
    while True:
      if needs_reset[0]:
        needs_reset[0] = False
        _reset()

      ik_cfg.damping = max(damping_slider.value, 1e-2)
      ik_cfg.position_weight = max(pos_w_slider.value, 0.0)
      ik_cfg.orientation_weight = max(ori_w_slider.value, 0.0)
      ik_cfg.joint_limit_weight = max(jlim_w_slider.value, 0.0)
      ik_cfg.posture_weight = max(posture_w_slider.value, 0.0)

      if viewer_scene.env_idx != selected_env:
        selected_env = viewer_scene.env_idx
        pos, quat = _get_frame_pose(ik_action, selected_env)
        transform_ctrl.position = pos
        transform_ctrl.wxyz = quat

      frame_pos, frame_quat = ik_action._get_frame_pose()
      target_action[:, :3] = frame_pos
      target_action[:, 3:] = frame_quat
      p = transform_ctrl.position
      w = transform_ctrl.wxyz
      target_action[selected_env, :3] = torch.tensor(
        (p[0], p[1], p[2]), device=device
      )
      target_action[selected_env, 3:] = torch.tensor(
        (w[0], w[1], w[2], w[3]), device=device
      )
      ik_action.process_actions(target_action)

      for _ in range(int(iterations_slider.value)):
        dq = ik_action.compute_dq()
        q_target = robot.data.joint_pos[:, joint_ids].clone()
        q_target[selected_env] += dq[selected_env]
        robot.set_joint_position_target(q_target, joint_ids=joint_ids)
        robot.set_joint_position_target(grip_open, joint_ids=grip_joint_ids)
        sim_scene.write_data_to_sim()
        sim.step()
        sim_scene.update(dt=physics_dt)
        sim.forward()

      viewer_scene.update(sim.data)
      if viewer_scene.needs_update:
        viewer_scene.refresh_visualization()

      time.sleep(1.0 / 30.0)
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()


if __name__ == "__main__":
  import tyro
  tyro.cli(main)
