"""Interactive Franka Panda IK viewer for PCAP-style tree scenes."""

from __future__ import annotations

import time

import torch
import viser

from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.viewer.viser import ViserMujocoScene
from mjlab_pcap.controllers.panda_ik import (
  DEFAULT_IK_ITERATIONS,
  build_panda_ik_action_cfg,
  get_frame_pose,
)
from mjlab_pcap.robots.franka import (
  FRANKA_PANDA_ARM_JOINTS,
  FRANKA_PANDA_GRIPPER_JOINTS,
)
from mjlab_pcap.scenes.tree_scene import (
  FrankaTreeSceneConfig,
  build_franka_tree_scene,
)


def run_panda_tree_ik_demo(config: FrankaTreeSceneConfig | None = None) -> None:
  """Launch the interactive Franka/tree IK control demo."""

  demo = build_franka_tree_scene(config)
  scene = demo.scene
  sim = demo.sim
  env = demo.env
  robot = demo.robot
  device = demo.device
  num_envs = demo.config.num_envs

  default_joint_pos = robot.data.default_joint_pos.to(
    device=device, dtype=robot.data.joint_pos.dtype
  )
  ik_cfg = build_panda_ik_action_cfg()
  ik_action = ik_cfg.build(env)  # type: ignore[arg-type]
  joint_ids = ik_action._joint_ids

  grip_ids, _ = robot.find_joints(FRANKA_PANDA_GRIPPER_JOINTS)
  grip_joint_ids = torch.tensor(grip_ids, device=device, dtype=torch.long)
  grip_open = default_joint_pos[:, grip_joint_ids].clone()
  zero_joint_vel = torch.zeros((num_envs, len(FRANKA_PANDA_ARM_JOINTS)), device=device)
  physics_dt = sim.cfg.mujoco.timestep

  server = viser.ViserServer(label="Panda IK Control Demo")
  viewer_scene = ViserMujocoScene(server, sim.mj_model, num_envs=num_envs)
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
  pos, quat = get_frame_pose(ik_action, selected_env)
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
      initial_value=DEFAULT_IK_ITERATIONS,
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

  target_action = torch.zeros(num_envs, 7, device=device)

  def _reset() -> None:
    sim.reset()
    reset_scene_to_default(env, None)
    robot.write_joint_velocity_to_sim(zero_joint_vel, joint_ids=joint_ids)
    env.scene.reset()
    sim.forward()
    ik_action.reset()
    pos, quat = get_frame_pose(ik_action, viewer_scene.env_idx)
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
        pos, quat = get_frame_pose(ik_action, selected_env)
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
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=physics_dt)
        sim.forward()

      viewer_scene.update(sim.data)
      if viewer_scene.needs_update:
        viewer_scene.refresh_visualization()

      time.sleep(1.0 / 30.0)
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()
