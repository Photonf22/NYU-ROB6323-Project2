# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
import numpy as np
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
# new added import
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.1 # or change to 0.1 if still unstable, try 0.1 then back to 0.25
    action_space = 12
    # The Raibert Heuristic is a classic control strategy that places feet to stabilize velocity. 
    # We will use it as a "teacher" reward to encourage the policy to learn proper stepping. 
    # For reference logic, see https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py#L670
    observation_space = 48 + 4  #Define the reward scales and increase observation space to include clock inputs (4 phases).
    # TODO:(Alejandro, Sanchez) 12/16/2025 lowering the tracking contacts shape force reward scale from -10.0 to -0.5 to -2.0
    #raibert_heuristic_reward_scale = -10.0
    # When the policy tracks commands and moves, gradually tighten it back down (more negative) to clean up foot placement.
    # after moves and tracks, you can push it more negative (e.g., -3, -5, maybe back to -10).
    raibert_heuristic_reward_scale = -1.0  #decrease after back toward -5 … -10 to sharpen gait/placement
    # also increase contact shaping a bit if footfalls aren’t clean
    # 6.1 Update Configuration line 36 and line 37
    # TODO:(Alejandro, Sanchez) 12/16/2025 lowering the reward scale from -30.0 to -5.0 to start with
    feet_clearance_reward_scale = -1.0
    # TODO:(Alejandro, Sanchez) 12/16/2025 lowering the tracking contacts shape force reward scale from 4.0 to 1.0 or 2.0 or 1.5
    # want gait shaping to “style” the motion, not become the objective.
    # tracking_contacts_shaped_force_reward_scale = 4.0
    tracking_contacts_shaped_force_reward_scale = 0.5
    state_space = 0
    debug_vis = True
    # 5.1 Update Configuration
    # Additional reward scales
    orient_reward_scale = -2.0#-5.0
    lin_vel_z_reward_scale = -1.0#-0.02
    dof_vel_reward_scale = -0.5#-0.0001
    ang_vel_xy_reward_scale = -0.05#-0.001
    feet_air_time_reward_scale = 0.5 # Encourages proper swing phase
    # In Rob6323Go2EnvCfg
    base_height_min = 0.20  # Terminate if base is lower than 20cm
    base_height_target = 0.34  # NEW: Target standing height for Go2
    # reward scales
    # TODO:(Alejandro, Sanchez) 12/16/2025 Lower this value action_rate_reward_scale up to but no lower than -0.001 and then adjust
    # If the action rate is too strong then the following will happen:
    # sluggish response to commands
    # poor steady-state tracking (losing 10 pts)
    # gait looks “stuck” or overly damped
    #action_rate_reward_scale = -0.1 #<------ changed from this (before edit)
    action_rate_reward_scale = -0.01       
    dof_acc_reward_scale = -2.5e-7  # NEW: Penalize joint accelerations
    dof_vel_reward_scale = -0.0001  # Keep this
    torque_reward_scale = -1e-5  # NEW: Penalize high torques
     # Joint limits (keep joints in reasonable range)
    dof_pos_limits_reward_scale = -10.0  # NEW: Strong penalty near limits
    base_height_reward_scale = -1.0  # NEW: Maintain proper standing height
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # PD control gains
    Kp = 10.0  # Proportional gain try this first 10.0 then back to 25.0
    Kd = 0.2   # Derivative gain and try this first 0.2 then back to 0.5
    torque_limits = 23.5  # Max torque

    # Update robot_cfg
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # "base_legs" is an arbitrary key we use to group these actuators
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit_sim=23.5,
        velocity_limit_sim=30.0,
        stiffness=0.0,  # CRITICAL: Set to 0 to disable implicit P-gain
        damping=0.0,    # CRITICAL: Set to 0 to disable implicit D-gain
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5