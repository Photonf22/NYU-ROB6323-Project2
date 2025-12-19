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
    action_scale = 0.1 # changed from 0.25 to 0.1 because it was unstable
    action_space = 12
    observation_space = 48
    state_space = 0
    debug_vis = True
    action_rate_reward_scale = -0.1
    #--------------------------------------------------------------------------------------------------------------
    # 2.1 Update Configuration
    # PD control gains
    Kp = 10.0  # Proportional gain / changed from 20.0 -> 10.0
    Kd = 0.2   # Derivative gain  / changed from 0.5 -> 0.2
    torque_limits = 23.5  # Max torque / changed from 100.0 to 23.5 since torque was too high which would make the robot make unstable moves

    #--------------------------------------------------------------------------------------------------------------
    # Part 3: Early Stopping (Min Base Height)
    # Define the threshold for termination.
    base_height_min = 0.20  # Terminate if base is lower than 20cm
    #--------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------
    # Part 5: Refining the Reward Function
    #To achieve stable and natural-looking locomotion, we need to shape the robot's behavior further. We will add penalties for:

    #- Non-flat body orientation (projected gravity).
    #- Vertical body movement (bouncing).
    #- Excessive joint velocities.
    #- Body rolling and pitching (angular velocity).
    # 5.1 Update Configuration
    # Additional reward scales
    # reward scales
    # TODO:(Alejandro, Sanchez) 12/16/2025 Lower this value action_rate_reward_scale up to but no lower than -0.001 and then adjust
    # If the action rate is too strong then the following will happen:
    # sluggish response to commands
    # poor steady-state tracking (losing 10 pts)
    # gait looks “stuck” or overly damped
    orient_reward_scale = -2.0  # changed from -5.0 to -2.0
    lin_vel_z_reward_scale = -1.0 # increased from -0.02 to -1.0
    ang_vel_xy_reward_scale = -0.05 # increased from -0.001 to -0.05
    dof_vel_reward_scale = -0.0001 # Kept as is

    #--------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------
    # Part 4: Raibert Heuristic (Gait Shaping)
    # The Raibert Heuristic is a classic control strategy that places feet to stabilize velocity. 
    # We will use it as a "teacher" reward to encourage the policy to learn proper stepping. For reference logic, see IsaacGymEnvs implementation.
    # 4.1 Update Configuration
    observation_space = 48 + 4  # Added 4 for clock inputs
    # TODO:(Alejandro, Sanchez) 12/16/2025 lowering the tracking contacts shape force reward scale from -10.0 to -0.5 to -2.0
    # raibert_heuristic_reward_scale = -10.0
    # When the policy tracks commands and moves, gradually tighten it back down (more negative) to clean up foot placement.
    # after moves and tracks, you can push it more negative (e.g., -3, -5, maybe back to -10).
    raibert_heuristic_reward_scale = -1.0  #decreased from -10.0 to -1.0
    #--------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------
    # Part 6: Advanced Foot Interaction
    # Next, we will add two critical rewards for legged locomotion: Foot Clearance (lifting feet during swing) and Contact Forces (grounding feet during stance).
    # We will adapt the implementation from IsaacGymEnvs.
    # also increase contact shaping a bit if footfalls aren’t clean
    # 6.1 Update Configuration line 36 and line 37
    # TODO:(Alejandro, Sanchez) 12/16/2025 lowering the reward scale from -30.0 to -1.0
    feet_clearance_reward_scale = -1.0
    # TODO:(Alejandro, Sanchez) 12/16/2025 lowering the tracking contacts shape force reward scale from 4.0 to 0.5
    # want gait shaping to “style” the motion, not become the objective.
    tracking_contacts_shaped_force_reward_scale = 0.5
    #--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
    # 2.1 Update Configuration
    # Update robot_cfg
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # "base_legs" is an arbitrary key we use to group these actuators
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # CRITICAL: Set to 0 to disable implicit P-gain
        damping=0.0,    # CRITICAL: Set to 0 to disable implicit D-gain
    )
    #--------------------------------------------------------------------------------------------------------------
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
    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

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