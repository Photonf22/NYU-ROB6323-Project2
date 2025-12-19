# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import gymnasium as gym
import math
import torch
from collections.abc import Sequence
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg3 import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self.rew_orient= torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.rew_lin_vel_z= torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.rew_dof_vel= torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.rew_ang_vel_xy = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # X/Y linear velocity and yaw angular velocity __commands
        self.__commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Get specific body indices
        #self._feet_ids = []
        
        foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        #foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        #for name in foot_names:
        #    id_list, _ = self.robot.find_bodies(name)
        #    self._feet_ids.append(id_list[0])
        self.fz = None
        # Find indices in the CONTACT SENSOR (for forces)
        self._feet_ids_sensor = []
        for name in foot_names:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(id_list[0])
        self.rew_contact = None
        # Update Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            #for key in [
            #    "track_lin_vel_xy_exp",
            #    "track_ang_vel_z_exp",
            #    "rew_action_rate",
            #    "raibert_heuristic",
            #    "orient",
            #    "lin_vel_z",
            #    "dof_vel",
            #    "ang_vel_xy",
            #    "feet_clearance",
            #    "tracking_contacts_shaped_force",
            #]
            for key in [
                 "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "raibert_heuristic",
                "feet_clearance",
                "tracking_contacts_shaped_force",
                "feet_air_time",  # NEW
                "orient",
                "base_height",  # NEW
                "lin_vel_z",
                "ang_vel_xy",
                "action_rate",
                "dof_vel",
                "dof_acc",  # NEW
                "torques",  # NEW
                "dof_pos_limits",  # NEW
            ]

        }
        # Initialize torque tracking
        self._applied_torques = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        # Variables needed for the raibert heuristic
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
        self.torque_limits = cfg.torque_limits
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        #base_ids, _ = self.__contact_sensor.find_bodies("base")
        #self._base_id = base_ids[0]
        #self._base_id, _ = self.__contact_sensor.find_bodies("base")
        # variables needed for action rate penalization
        # Shape: (num_envs, action_dim, history_length)
        self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), 3, dtype=torch.float, device=self.device, requires_grad=False)

        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        # self._undesired_contact_body_ids, _ = self.__contact_sensor.find_bodies(".*thigh")

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
    # We need to know which body indices correspond to the feet to get their positions.
    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns the feet positions in the world frame.
        Shape: (num_envs, num_feet, 3)
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    #def _pre_physics_step(self, actions: torch.Tensor) -> None:
    #    self._actions = actions.clone()
        # Compute desired joint positions from policy actions
    #    self.desired_joint_pos = (
    #        self.cfg.action_scale * self._actions 
    #        + self.robot.data.default_joint_pos
    #    )
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions and compute PD control torques."""
        # Initialize on first call
        if not hasattr(self, '_actions'):
            self._actions = torch.zeros_like(actions)
            self._previous_actions = torch.zeros_like(actions)
            self._applied_torques = torch.zeros_like(actions)
        
        # Store previous actions
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        
        # Compute desired joint positions
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions 
            + self.robot.data.default_joint_pos
        )
        
        # Get current states
        current_pos = self.robot.data.joint_pos
        current_vel = self.robot.data.joint_vel
        
        # PD Controller: τ = Kp*(q_des - q) - Kd*q_dot
        # Note: We subtract Kd*velocity to damp motion (not add!)
        position_error = self.desired_joint_pos - current_pos
        
        torques = self.cfg.Kp * position_error - self.cfg.Kd * current_vel
        
        # Clip to actuator limits
        torques = torch.clamp(
            torques,
            -self.cfg.torque_limits,
            self.cfg.torque_limits
        )
        
        # Store for rewards
        self._applied_torques = torques.clone()
        
        # Apply to robot
        self.robot.set_joint_effort_target(torques)
    # We calculate the torques manually using the standard PD formula: τ = K p ( q d e s − q ) − K d q ˙ .
    def _apply_action(self) -> None:
        # Compute PD torques
        torques = torch.clip(
            (
                self.Kp * (
                    self.desired_joint_pos 
                    - self.robot.data.joint_pos 
                )
                - self.Kd * self.robot.data.joint_vel
            ),
            -self.torque_limits,
            self.torque_limits,
        )

        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self.__commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,  # Add gait phase info
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    def _reward_feet_clearance(self) -> torch.Tensor:
        """Penalize feet from being too high above ground.
        
        This prevents the robot from lifting feet excessively high during swing,
        which wastes energy and creates unstable gaits.
        """
        # Get foot positions in world frame
        feet_pos_w = self.robot.data.body_pos_w[:, self._feet_ids, :]
        
        # Calculate foot heights above ground
        feet_heights = feet_pos_w[:, :, 2]
        
        # Penalize any foot that's above a threshold height
        # Typical threshold: 0.05 - 0.15 meters
        clearance_threshold = 0.10  # 10cm
        
        # Only penalize heights above threshold
        excessive_clearance = (feet_heights - clearance_threshold).clip(min=0.0)
        
        # Sum over all feet
        return torch.sum(torch.square(excessive_clearance), dim=1)
    def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        """Reward for encouraging trotting gait through contact force shaping.
        
        Rewards diagonal pair contacts (FL+RR or FR+RL) being in phase.
        """
        # Get contact forces in z-direction (normal to ground)
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2]
        
        # Binary contact detection (1 = contact, 0 = no contact)
        # Using a threshold of 1 Newton
        in_contact = (contact_forces > 1.0).float()
        
        # Foot indices: 0=FL, 1=FR, 2=RL, 3=RR
        # For trotting, we want:
        # - Diagonal pair 1: FL (0) + RR (3)
        # - Diagonal pair 2: FR (1) + RL (2)
        
        # Reward when diagonal pairs are both in contact OR both in air
        # This encourages trotting pattern
        pair1_sync = in_contact[:, 0] * in_contact[:, 3]  # Both FL and RR in contact
        pair2_sync = in_contact[:, 1] * in_contact[:, 2]  # Both FR and RL in contact
        
        # Penalize when non-diagonal feet are in contact (pacing or bounding)
        pair1_opposite = in_contact[:, 0] * in_contact[:, 1]  # FL and FR (front pair)
        pair2_opposite = in_contact[:, 2] * in_contact[:, 3]  # RL and RR (rear pair)
        
        # Combine rewards
        diagonal_reward = pair1_sync + pair2_sync
        opposite_penalty = pair1_opposite + pair2_opposite
        
        reward = diagonal_reward - 0.5 * opposite_penalty
        
        return reward
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return all reward components as a dictionary."""
        self._step_contact_targets() # Update gait state
        # Primary objectives - command tracking
        lin_vel_error = torch.sum(
            torch.square(self.__commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_reward = torch.exp(-lin_vel_error / 0.25)
        
        ang_vel_error = torch.square(self.__commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        ang_vel_reward = torch.exp(-ang_vel_error / 0.25)
        
        # Gait shaping rewards
        raibert_heuristic = self._reward_raibert_heuristic()
        feet_clearance = self._reward_feet_clearance()
        tracking_contacts_shaped_force = self._reward_tracking_contacts_shaped_force()
        feet_air_time = self._reward_feet_air_time()  # NEW
        
        # Stability and posture
        orient_penalty = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        base_height = self._reward_base_height()  # NEW
        lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        
        # Action regularization
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        dof_acc = self._reward_dof_acc()  # NEW
        torques = self._reward_torques()  # NEW
        dof_pos_limits = self._reward_dof_pos_limits()  # NEW
        
        # Combine into dictionary with reward scales from config
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_reward * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": ang_vel_reward * self.cfg.yaw_rate_reward_scale,
            "raibert_heuristic": raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            "feet_clearance": feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": tracking_contacts_shaped_force * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale,  # NEW
            "orient": orient_penalty * self.cfg.orient_reward_scale,
            "base_height": base_height * self.cfg.base_height_reward_scale,  # NEW
            "lin_vel_z": lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "ang_vel_xy": ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale,
            "dof_vel": dof_vel * self.cfg.dof_vel_reward_scale,
            "dof_acc": dof_acc * self.cfg.dof_acc_reward_scale,  # NEW
            "torques": torques * self.cfg.torque_reward_scale,  # NEW
            "dof_pos_limits": dof_pos_limits * self.cfg.dof_pos_limits_reward_scale,  # NEW
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def _reward_feet_air_time(self):
        """Reward feet for being in the air for a reasonable duration."""
        # Get contact states for feet
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        
        # Reward air time between 0.1 and 0.25 seconds (typical for trotting)
        target_air_time = 0.15
        rew = torch.sum((last_air_time - target_air_time).abs() * first_contact, dim=1)
        return rew
    def _reward_base_height(self):
        """Penalize deviation from target base height."""
        base_height = self.robot.data.root_pos_w[:, 2]
        return torch.square(base_height - self.cfg.base_height_target)

    def _reward_dof_acc(self):
        """Penalize joint accelerations for smoothness."""
        return torch.sum(torch.square(self.robot.data.joint_acc), dim=1)
    def _compute_estimated_torques(self) -> torch.Tensor:
        """Estimate torques from joint positions and velocities.
        
        Use this if you're using implicit actuators and don't have direct torque access.
        """
        # Estimate based on your controller gains
        targets = self.robot.data.default_joint_pos + self._actions * self.cfg.action_scale
        pos_error = targets - self.robot.data.joint_pos
        vel_error = -self.robot.data.joint_vel
        
        estimated_torques = self.cfg.Kp * pos_error + self.cfg.Kd * vel_error
        estimated_torques = torch.clamp(
            estimated_torques, 
            -self.cfg.torque_limits, 
            self.cfg.torque_limits
        )
        
        return estimated_torques

    def _reward_torques(self) -> torch.Tensor:
        """Penalize high torques."""
        torques = self._compute_estimated_torques()
        return torch.sum(torch.square(torques), dim=1)
    
    def _reward_dof_pos_limits(self):
        """Penalize joints approaching their limits."""
        # Get joint positions and limits
        dof_pos = self.robot.data.joint_pos
        dof_pos_limits = self.robot.data.soft_joint_pos_limits
        
        # Calculate how close to limits (0 = at center, 1 = at limit)
        dof_pos_normalized = (dof_pos - dof_pos_limits[:, :, 0]) / \
                            (dof_pos_limits[:, :, 1] - dof_pos_limits[:, :, 0])
        
        # Penalize when close to limits (< 0.1 or > 0.9)
        out_of_limits = -(dof_pos_normalized - 0.5).abs() + 0.4
        return torch.sum(out_of_limits.clip(min=0.), dim=1)
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminate if base is too low
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # TODO: (Alejandro, Sanchez ) 12/16/2025 check if the following fixes the stationary robot (history max base force)
        # changing from line 259 to 261
        #base_forces = net_contact_forces[:, :, self._base_id, :]             # (N,H,3)
        #base_norm   = torch.linalg.norm(base_forces, dim=-1)          # (N,H)
        #base_max    = torch.amax(base_norm, dim=1)                    # (N,)
        #cstr_termination_contacts = base_max > 1.0
        cstr_termination_contacts = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        # apply all terminations
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Reset last actions hist
        self.last_actions[env_ids] = 0.
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new __commands
        self.__commands[env_ids] = torch.zeros_like(self.__commands[env_ids]).uniform_(-1.0, 1.0)
        # vx only, positive forward
        # TODO: for testing only (Alejandro, Sanchez) 12/16/2025
        #self.__commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(0.3, 1.0)
        #self.__commands[env_ids, 1] = 0.0  # vy
        #self.__commands[env_ids, 2] = 0.0  # yaw rate
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
     # 4.4 Implement Gait Logic
    # Defines contact plan
    def _step_contact_targets(self):
        frequencies = 3.
        phases = 0.5
        offsets = 0.
        bounds = 0.
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR
    # 4.5 Implement Raibert Reward
    # We calculate the error between where the foot IS and where the Raibert Heuristic says it SHOULD be.
    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(math_utils.quat_conjugate(self.robot.data.root_quat_w),
                                                            cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self.__commands[:, 0:1]
        yaw_vel_des = self.__commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.__commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
    


'''
    def _get_rewards(self) -> torch.Tensor:
        self._step_contact_targets() # Update gait state

        # Raibert shaping is applied only when translational __commands are non-zero to avoid biasing the policy toward in-place stepping.
        rew_raibert_heuristic = self._reward_raibert_heuristic() # (N,)
        cmd_speed = torch.linalg.norm(self.__commands[:, :2], dim=1)
        mask = (cmd_speed > 0.1).float()

        rew_raibert_heuristic = rew_raibert_heuristic * mask

        # action rate penalization
        # First derivative (Current - Last)
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale ** 2)
        # Second derivative (Current - 2*Last + 2ndLast)
        rew_action_rate += torch.sum(torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.cfg.action_scale ** 2)

        # Update the prev action hist (roll buffer and insert new action)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]
        # TODO: (Alejandro, Sanchez) 12/16/2025
        # if error is modest, reward stays high and doesn’t push hard.
        # linear velocity tracking
        # Increase tracking sharpness by increasing the exponential “tracking_sigma” (or, equivalently, decrease sigma → sharper).
        #tracking_sigma = 0.25
        tracking_sigma = 0.05
        lin_vel_error = torch.sum(torch.square(self.__commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / tracking_sigma)
        # yaw rate tracking
        yaw_rate_error = torch.square(self.__commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # 1. Penalize non-vertical orientation (projected gravity on XY plane)
        # Hint: We want the robot to stay upright, so gravity should only project onto Z.
        # Calculate the sum of squares of the X and Y components of projected_gravity_b.
        self.rew_orient = torch.sum( self.robot.data.projected_gravity_b[:,:2] ** 2, dim=1)

        # 2. Penalize vertical velocity (z-component of base linear velocity)
        # Hint: Square the Z component of the base linear velocity.
        #lin_vel_z  = self.robot.data.root_lin_vel_b[:, 2]**2
        #excess = torch.clamp(torch.abs(lin_vel_z) - 0.05, min=0.0)
        #self.rew_lin_vel_z = excess**2
        self.rew_lin_vel_z = self.robot.data.root_lin_vel_b[:, 2]**2
        # 3. Penalize high joint velocities
        # Hint: Sum the squares of all joint velocities.
        self.rew_dof_vel = torch.sum(self.robot.data.joint_vel ** 2, dim=1)

        # 4. Penalize angular velocity in XY plane (roll/pitch)
        # Hint: Sum the squares of the X and Y components of the base angular velocity.
        self.rew_ang_vel_xy = torch.sum( self.robot.data.root_ang_vel_b[:, :2] ** 2, dim=1)
        # part 6
                # Foot clearance: encourage feet to lift during swing phase.
        # Use desired_contact_states as a soft stance probability in [0,1].
        # Swing mask: 1 - stance_prob
        stance_prob = self.desired_contact_states  # (num_envs, 4)
        #stance_prob = 1.0 - self.__contact_sensor.data.is_contact[:, self.feet_indices].float()
        swing_prob = 1.0 - stance_prob
        # TODO:(Alejandro, Sanchez) 12/16/2025 lowered value from 0.06 and will adjust since 0.06 was not working well 
        # start with target_clearance = 0.05–0.07
        target_clearance = 0.06 # can be tuned added TODO: Alejandro Sanchez <------ changed from this (before edit)
        foot_height = self.foot_positions_w[:, :, 2]

        rew_feet_clearance = torch.sum(swing_prob * (foot_height - target_clearance).pow(2), dim=1)  # (N,)

        # TODO:(Alejandro, Sanchez) 12/16/2025 Applying a patch
        # hardening contact shaping block against NaNs and indexing
        # replacing torch.tang portion with the following in codeline 227
        #forces_hist = self.__contact_sensor.data.net_forces_w_history          # (N, H, B, 3)
        #feet_forces_hist = forces_hist[:, :, self._feet_ids_sensor, :]        # (N, H, 4, 3)
        #fz_hist = torch.clamp(feet_forces_hist[..., 2], min=0.0)              # (N, H, 4)
        #fz = torch.mean(fz_hist, dim=1)   # (N, 4) smoother
        #fz_scaled = torch.tanh(fz / 50.0)
        # (NEW) replacement for the above
        forces_hist = self.__contact_sensor.data.net_forces_w_history          # (N,H,B,3)
        feet_ids = list(self._feet_ids_sensor)                               # ensure safe indexing
        feet_forces_hist = forces_hist[:, :, feet_ids, :]                    # (N,H,4,3)
        fz_hist = torch.clamp(feet_forces_hist[:, :, :, 2], min=0.0)             # (N,H,4)

        fz = torch.mean(fz_hist, dim=1)                                      # (N,4)
        fz = torch.nan_to_num(fz, nan=0.0, posinf=0.0, neginf=0.0)          # avoid CUDA asserts
        #If instability is seen, increase the divisor from 50 -> 100 to soften it
        fz_scaled = torch.tanh(fz / 50.0)                                   

        # Reward stance having force, and swing having low force.
        # Using soft stance_prob/swing_prob keeps it differentiable-ish and matches your von-mises smoothing.
        # bounded “contact match” score:
        rew_tracking_contacts_shaped_force = torch.sum(
            stance_prob * fz_scaled + (1.0 - stance_prob) * (1.0 - fz_scaled),
            dim=1
        )
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale* self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale* self.step_dt,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            # Note: This reward is negative (penalty) in the config
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale, 
            "orient": self.rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": self.rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": self.rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": self.rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            # Part 6.3 (new)
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": rew_tracking_contacts_shaped_force * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "_reward_feet_air_time": rew_reward_feet_air_time *self.cfg.feet_air_time_reward_scale,
            "_reward_base_height":rew_reward_base_height*
            "_reward_dof_acc":rew_reward_dof_acc*
            "_reward_torques":rew_reward_torques*
            "_reward_dof_pos_limits":rew_reward_dof_pos_limits*
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
'''