from pybulletgym.envs.mujoco.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot
import numpy as np


class HalfCheetah(WalkerBase, MJCFBasedRobot):
    """
    Half Cheetah implementation based on MuJoCo.
    """
    foot_list = ["ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh"]  # track these contacts with ground

    def __init__(self):
        WalkerBase.__init__(self, power=1)
        MJCFBasedRobot.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=17, add_ignored_joints=True)

        self.pos_after = 0

    def calc_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)

        return np.concatenate([
            qpos.flat[1:],           # self.sim.data.qpos.flat[1:],
            qvel.flat		         # self.sim.data.qvel.flat,
        ])

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        pos_before = self.pos_after
        self.pos_after = self.robot_body.get_pose()[0]
        debugmode = 0
        if debugmode:
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        return (self.pos_after-pos_before) / self.scene.dt

    def robot_specific_reset(self, bullet_client,joints_coef = None,dynamics_coef = None):
        WalkerBase.robot_specific_reset(self, bullet_client)
        if dynamics_coef == None:
            for part_id, part in self.parts.items():
                self._p.changeDynamics(part.bodyIndex, part.bodyPartIndex, lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.1, restitution=0.5)
        else:
            for part_id, part in self.parts.items():
                self._p.changeDynamics(part.bodyIndex, part.bodyPartIndex, lateralFriction=dynamics_coef['lateralFriction'], spinningFriction=dynamics_coef['spinningFriction'],
                                       rollingFriction=dynamics_coef['rollingFriction'], restitution=dynamics_coef['restitution'])
        if joints_coef == None:
            self.jdict["bthigh"].power_coef = 120.0 #120 back thigh
            self.jdict["bshin"].power_coef  = 90.0  #90 back shin
            self.jdict["bfoot"].power_coef  = 60.0  #60 back foot
            self.jdict["fthigh"].power_coef = 140.0 #140 front thigh
            self.jdict["fshin"].power_coef  = 60.0 #60  front shin
            self.jdict["ffoot"].power_coef  = 30.0 #30 front foot
        else:
            self.jdict["bthigh"].power_coef = joints_coef['bthigh']  # 120 back thigh
            self.jdict["bshin"].power_coef = joints_coef['bshin']  # 90 back shin
            self.jdict["bfoot"].power_coef = joints_coef['bfoot']  # 60 back foot
            self.jdict["fthigh"].power_coef = joints_coef['fthigh']  # 140 front thigh
            self.jdict["fshin"].power_coef = joints_coef['fshin']  # 60  front shin
            self.jdict["ffoot"].power_coef = joints_coef['ffoot']  # 30 front foot
