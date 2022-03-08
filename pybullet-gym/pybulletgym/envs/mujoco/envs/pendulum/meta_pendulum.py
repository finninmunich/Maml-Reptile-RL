from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.mujoco.scenes.scene_bases import SingleRobotEmptyScene
from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot
import numpy as np
import gym
from time import sleep
class InvertedPendulum(MJCFBasedRobot):

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'inverted_pendulum.xml', 'cart', action_dim=1, obs_dim=4)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        u = self.np_random.uniform(low=-.1, high=.1)
        self.j1.reset_current_position(u, 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert(np.isfinite(a).all())
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0
        self.slider.set_motor_torque(100*float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        x, vx = self.slider.current_position()
        self.theta, theta_dot = self.j1.current_position()
        assert(np.isfinite(x))

        if not np.isfinite(x):
            print("x is inf")
            x = 0

        if not np.isfinite(vx):
            print("vx is inf")
            vx = 0

        if not np.isfinite(self.theta):
            print("theta is inf")
            self.theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        qpos = np.array([x, self.theta])  # shape (2,)
        qvel = np.array([vx, theta_dot])  # shape (2,)

        return np.concatenate([
            qpos,   # self.sim.data.qpos
            qvel]).ravel() # self.sim.data.qvel


class MetaInvertedPendulum(BaseBulletEnv):
    def __init__(self,gravity=9.8):
        self.robot = InvertedPendulum()
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1
        self.gravity = gravity
    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=self.gravity, timestep=0.0165, frame_skip=1)

    def reset(self):
        if self.stateId >= 0:
            # print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
            self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        # print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
        return r

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.pos_x self.pos_y
        vel_penalty = 0
        reward = 1.0
        done = not np.isfinite(state).all() or np.abs(state[1]) > .2
        self.rewards = [float(reward)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self,i=0,j=1.2,k=1.0,x=0,y=0,z=0.5):
        self.move_and_look_at(i,j,k,x,y,z)
