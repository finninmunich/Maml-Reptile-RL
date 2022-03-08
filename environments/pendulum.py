from time import sleep

import numpy as np

from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot
from pybulletgym.envs.mujoco.scenes.scene_bases import SingleRobotEmptyScene


class InvertedDoublePendulum(MJCFBasedRobot):
    def __init__(self,torque_factor = 200):
        self.torque_factor = torque_factor
        MJCFBasedRobot.__init__(self, 'inverted_double_pendulum.xml', 'cart', action_dim=1, obs_dim=11)

    def robot_specific_reset(self, bullet_client,joints_coef = None,dynamics_coef = None):
        self._p = bullet_client
        self.pole2 = self.parts["pole2"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        u = self.np_random.uniform(low=-.1, high=.1, size=[2])
        self.j1.reset_current_position(float(u[0]), 0)
        self.j2.reset_current_position(float(u[1]), 0)
        self.j1.set_motor_torque(0)
        self.j2.set_motor_torque(0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.slider.set_motor_torque(self.torque_factor * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        x, vx = self.slider.current_position()
        theta, theta_dot = self.j1.current_position()
        gamma, gamma_dot = self.j2.current_position()

        assert (np.isfinite(x))

        qpos = np.array([x, theta, gamma])  # shape (3,)
        qvel = np.array([vx, theta_dot, gamma_dot])  # shape (3,)
        qfrc_constraint = np.zeros(3)  # shape (3,)  # TODO: FIND qfrc_constraint in pybullet
        return np.concatenate([
            qpos[:1],  # self.sim.data.qpos[:1],  # cart x pos
            np.sin(qpos[1:]),  # np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(qpos[1:]),  # np.cos(self.sim.data.qpos[1:]),
            np.clip(qvel, -10, 10),  # np.clip(self.sim.data.qvel, -10, 10),
            np.clip(qfrc_constraint, -10, 10)  # np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()


class MetaInvertedDoublePendulum(BaseBulletEnv):
    def __init__(self, task=None):
        if task != None:
            self.gravity = task['gravity']
            self.torque_factor = task['torque_factor']
        else:
            self.gravity = 9.8
            self.torque_factor = 200
        self.robot = InvertedDoublePendulum(torque_factor = self.torque_factor)
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=self.gravity, timestep=0.0165, frame_skip=1)

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return r

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
        # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
        pos_x, _, pos_y = self.robot.pole2.pose().xyz()
        dist_penalty = 0.01 * pos_x ** 2 + (pos_y + 0.3 - 2) ** 2
        v1, v2 = self.robot.j1.current_position()[1], self.robot.j2.current_position()[1]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        done = pos_y + 0.3 <= 1
        self.rewards = [float(alive_bonus), float(-dist_penalty), float(-vel_penalty)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def sample_tasks(self, glow=None, ghigh=None,tlow=None,thigh=None ,num_tasks=None):
        if glow == None:
            gravities = [9.8]*num_tasks
        else:
            np.random.seed(0)
            gravities = np.random.uniform(glow, ghigh, size=(num_tasks,))
        if tlow == None:
            torque_factors = [200]*num_tasks
        else:
            np.random.seed(0)
            torque_factors = np.random.uniform(tlow,thigh,size = (num_tasks))
        dic = zip(gravities,torque_factors)
        tasks = [{'gravity': gravity,'torque_factor':torque_factor} for gravity,torque_factor in dic]
        return tasks

    def camera_adjust(self, i=0, j=1.2, k=1.0, x=0, y=0, z=0.5):
        self.move_and_look_at(i, j, k, x, y, z)


if __name__ == '__main__':
    # env = gym.make('InvertedDoublePendulumMuJoCoEnv-v0')
    # env = gym.make('InvertedPendulumMuJoCoEnv-v1')
    task = {"gravity": 9.8,'torque_factor':200}
    env = MetaInvertedDoublePendulum(task=task)
    env.render('human')  # call this before env.reset, if you want a window showing the environment
    state = env.reset()  # should return a state vector if everything worked
    for _ in range(10000):
        # env.camera_adjust(i=0,j=1.2,k=1.0,x=5, y=0, z=0.5)
        act = env.action_space.sample()  # 在动作空间中随机采样
        obs, reward, done, _ = env.step(act)  # 与环境交互
        sleep(1 / 100)

    env.close()
