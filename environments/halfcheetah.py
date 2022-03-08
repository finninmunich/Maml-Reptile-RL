from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.half_cheetah import HalfCheetah
import numpy as np
from time import sleep
import gym
gym.logger.set_level(40)
class HalfCheetahVelEnv(WalkerBaseMuJoCoEnv):
    def __init__(self,joints_coef = None,dynamics_coef = None):
        self.joints_coef = joints_coef
        self.dynamics_coef = dynamics_coef
        self.robot = HalfCheetah()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot,joints_coef=self.joints_coef,dynamics_coef=self.dynamics_coef)
    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()
        # state = self.robot.calc_state()
        # alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
        #     1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        # done = alive < 0
        # if not np.isfinite(state).all():
        #     print("~INF~", state)
        #     done = True
        potential = self.robot.calc_potential()
        #potential = -1.0 * abs(potential - self.goal_vel)
        power_cost = -0.1 * np.square(a).sum()
        state = self.robot.calc_state()

        done = False

        debugmode = 0
        if debugmode:
            print("potential=")
            print(potential)
            print("power_cost=")
            print(power_cost)

        self.rewards = [done,
            potential,
            power_cost
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def sample_tasks_joint(self, num_tasks,percent):
        low = 1 - percent
        high  = 1+ percent
        np.random.seed(0)
        bthigh = np.random.uniform(int(120*low), int(120*high), size=(num_tasks,))
        np.random.seed(0)
        bshin = np.random.uniform(int(90*low), int(90*high), size=(num_tasks,))
        np.random.seed(0)
        bfoot = np.random.uniform(int(60*low), int(60*high), size=(num_tasks,))
        np.random.seed(0)
        fthigh = np.random.uniform(int(140*low), int(140*high), size=(num_tasks,))
        np.random.seed(0)
        fshin = np.random.uniform(int(60*low), int(60*high), size=(num_tasks,))
        np.random.seed(0)
        ffoot = np.random.uniform(int(30*low), int(30*high), size=(num_tasks,))
        dic = zip(bthigh,bshin,bfoot,fthigh,fshin,ffoot)
        joints_coef = [{'bthigh': bthigh, 'bshin': bshin, 'bfoot': bfoot, 'fthigh': fthigh, 'fshin': fshin, "ffoot": ffoot}
                       for bthigh, bshin,bfoot,fthigh,fshin,ffoot in dic]
        return joints_coef
    def sample_tasks_dynamics(self, num_tasks,percent):
        low = 1 - percent
        high  = 1+ percent
        np.random.seed(0)
        lateralFriction = np.random.uniform(0.8*low, 0.8*high, size=(num_tasks,))
        np.random.seed(0)
        spinningFriction = np.random.uniform(0.1*low, 0.1*high, size=(num_tasks,))
        np.random.seed(0)
        rollingFriction = np.random.uniform(0.1*low, 0.1*high, size=(num_tasks,))
        np.random.seed(0)
        restitution = np.random.uniform(0.5*low,0.5*high, size=(num_tasks,))
        dic = zip(lateralFriction,spinningFriction,rollingFriction,restitution)
        dynamics_coef = [{'lateralFriction': lateralFriction, 'spinningFriction': spinningFriction, 'rollingFriction': rollingFriction, 'restitution': restitution}
                       for lateralFriction, spinningFriction,rollingFriction,restitution in dic]
        return dynamics_coef

    # def reset_task(self, task):
    #     self._task = task
    #     self._goal_vel = task['velocity']
if __name__ == '__main__':
    env = HalfCheetahVelEnv()
    joints_coef = env.sample_tasks_joint(10,0.3)
    #dynamics_coef = env.sample_tasks_dynamics(10,0.3)
    dynamics_coef = {'lateralFriction':0.8,'spinningFriction':0.1,'rollingFriction':0.1,'restitution':0.5}
    joints_coef = {'bthigh':120,'bshin':90,'bfoot':60,'fthigh':140,'fshin':60,"ffoot":30}
    #print(joints_coef[0])
    #print(dynamics_coef[0])
    env = HalfCheetahVelEnv(joints_coef = joints_coef,dynamics_coef = dynamics_coef)
    env.render('human')  # call this before env.reset, if you want a window showing the environment
    state = env.reset()  # should return a state vector if everything worked
    print("start simulation")
    for _ in range(10000):
        act = env.action_space.sample()  # 在动作空间中随机采样
        obs, reward, done, _ = env.step(act)  # 与环境交互
        env.camera_adjust()
        sleep(1 / 100)

    env.close()