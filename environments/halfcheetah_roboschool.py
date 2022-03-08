from pybulletgym.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from pybulletgym.envs.roboschool.robots.locomotors import HalfCheetah
from time import sleep
import numpy as np
class HalfCheetahBulletEnv(WalkerBaseBulletEnv):
    def __init__(self,joints_coef=None):
        self.robot = HalfCheetah()
        WalkerBaseBulletEnv.__init__(self, self.robot,joints_coef=joints_coef)
    def sample_tasks_joint(self, num_tasks,percent):
        low = 1 - percent
        high = 1 + percent
        np.random.seed(0)
        bthigh = np.random.uniform(int(120 * low), int(120 * high), size=(num_tasks,))
        np.random.seed(0)
        bshin = np.random.uniform(int(90 * low), int(90 * high), size=(num_tasks,))
        np.random.seed(0)
        bfoot = np.random.uniform(int(60 * low), int(60 * high), size=(num_tasks,))
        np.random.seed(0)
        fthigh = np.random.uniform(int(140 * low), int(140 * high), size=(num_tasks,))
        np.random.seed(0)
        fshin = np.random.uniform(int(60 * low), int(60 * high), size=(num_tasks,))
        np.random.seed(0)
        ffoot = np.random.uniform(int(30 * low), int(30 * high), size=(num_tasks,))
        dic = zip(bthigh,bshin,bfoot,fthigh,fshin,ffoot)
        joints_coef = [{'bthigh': bthigh, 'bshin': bshin, 'bfoot': bfoot, 'fthigh': fthigh, 'fshin': fshin, "ffoot": ffoot}
                       for bthigh, bshin,bfoot,fthigh,fshin,ffoot in dic]
        return joints_coef
if __name__ == '__main__':
    env = HalfCheetahBulletEnv()
    joints_coef = env.sample_tasks_joint(10,0.3)
    print(joints_coef[0])
    env = HalfCheetahBulletEnv(joints_coef = joints_coef[0])
    env.render('human')  # call this before env.reset, if you want a window showing the environment
    state = env.reset()  # should return a state vector if everything worked
    print("start simulation")
    for _ in range(1000):
        act = env.action_space.sample()  # 在动作空间中随机采样
        obs, reward, done, _ = env.step(act)  # 与环境交互
        env.camera_adjust()
        sleep(1 / 100)
    env.close()