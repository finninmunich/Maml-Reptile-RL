import numpy as np
import torch
from torch.distributions import Categorical
import time


# TODO: Fix the code of this page
def _log_summary(ep_len, ep_ret, ep_num):
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def normalization(x, mean, std, clip):
    x = x - mean
    x = x / (std + 1e-9)
    x = np.clip(x, -clip, clip)
    return x


def rollout(policy, env, render, obs_norm=None, mean=None, std=None, clip=5.0, continuous=False):
    while True:
        env.render('human')
        obs = env.reset()
        done = False
        # Logging data
        ep_len = 0  # episodic length
        ep_ret = 0  # episodic return
        count = 0
        if continuous:
            while not done:
                ep_len += 1
                if obs_norm:
                    obs = normalization(obs, mean, std, clip)
                action, _, _ = policy(torch.tensor(obs).float())
                action = action.detach().cpu().numpy()
                obs, rew, done, _ = env.step(action)
                # env.render('human')
                # env._p.stepSimulation()
                #if count %100 ==0:
                env.camera_adjust()
                time.sleep(1 / 60)
                ep_ret += rew
                count+=1
        else:
            while not done:
                ep_len += 1
                if obs_norm:
                    obs = normalization(obs, mean, std, clip)
                action_prob, _ = policy(torch.tensor(obs).float())
                dist = Categorical(action_prob)
                action = dist.sample()
                action = action.detach().cpu().numpy()
                obs, rew, done, _ = env.step(action)
                env.camera_adjust()
                time.sleep(1/240)
                ep_ret += rew

        yield ep_len, ep_ret


def eval_policy(policy, env, render=False, normalization=False, continuous=False):
    if normalization:
        mean = np.load('obs_mean.npy')
        std = np.load('obs_std.npy')
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render, normalization, continuous=continuous)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)



