import argparse
import copy
from collections import OrderedDict

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from AC_Network.network import ActorCritic
from PPO.reptile_ppo import PPO
from environments.halfcheetah_roboschool import HalfCheetahBulletEnv


def update_init_params(target, old, step_size=0.1):
    """Apply one step of gradient descent on the loss function `loss`, with
    step-size `step_size`, and returns the updated parameters of the neural
    meta_policy.
    """
    updated = OrderedDict()
    for ((name_old, oldp), (name_target, targetp)) in zip(old.items(), target.items()):
        assert name_old == name_target, "target and old params are different"
        updated[name_old] = oldp + step_size * (targetp - oldp)  # grad ascent so its a plus
    return updated


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--save_name', dest='save_name', type=str, default='maml')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, default='')
    parser.add_argument('--N', dest='N', type=int, default=0)
    # parser.add_argument('--ac_model', dest='ac_model', type=str, default='')
    # parser.add_argument('--env', type=str, default='')
    # parser.add_argument("--continuous", action="store_true", help="Continuous or not")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    meta_iterations = 2000
    reptile_policy_lr = 0.01
    num_tasks = 20
    joints_offset = 0.3
    dynamics_offset = 0.3
    env = HalfCheetahBulletEnv()
    #env = HalfCheetahVelEnv()
    # env = MetaInvertedDoublePendulum()
    # task_flag = True
    # while task_flag == True:
    #     tasks = env.sample_tasks(1, 30, 50, 500, num_tasks)  # gravity: 5~15 generate 10 task
    #     task_flag = False
    #     for each in tasks:
    #         if abs(each['gravity'] - 9.8) < 1.0:
    #             task_flag = True
    #         if abs(each['torque_factor'] - 200) < 20:
    #             task_flag = True
    joints_coefs = env.sample_tasks_joint(num_tasks, joints_offset)
    dynamics_coefs = env.sample_tasks_dynamics(num_tasks, dynamics_offset)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    continuous = True if type(env.action_space) == gym.spaces.box.Box else False
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}, continuous? {continuous}")
    hyperparameters = {
        'N': args.N,
        'K': 16,
        'max_timesteps_per_episode': 256,
        'mini_batch_size': 32,
        'state_norm': False,
        'continuous': continuous,
        'gamma': 0.995,
        'lamda': 0.97,
        'lr': 1e-3,  # inner loop lr
        'clip': 0.2,
        'render': True,
        'update_per_iteration': 10,
    }
    # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # writer = SummaryWriter('runs/T1000/S2R_N_more_epoch' + str(hyperparameters['N']) + '_task_' + str(num_tasks))
    writer = SummaryWriter(
        'runs/reptile_train/' + str(args.save_name) + '_as_' + str(hyperparameters['N']) + '_task_' + str(
            num_tasks))
    meta_policy = ActorCritic(obs_dim, act_dim, continuous=continuous, layer_norm=True)
    model = PPO(meta_policy, env,**hyperparameters)
    for i in range(meta_iterations):
        print(f"meta iteration {i+1}/{meta_iterations}")
        init_params = copy.deepcopy(OrderedDict(model.network.named_parameters()))
        temp_params = copy.deepcopy(OrderedDict(model.network.named_parameters()))
        meta_reward = []
        for task_index in range(num_tasks):
            print(f"task {task_index}/{num_tasks}")
            #task = HalfCheetahVelEnv(joints_coef=joints_coefs[task_index],dynamics_coef=dynamics_coefs[task_index])  # set a specific task
            #task = MetaInvertedDoublePendulum(task=tasks[task_index])
            task = HalfCheetahBulletEnv(joints_coef=joints_coefs[task_index])
            model.set_env(task)  # set this test to our model
            model.network.load_state_dict(init_params)  # load initial params for our model
            model.init_optimizers()  # reset init_optimizer
            reward = model.learn()  # train our model under this task
            meta_reward.append(reward)
            print(f"Meta Iter{i + 1}/{meta_iterations}, task {task_index + 1}/{num_tasks}, average reward:{reward}")
            target_policy = OrderedDict(model.network.named_parameters())  # get target params from this training

            temp_params = update_init_params(target_policy, temp_params,
                                             reptile_policy_lr / hyperparameters['N'])  # update our params from this task

        avg_meta_reward = np.mean(np.array(meta_reward))
        print(f"avg reward of this meta_iteration is :{avg_meta_reward}")
        model.network.load_state_dict(temp_params)  # update our network
        torch.save(model.network.state_dict(),
                   'model/reptile_model/' + str(args.save_name) + '_as_' + str(hyperparameters['N']) + '_task_' + str(
                       num_tasks) + '.pth')
        writer.add_scalar('Meta_Reward', avg_meta_reward, i+1)
