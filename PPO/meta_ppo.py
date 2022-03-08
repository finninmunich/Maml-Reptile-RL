from typing import List

import higher

from tools.Zfilter import ZFilter
from tools.utils import *


def gradadd(grads1: List[torch.Tensor], grads2: List[torch.Tensor]):
    for i in range(len(grads1)):
        grads1[i] += grads2[i]
    return grads1


class MetaPPo:
    def __init__(self, meta_policy, env, **hyperparameters):

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_hyperparameters(hyperparameters)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        if self.continuous:
            self.act_dim = env.action_space.shape[0]
        else:
            self.act_dim = env.action_space.n
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        print(f"using {self.device}!")
        # self.device = 'cpu'
        self.meta_policy = meta_policy.to(self.device)
        # self.fmeta_policy = copy.deepcopy(self.meta_policy)
        # self.logger = {
        #     'delta_t': time.time_ns(),
        #     't_so_far': 0,  # timesteps so far
        #     'epoch_so_far': 0,  # iterations so far
        #     'batch_lens': [],  # episodic lengths in batch
        #     'batch_rews': [],  # episodic returns in batch
        #     'total_losses': [],  # losses of actor meta_policy in current iteration
        #     'mean': 0,  # mean of observation
        #     'std': 0,  # std of observation
        # }
        # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        # self.writer = SummaryWriter('runs/' + TIMESTAMP)

    def _init_hyperparameters(self, hyperparameters):
        # Default values for hyperparameters
        self.N = 3
        self.K = 16  # timesteps per batch
        self.max_timesteps_per_episode = 200  # timesteps per episode
        self.mini_batch_size = 256
        self.gamma = 0.995  # discounted factor
        self.lamda = 0.97  # advantages factor
        self.clip = 0.2
        self.lr = 1e-3
        # Miscellaneous parameters
        self.render = True
        self.seed = None
        self.layer_norm = True
        self.state_norm = False
        self.advantage_norm = True
        self.lossvalue_norm = True
        self.schedule_adam = 'linear'
        self.schedule_clip = 'linear'
        self.loss_coeff_value = 0.5
        self.loss_coeff_entropy = 0.01
        self.update_per_iteration = 10
        self.continuous = True
        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))  # exec can run the python command on str format

    def rollout(self, running_state, memory, reward_list, continuous, temp_policy=None):
        """
        we spend most of our time on this function
        """
        # collect K episodes
        for episode in range(self.K):
            state = self.env.reset()
            if self.state_norm:
                state = running_state(state)
            reward_sum = 0
            for t in range(self.max_timesteps_per_episode):
                t += 1
                # if temp_policy != None:
                if continuous:
                    action_mean, action_std, value = temp_policy(torch.tensor(state, device=self.device).float())
                    action, logproba = temp_policy.sample_action(action_mean, action_std)
                    action = action.detach().cpu().numpy()
                    logproba = np.float32(logproba.detach().cpu())
                else:
                    action_prob, value = temp_policy(torch.tensor(state, device=self.device).float())
                    action, logproba = temp_policy.sample_action(action_prob)
                    action = action.detach().cpu().numpy()
                    logproba = np.float32(logproba.detach().cpu())
                # else:
                #     if continuous:
                #         action_mean, action_std, value = self.fmeta_policy(state)
                #         action, logproba = self.fmeta_policy.sample_action(action_mean, action_std)
                #         action = action.detach().cpu().numpy()
                #         logproba = np.float32(logproba.detach().cpu())
                #     else:
                #         action_prob, value = self.fmeta_policy(state)
                #         action, logproba = self.fmeta_policy.sample_action(action_prob)
                #         action = action.detach().cpu().numpy()
                #         logproba = np.float32(logproba.detach().cpu())
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                if self.state_norm:
                    next_state = running_state(next_state)
                mask = 0 if done else 1
                memory.push(state, value, action, logproba, mask, next_state, reward)
                if done:
                    break
                state = next_state
            reward_list.append(reward_sum)

        # self.logger['batch_rews'] = reward_list
        # self.logger['batch_lens'] = len_list
        # if self.state_norm:
        #     obs_mean, obs_std = nor_std.get_mean_std()
        #     self.logger['mean'] = obs_mean
        #     self.logger['std'] = obs_std
        #     np.save('obs_mean', obs_mean)
        #     np.save('obs_std', obs_std)
        batch = memory.sample()
        batch_size = len(memory)

        rewards = torch.tensor(batch.reward, device=self.device)
        # normalize rewards:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        values = torch.tensor(batch.value, device=self.device)
        masks = torch.tensor(batch.mask, device=self.device)
        actions = torch.tensor(np.array(batch.action), device=self.device)
        states = torch.tensor(np.array(batch.state), device=self.device)
        oldlogproba = torch.tensor(batch.logproba, device=self.device)

        returns = torch.Tensor(batch_size).to(self.device)
        deltas = torch.Tensor(batch_size).to(self.device)
        advantages = torch.Tensor(batch_size).to(self.device)
        return rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size

    def meta_learn(self):
        """
        In this function, we use the episode we collect to update our fmeta_policy / return grads for update(on validation step)
        one step of adaptation
        :return:
        """
        optimizer = torch.optim.SGD(self.meta_policy.parameters(),
                                    lr=self.lr)  # each learn step, we re-initialize the optimizer
        running_state = ZFilter((self.obs_dim,), clip=5.0)
        continuous = self.meta_policy.continuous
        clip_now = self.clip
        # step1: perform current policy to collect trajectories
        # self.logger['epoch_so_far'] = epoch + 1
        # self.logger['t_so_far'] = global_steps

        # first iteration: collect K iteration and use it to update fmeta_policy
        # second iteration: collect K iteration to validation fmeta_policy. i.e. one more update step but return grad
        # collect K episodes from fmeta_policy
        # create an empty memory
        print(f"collecting k episodes for updating fmeta_policy")
        with higher.innerloop_ctx(self.meta_policy, optimizer, copy_initial_weights=False) as (fmeta_policy, diffopt):
            for n_adapt in range(self.N):
                print(f"updating fmeta policy {n_adapt + 1}/{self.N}")
                memory = Memory()
                # create reward list for this roolout
                reward_list = []  # this reward_list seems to be useless
                print(f"collecting K rollouts!")
                rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size = self.rollout(
                    running_state,
                    memory, reward_list, continuous, temp_policy=fmeta_policy)  # also collect it from fmeta_policy
                print(f"calculating losses!")
                prev_return = 0
                prev_value = 0
                prev_advantage = 0
                # calculate advantages
                for i in reversed(range(batch_size)):
                    returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
                    deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
                    advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage * masks[i]
                    prev_return = returns[i]
                    prev_value = values[i]
                    prev_advantage = advantages[i]
                if self.advantage_norm:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                for i_epoch in range(int(self.update_per_iteration * batch_size / self.mini_batch_size)):
                    minibatch_ind = np.random.choice(batch_size, self.mini_batch_size, replace=False)
                    minibatch_states = states[minibatch_ind].float()
                    minibatch_actions = actions[minibatch_ind].float()
                    minibatch_oldlogproba = oldlogproba[minibatch_ind]
                    minibatch_advantages = advantages[minibatch_ind]
                    minibatch_returns = returns[minibatch_ind]
                    minibatch_newlogproba, minibatch_newvalues, loss_entropy = fmeta_policy.evaluate(
                        minibatch_states, minibatch_actions)  # probably need to set to self.fmeta_policy
                    ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                    surr1 = ratio * minibatch_advantages
                    surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
                    loss_surr = -torch.mean(torch.min(surr1, surr2))
                    if self.lossvalue_norm:
                        minibatch_return_6std = 6 * minibatch_returns.std()
                        loss_value = torch.mean(
                            (minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
                    else:
                        loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                    loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
                    diffopt.step(loss)

            print('validating fmeta_policy!')
            memory = Memory()
            # create reward list for this roolout
            reward_list = []  # this reward_list seems to be useless
            print("collecting K episodes from updated fmeta_policy")
            rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size = self.rollout(
                running_state,memory, reward_list, continuous, temp_policy=fmeta_policy)  # also collect it from fmeta_policy
            print(f"the average_rewards of validation step is {np.mean(np.array(reward_list))}")
            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            # calculate advantages
            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
                advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage * masks[i]
                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            if self.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

            print("updating fmeta_policy")
            newlogproba, newvalues, loss_entropy = fmeta_policy.evaluate(
                states.float(), actions)  # probably need to set to self.fmeta_policy
            ratio = torch.exp(newlogproba - oldlogproba)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * advantages
            loss_surr = -torch.mean(torch.min(surr1, surr2))
            if self.lossvalue_norm:
                return_6std = 6 * returns.std()
                loss_value = torch.mean(
                    (newvalues - returns).pow(2)) / return_6std
            else:
                loss_value = torch.mean((newvalues - returns).pow(2))

            loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
        return np.mean(np.array(reward_list)), loss














































    def maml_learn(self):
        """
        In this function, we use the episode we collect to update our fmeta_policy / return grads for update(on validation step)
        one step of adaptation
        :return:
        """
        optimizer = torch.optim.Adam(self.fmeta_policy.parameters(),
                                     lr=self.lr)  # each learn step, we re-initialize the optimizer
        nor_std = Normalization((self.obs_dim,), clip=10.0)  # TODO: right now we don't use nor_std
        continuous = self.meta_policy.continuous
        clip_now = self.clip
        # step1: perform current policy to collect trajectories
        # self.logger['epoch_so_far'] = epoch + 1
        # self.logger['t_so_far'] = global_steps

        # first iteration: collect K iteration and use it to update fmeta_policy
        # second iteration: collect K iteration to validation fmeta_policy. i.e. one more update step but return grad
        # collect K episodes from fmeta_policy
        # create an empty memory
        print(f"collecting k episodes for updating fmeta_policy")
        memory = Memory()
        # create reward list for this roolout
        reward_list = []  # this reward_list seems to be useless
        rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size = self.rollout(
            memory, reward_list, continuous)  # also collect it from fmeta_policy
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        # calculate advantages
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage * masks[i]
            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if self.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        # update fmeta_policy if it is not validation, return grad if it is validation
        with higher.innerloop_ctx(self.fmeta_policy, optimizer) as (fmeta_policy, diffopt):
            print("updating fmeta_policy")
            for i_epoch in range(int(batch_size / self.mini_batch_size)):
                minibatch_ind = np.random.choice(batch_size, self.mini_batch_size, replace=False)
                minibatch_states = states[minibatch_ind].float()
                minibatch_actions = actions[minibatch_ind].float()
                minibatch_oldlogproba = oldlogproba[minibatch_ind]
                minibatch_advantages = advantages[minibatch_ind]
                minibatch_returns = returns[minibatch_ind]
                minibatch_newlogproba, minibatch_newvalues, loss_entropy = fmeta_policy.evaluate(
                    minibatch_states, minibatch_actions)  # probably need to set to self.fmeta_policy
                ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                surr1 = ratio * minibatch_advantages
                surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
                loss_surr = -torch.mean(torch.min(surr1, surr2))
                if self.lossvalue_norm:
                    minibatch_return_6std = 6 * minibatch_returns.std()
                    loss_value = torch.mean(
                        (minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
                else:
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
                diffopt.step(loss)

            print('validating fmeta_policy!')
            memory = Memory()
            # create reward list for this roolout
            reward_list = []  # this reward_list seems to be useless
            print("collecting K episodes from updated fmeta_policy")
            rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size = self.rollout(
                memory, reward_list, continuous, temp_policy=fmeta_policy)  # also collect it from fmeta_policy
            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            # calculate advantages
            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
                advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage * masks[i]
                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            if self.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

            print("updating fmeta_policy")
            newlogproba, newvalues, loss_entropy = fmeta_policy.evaluate(
                states, actions)  # probably need to set to self.fmeta_policy
            ratio = torch.exp(newlogproba - oldlogproba)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * advantages
            loss_surr = -torch.mean(torch.min(surr1, surr2))
            if self.lossvalue_norm:
                return_6std = 6 * returns.std()
                loss_value = torch.mean(
                    (newvalues - returns).pow(2)) / return_6std
            else:
                loss_value = torch.mean((newvalues - returns).pow(2))

            loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
            print(f" the total loss of K validation episodes is: {loss.item()}")
            grad_wrt_original = list(
                torch.autograd.grad(loss, fmeta_policy.parameters(time=0)))
        return np.mean(np.array(reward_list)), grad_wrt_original

    def sim2real_learn(self):
        """
        In this function, we use the episode we collect to update our fmeta_policy / return grads for update(on validation step)
        one step of adaptation
        :return:
        """
        optimizer = torch.optim.Adam(self.fmeta_policy.parameters(),
                                     lr=self.lr)  # each learn step, we re-initialize the optimizer
        nor_std = Normalization((self.obs_dim,), clip=10.0)  # TODO: right now we don't use nor_std
        continuous = self.meta_policy.continuous
        clip_now = self.clip
        # step1: perform current policy to collect trajectories
        # self.logger['epoch_so_far'] = epoch + 1
        # self.logger['t_so_far'] = global_steps

        # first iteration: collect K iteration and use it to update fmeta_policy
        # second iteration: collect K iteration to validation fmeta_policy. i.e. one more update step but return grad
        # collect K episodes from fmeta_policy
        # create an empty memory
        print(f"collecting k episodes for updating fmeta_policy")
        with higher.innerloop_ctx(self.fmeta_policy, optimizer) as (fmeta_policy, diffopt):
            print("updating fmeta_policy")
            for n_adapt in range(self.N):
                print(f"Adaptation step {n_adapt + 1}/{self.N}")
                memory = Memory()
                # create reward list for this roolout
                reward_list = []  # this reward_list seems to be useless
                if n_adapt == 0:
                    rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size = self.rollout(
                        memory, reward_list, continuous)  # also collect it from fmeta_policy
                else:
                    rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size = self.rollout(
                        memory, reward_list, continuous,
                        temp_policy=fmeta_policy)  # also collect it from fmeta_policy
                print(f"the average_rewards of {n_adapt + 1} adaptation step is {np.mean(np.array(reward_list))}")
                prev_return = 0
                prev_value = 0
                prev_advantage = 0
                # calculate advantages
                for i in reversed(range(batch_size)):
                    returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
                    deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
                    advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage * masks[i]
                    prev_return = returns[i]
                    prev_value = values[i]
                    prev_advantage = advantages[i]
                if self.advantage_norm:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                total_loss = []
                for i_epoch in range(int(batch_size / self.mini_batch_size)):
                    minibatch_ind = np.random.choice(batch_size, self.mini_batch_size, replace=False)
                    minibatch_states = states[minibatch_ind].float()
                    minibatch_actions = actions[minibatch_ind].float()
                    minibatch_oldlogproba = oldlogproba[minibatch_ind]
                    minibatch_advantages = advantages[minibatch_ind]
                    minibatch_returns = returns[minibatch_ind]
                    minibatch_newlogproba, minibatch_newvalues, loss_entropy = fmeta_policy.evaluate(
                        minibatch_states, minibatch_actions)  # probably need to set to self.fmeta_policy
                    ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                    surr1 = ratio * minibatch_advantages
                    surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
                    loss_surr = -torch.mean(torch.min(surr1, surr2))
                    if self.lossvalue_norm:
                        minibatch_return_6std = 6 * minibatch_returns.std()
                        loss_value = torch.mean(
                            (minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
                    else:
                        loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                    loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
                    diffopt.step(loss)
                    total_loss.append(loss.item())
                print(f"the total_loss of {n_adapt + 1} adaptation step is {np.mean(np.array(total_loss))}")

            print('validating fmeta_policy!')
            memory = Memory()
            # create reward list for this roolout
            reward_list = []  # this reward_list seems to be useless
            print("collecting K episodes from updated fmeta_policy")
            rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size = self.rollout(
                memory, reward_list, continuous, temp_policy=fmeta_policy)  # also collect it from fmeta_policy
            print(f"the average_rewards of validation step is {np.mean(np.array(reward_list))}")
            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            # calculate advantages
            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
                advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage * masks[i]
                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            if self.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

            print("updating fmeta_policy")
            newlogproba, newvalues, loss_entropy = fmeta_policy.evaluate(
                states, actions)  # probably need to set to self.fmeta_policy
            ratio = torch.exp(newlogproba - oldlogproba)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * advantages
            loss_surr = -torch.mean(torch.min(surr1, surr2))
            if self.lossvalue_norm:
                return_6std = 6 * returns.std()
                loss_value = torch.mean(
                    (newvalues - returns).pow(2)) / return_6std
            else:
                loss_value = torch.mean((newvalues - returns).pow(2))

            loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
            grad_wrt_original = list(
                torch.autograd.grad(loss, fmeta_policy.parameters(time=0)))
            print(f" the total loss of K validation episodes is: {loss.item()}")
        return np.mean(np.array(reward_list)), grad_wrt_original

        # update fmeta_policy if it is not validation, return grad if it is validation

        # self.logger['total_losses'].append(total_loss.detach().cpu())

        # if self.schedule_clip == 'linear':
        #     ep_ratio = 1 - (epoch / self.N)
        #     clip_now = self.clip * ep_ratio
        # if self.schedule_adam == 'linear':
        #     ep_ratio = 1 - (epoch / self.N)
        #     lr_now = self.lr * ep_ratio
        #     for g in optimizer.param_groups:
        #         g['lr'] = lr_now
        # self._log_summary()
        # torch.save(self.meta_policy.state_dict(), './actor_critic.pth')

    # def _log_summary(self):
    #     delta_t = self.logger['delta_t']
    #     self.logger['delta_t'] = time.time_ns()
    #     delta_t = (self.logger['delta_t'] - delta_t) / 1e9
    #     delta_t = str(round(delta_t, 2))
    #
    #     global_steps = self.logger['t_so_far']
    #     epoch_so_far = self.logger['epoch_so_far']
    #     avg_ep_lens = np.mean(self.logger['batch_lens'])
    #     avg_ep_rews = np.mean(self.logger['batch_rews'])
    #     avg_total_loss = np.mean([losses.float() for losses in self.logger['total_losses']])
    #     self.writer.add_scalar('Training loss', avg_total_loss, global_steps)
    #     self.writer.add_scalar('Average_rewards', avg_ep_rews, global_steps)
    #     avg_ep_lens = str(round(avg_ep_lens, 2))
    #     avg_ep_rews = str(round(avg_ep_rews, 2))
    #     avg_total_loss = str(round(avg_total_loss, 5))
    #
    #     # Print logging statements
    #     print(flush=True)
    #     print(f"-------------------- Epoch #{epoch_so_far} --------------------", flush=True)
    #     print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
    #     print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
    #     print(f"Average Loss: {avg_total_loss}", flush=True)
    #     print(f"Timesteps So Far: {global_steps}", flush=True)
    #     print(f"Epoch took: {delta_t} secs", flush=True)
    #     print(f"------------------------------------------------------", flush=True)
    #     print(flush=True)
    #
    #     self.logger['batch_lens'] = []
    #     self.logger['batch_rews'] = []
    #     self.logger['total_losses'] = []
