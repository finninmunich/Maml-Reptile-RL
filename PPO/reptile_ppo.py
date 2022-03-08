from datetime import datetime

from tools.utils import *
from tools.Zfilter import ZFilter

class PPO:
    def __init__(self, policy, env, **hyperparameters):

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
        self.network = policy.to(self.device)
        # self.logger = {
        #     'delta_t': time.time_ns(),
        #     't_so_far': 0,  # timesteps so far
        #     'epoch_so_far': 0,  # iterations so far
        #     'batch_lens': [],  # episodic lengths in batch
        #     'batch_rews': [],  # episodic returns in batch
        #     'mean': 0,  # mean of observation
        #     'std': 0,  # std of observation
        # }
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        # if self.mode == 'combined_train':
        # self.writer = SummaryWriter('runs/reptile_train/' + self.save_name + '_' + TIMESTAMP)
        # if self.mode == 'ppo_train':
        #     self.writer = SummaryWriter('runs/ppo_train/' + self.save_name + '_' + TIMESTAMP)

    def init_optimizers(self):
        self.opt_a = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def _init_hyperparameters(self, hyperparameters):
        # Default values for hyperparameters
        self.N = 5
        self.K = 16  # timesteps per batch
        self.max_timesteps_per_episode = 200  # timesteps per episode
        self.mini_batch_size = 256
        self.gamma = 0.995  # discounted factor
        self.lamda = 0.97  # advantages factor
        self.clip = 0.2
        self.lr = 3e-4
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
        self.continuous = True
        self.update_per_iteration = 10
        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))  # exec can run the python command on str format

    def set_env(self, env):
        self.env = env

    def rollout(self, running_state,memory, reward_list, continuous):
        for episode in range(self.K):
            state = self.env.reset()
            if self.state_norm:
                state = running_state(state)
            reward_sum = 0
            for t in range(self.max_timesteps_per_episode):
                t += 1
                if continuous:
                    action_mean, action_std, value = self.network(torch.tensor(state, device=self.device).float())
                    action, logproba = self.network.sample_action(action_mean, action_std)
                    action = action.detach().cpu().numpy()
                    logproba = np.float32(logproba.detach().cpu())
                else:
                    action_prob, value = self.network(torch.tensor(state, device=self.device).float())
                    action, logproba = self.network.sample_action(action_prob)
                    action = action.detach().cpu().numpy()
                    logproba = np.float32(logproba.detach().cpu())
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
        return rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size,

    def learn(self):

        continuous = self.network.continuous
        clip_now = self.clip
        running_state = ZFilter((self.obs_dim,), clip=5.0)
        # print(f"Learning... Running maximum {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        # print(f"{self.total_batch_size} timesteps per batch for a total of {self.total_epochs} epochs")
        for epoch in range(self.N):
            # step1: perform current policy to collect trajectories
            # self.logger['epoch_so_far'] = epoch + 1
            # self.logger['t_so_far'] = global_steps
            print(f"updating steps {epoch + 1}/{self.N}")
            memory = Memory()
            reward_list = []
            print(f"collecting K episodes...")
            rewards, values, masks, actions, states, oldlogproba, returns, deltas, advantages, batch_size = self.rollout(running_state,
                memory, reward_list, continuous)
            prev_return = 0
            prev_value = 0
            prev_advantage = 0

            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
                advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage * masks[i]
                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            if self.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            print(f"updating...")
            for i_epoch in range(int(self.update_per_iteration * batch_size / self.mini_batch_size)):
                minibatch_ind = np.random.choice(batch_size, self.mini_batch_size, replace=False)
                minibatch_states = states[minibatch_ind].float()
                minibatch_actions = actions[minibatch_ind].float()
                minibatch_oldlogproba = oldlogproba[minibatch_ind]
                minibatch_newlogproba = self.network.get_logproba(minibatch_states, minibatch_actions)
                minibatch_advantages = advantages[minibatch_ind]
                minibatch_returns = returns[minibatch_ind]
                minibatch_newvalues = self.network._forward_critic(minibatch_states).flatten()
                ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                surr1 = ratio * minibatch_advantages
                surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
                loss_surr = -torch.mean(torch.min(surr1, surr2))
                if self.lossvalue_norm:
                    minibatch_return_6std = 6 * minibatch_returns.std()
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
                else:
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))
                loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

                total_loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
                self.opt_a.zero_grad()
                total_loss.backward()
                self.opt_a.step()

            if self.schedule_clip == 'linear':
                ep_ratio = 1 - (epoch / self.N)
                clip_now = self.clip * ep_ratio
            if self.schedule_adam == 'linear':
                ep_ratio = 1 - (epoch / self.N)
                lr_now = self.lr * ep_ratio
                for g in self.opt_a.param_groups:
                    g['lr'] = lr_now
        return np.mean(np.array(reward_list))
        # self._log_summary()
        # if self.mode == 'combined_train':
        #     torch.save(self.network.state_dict(), 'model/combined_model/' + self.save_name + '_actor_critic.pth')
        # if self.mode == 'ppo_train':
        #     torch.save(self.network.state_dict(), 'model/direct_model/' + self.save_name + '_actor_critic.pth')
    #
    #
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
    #     self.writer.add_scalar('Average_rewards', avg_ep_rews, global_steps)
    #     avg_ep_lens = str(round(avg_ep_lens, 2))
    #     avg_ep_rews = str(round(avg_ep_rews, 2))
    #
    #     # Print logging statements
    #     print(flush=True)
    #     print(f"-------------------- Epoch #{epoch_so_far} --------------------", flush=True)
    #     print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
    #     print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
    #     print(f"Timesteps So Far: {global_steps}", flush=True)
    #     print(f"Epoch took: {delta_t} secs", flush=True)
    #     print(f"------------------------------------------------------", flush=True)
    #     print(flush=True)
    #
    #     self.logger['batch_lens'] = []
    #     self.logger['batch_rews'] = []
