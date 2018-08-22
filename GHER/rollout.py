from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException
from GHER.util import convert_episode_to_batch_major, store_args
import GHER.experiment.config as config
import gym, time

class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called  
            policy (object): DDPG的对象
            dims (dict of ints): 维度. 例：{'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used. 一般设为2
            
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration 控制是否进行探索

            use_target_net (boolean): 控制在执行动作时使用的是 self.main 网络还是 self.target 网络
            compute_Q (boolean): 控制是否在计算输出的动作时计算Q值
            noise_eps (float): scale of the additive Gaussian noise  在动作基础上添加的高斯噪声参数，设为0.2
            random_eps (float): probability of selecting a completely random action 探索因子，设为0.3
            history_len (int): length of history for statistics smoothing 用于平滑时使用的历史长度，设置为100
            render (boolean): whether or not to render the rollouts  是否显示
        """

        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        # 用于记录
        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        # g: goal   o: observation   ag: achieved goals
        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)           # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)   # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """
            Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
            i 代表并行的 worker 的序号. 周期开始时需进行此操作
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
            连续调用 reset_rollout 函数，重设所有 worker
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """
            Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
            policy acting on it accordingly.
            rollout_batch_size = 2，表示有两个并行的 worker 进行采样
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)   # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        # self.T=50,循环产生一个周期的样本
        for t in range(self.T):
            # self.policy执行可以获得执行的动作 u 和 动作对应的Q值
            policy_output = self.policy.get_actions(                      # 该函数定义在DDPG类中
                o, ag, self.g,
                compute_Q=self.compute_Q,
                # 如果 self.exploit 为False,则会使用噪声
                noise_eps=self.noise_eps if not self.exploit else 0.,     # 是否添加
                random_eps=self.random_eps if not self.exploit else 0.,   # 是否 epsilon-greedy
                use_target_net=self.use_target_net)   # use_target_net 一般为 False

            # 提取动作和Q值. 当 rollout_batch_size=1 时，u 和 Q 存储了2个 worker 的动作和值函数
            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output
            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            # 根据选择的动作 u，执行该动作，获得新的状态 o 和 ag
            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # 对每个worker分别执行动作，得到下一步的 obs 和 ag
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']       # 提取 o
                    ag_new[i] = curr_o_new['achieved_goal']    # 提取 ag
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()

                except MujocoException as e:
                    return self.generate_rollouts()

            # warning
            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            # 记录 obs 和 ag
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())             # 动作序列
            goals.append(self.g.copy())       # g 保持不变

            # 更新，进行下一步动作选取
            o[...] = o_new
            ag[...] = ag_new
        

        # obs 和 ag 均在添加1维. 因此执行结束后，obs和ag的第一维长度为 self.T+1=51
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o                # 准备进行下一次

        # obs规模为：(51, 2, 10)   acts为(50, 2, 4)   goals为(50, 2, 3)   achieved_goals为(51, 2, 3)
        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            # print("key =", key, "value =", value)
            episode['info_{}'.format(key)] = value

        # stats  只保留 successes 最后一个元素，即只记录了周期末尾是否成功
        # successes.shape=(50,2)(worker数目为2),  successful.shape=(2,)
        successful = np.array(successes)[-1, :]                
        assert successful.shape == (self.rollout_batch_size,)  # (2,)
        
        # 成功率. 在 T=50 步内达到目标的成功率. 在 worker 之间取平均
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)  # 记录成功率
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))     # 记录Q的均值
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)         # 将 self.policy 参数 dump 在本地

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
