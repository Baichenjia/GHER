from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from GHER import logger
from GHER.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from GHER.normalizer import Normalizer
from GHER.replay_buffer import ReplayBuffer
from GHER.common.mpi_adam import MpiAdam


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'GHER.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        
        # # print("\n\n\n\n1--", input_dims, "\n2--", buffer_size, "\n3--", hidden, 
        #         "\n4--", layers, "\n5--", network_class, "\n6--", polyak, "\n7--", batch_size,
        #          "\n8--", Q_lr, "\n9--", pi_lr, "\n10--", norm_eps, "\n11--", norm_clip, 
        #          "\n12--", max_u, "\n13--", action_l2, "\n14--", clip_obs, "\n15--", scope, "\n16--", T,
        #          "\n17--", rollout_batch_size, "\n18--", subtract_goals, "\n19--", relative_goals, 
        #          "\n20--", clip_pos_returns, "\n21--", clip_return,
        #          "\n22--", sample_transitions, "\n23--", gamma)

        """
        在FetchReach-v1运行中参数值示例：
            input_dims (dict of ints):  {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}  （o,u,g均作为网络的输入） 
            buffer_size (int):  1E6     (经验池样本总数)
            hidden (int): 256          （隐含层神经元个数）
            layers (int): 3            （三层神经网络）
            network_class (str):        GHER.ActorCritic'
            polyak (float): 0.95       （target-Network更新的平滑的参数）
            batch_size (int): 256      （批量大小）
            Q_lr (float): 0.001         (学习率)
            pi_lr (float): 0.001        (学习率)
            norm_eps (float): 0.01      (为避免数据溢出使用)
            norm_clip (float): 5        (norm_clip)
            max_u (float): 1.0          (动作的范围是[-1.0, 1.0])
            action_l2 (float): 1.0      (Actor网络的损失正则项系数)
            clip_obs (float): 200       (obs限制在 (-200, +200))
            scope (str): "ddpg"         (tensorflow 使用的 scope 命名域)
            T (int): 50                 (周期的交互次数)
            rollout_batch_size (int): 2 (number of parallel rollouts per DDPG agent)
            subtract_goals (function):  对goal进行预处理的函数， 输入为a和b，输出a-b
            relative_goals (boolean):   False  (如果需要对goal进行函数subtract_goals处理，则为True）
            clip_pos_returns (boolean): True   (是否需要将正的return消除)
            clip_return (float): 50     (将return的范围限制在[-clip_return, clip_return])
            sample_transitions (function):  her返回的函数. 参数由 config.py 定义
            gamma (float): 0.98         (Q 网络更新时使用的折扣因子)

            其中 sample_transition 来自与 HER 的定义，是关键部分
        """

        if self.clip_return is None:
            self.clip_return = np.inf

        # 网络结构和计算图的创建由 actor_critic.py 文件完成
        self.create_actor_critic = import_function(self.network_class)

        # 提取维度
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']   # 10
        self.dimg = self.input_dims['g']   # 4
        self.dimu = self.input_dims['u']   # 3
        # print("+++", input_shapes)    #  {'o': (10,), 'u': (4,), 'g': (3,), 'info_is_success': (1,)}

        # https://www.tensorflow.org/performance/performance_models
        # StagingArea 提供了更简单的功能且可在 CPU 和 GPU 中与其他阶段并行执行。
        #       将输入管道拆分为 3 个独立并行操作的阶段，并且这是可扩展的，充分利用大型的多核环境

        # 定义需要的存储变量. 假设 self.dimo=10, self.dimg=5, self.dimu=5
        # 则 state_shapes={'o':(None, 10), 'g':(None, 5), 'u':(None:5)}
        # 同时添加target网络使用的变量 state_shapes={'o_2':(None, 10), 'g_2': (None, 5)}
        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)       # 奖励为标量 
        self.stage_shapes = stage_shapes
        # 执行后 self.stage_shapes = 
        #       OrderedDict([('g', (None, 3)), ('o', (None, 10)), ('u', (None, 4)), ('o_2', (None, 10)), ('g_2', (None, 3)), ('r', (None,))])
        # 其中包括 g, o, u、target网络中使用的 o_2, g_2 和奖励 r

        # Create network.
        # 根据 state_shape 创建 tf 变量，其中包括 g, o, u, o_2, g_2, r
        # self.buffer_ph_tf = [<tf.Tensor 'ddpg/Placeholder:0' shape=(?, 3) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_1:0' shape=(?, 10) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_2:0' shape=(?, 4) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_3:0' shape=(?, 10) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_4:0' shape=(?, 3) dtype=float32>, 
        #                     <tf.Tensor 'ddpg/Placeholder_5:0' shape=(?,) dtype=float32>]
        with tf.variable_scope(self.scope):
            # 创建 StagingArea 变量
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            # 创建 Tensorflow 变量 placeholder
            self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) 
                for shape in self.stage_shapes.values()]
            # 将 tensorflow 变量与 StagingArea 变量相互对应
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            #
            self._create_network(reuse=reuse)

        # 经验池相关操作
        # 当T = 50时，执行结束后 buffer_shapes=
        #         {'o': (51, 10), 'u': (50, 4), 'g': (50, 3), 'info_is_success': (50, 1), 'ag': (51, 3)}
        # 注意 a,g,u 均记录一个周期内经历的所有样本，因此为 50 维，但 o 和 ag 需要多1维 ？？？？
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}      # 
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)     #
        buffer_shapes['ag'] = (self.T+1, self.dimg)                 #
        # print("+++", buffer_shapes)

        # buffer_size 是按照样本进行计数的长度
        # self.buffer_size=1E6  self.rollout_batch_size=2 buffer_size=1E6
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def _random_action(self, n):
        """
            从 [-self.max_u, +self.max_u] 中随机采样 n 个动作
        """
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        """
            obs, goal, achieve_goal 进行预处理
            如果 self.relative_goal=True，则 goal = goal - achieved_goal
        """
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)    # 增加1维
            ag = ag.reshape(-1, self.dimg)  # 增加1维
            g = self.subtract_goals(g, ag)  # g = g - ag
            g = g.reshape(*g_shape)         
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        """
            根据 self.main 网络选择动作，随后添加高斯噪声，clip，epsilon-greedy操作，输出处理后的动作
        """
        # 如果 self.relative_goal=True，则对 goal 进行预处理. 否则只进行 clip
        o, g = self._preprocess_og(o, ag, g)
        # 在调用本类的函数 self._create_network 后，创建了 self.main 网络和 self.target 网络，均为 ActorCritic 对象
        policy = self.target if use_target_net else self.main   # 根据 self.main 选择动作
        # actor 网络输出的动作的 tensor
        vals = [policy.pi_tf]

        # print("+++")
        # print(vals.shape)

        # 将 actor 输出的 vals 再次输入到 critic 网络中，获得输出为 Q_pi_tf
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed_dict的构建，包括 obs, goal 和 action，作为 Actor和Critic的输入 
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }
        # 执行当前的策略网络，输出ret.  ret[0]代表action, ret[1]代表Q值
        ret = self.sess.run(vals, feed_dict=feed)
        
        # action postprocessing
        # 对Action添加高斯噪声. np.random.randn 指从一个高斯分布中进行采样，噪声服从高斯分布
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)   # 添加噪声后进行 clip
        
        # 进行 epsilon-greedy 操作，epsilon为random_eps
        # np.random.binomial指二项分布，输出的结果是0或1，其中输出为1的概率为 random_eps.
        # 如果二项分布输出0，则 u+=0相当于没有操作；如果输出为1，则 u = u + (random_action - u) = random_action
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u
        # 
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True, verbose=False):
        """
            episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
            调用 replay_buffer 中的 store_episode 函数对一个采样周期的样本进行存储
            o_stats 和 g_stats 分别更新和存储 obs 和 goal 的均值和标准差，并定期更新
        """

        # episode_batch 存储了 rollout.py 中 generate_rollout 产生的一个周期样本
        # episode_batch 是一个字典，键包括 o, g, u, ag, info，值的shape分别为
        #      o (2, 51, 10), u (2, 50, 4), g (2, 50, 3), ag (2, 51, 3), info_is_success (2, 50, 1)
        # 其中第1维是 worker的数目，第2维由周期长度决定

        self.buffer.store_episode(episode_batch, verbose=verbose)

        # 更新 o_stats 和 g_stats 的均值和标准差
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]    # 提取出 next_obs 和 next_state
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)   # 将周期转换为总样本数

            # 调用 sample_transitions 中的采样函数
            # episode_batch是一个字典，键和元素shape分别为 o (2, 51, 10) u (2, 50, 4) g (2, 50, 3) ag (2, 51, 3) info_is_success (2, 50, 1)
            #                                          o_2 (2, 50, 10)  ag_2 (2, 50, 3)
            # num_normalizing_transitions=100，原有是有 2 个 worker，每个 worker 含有1个周期的50个样本
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            # 采样出的样本经过预处理后，用于更新计算 o_stats 和 g_stats，定义在Normalizer中，用于存储 mean 和 std
            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        """
            返回当前经验池的样本数量
        """
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        """
            Q_adam 和 pi_adam 为更新 actor网络 和 critic网络的运算符
        """
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        """
            返回损失函数和梯度
            Q_loss_tf, main.Q_pi_tf, Q_grad_tf, pi_grad_tf 均定义在 _create_network 函数中
        """

        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        """
            更新 main 的 Actor 和 Critic 网络
            更新的 op 均定义在 _create_network 中
        """
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        """
            调用 replay_buffer.py 中的 sample 函数进行采样，后者调用的采样方法来源于 her.py 中的定义
            返回的样本组成 batch，用于 self.stage_batch 函数中构建 feed_dict
            feed_dict将作为 选择动作 和 更新网络参数 的输入

            调用采样一个批量的样本，随后对 o 和 g 进行预处理. 样本的 key 包括 o, o_2, ag, ag_2, g
        """
        # 调用sample后返回transition为字典, key 和 val.shape:
        # o (256, 10) u (256, 4) g (256, 3) info_is_success (256, 1) ag (256, 3) o_2 (256, 10) ag_2 (256, 3) r (256,)
        # print("In DDPG: ", self.batch_size)
        transitions = self.buffer.sample(self.batch_size)
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))
        # tensorboard可视化
        self.tfboard_sample_batch = batch
        self.tfboard_sample_tf = self.buffer_ph_tf
  

    def train(self, stage=True):
        """
            计算梯度，随后更新
            train 中执行参数更新之前先执行了 self.stage_batch，用于构建训练使用的feed_dict. 该函数中调用了 
                    self.sample_batch 函数，后者又调用了 self.buffer.sample，后者调用了 config.py 中的 config_her, 后者对 her.py 的函数进行参数配置.
            train 中的运算符在 self._create_network 中定义.
        """
        if stage:
            self.stage_batch()         # 返回使用 her.py 的采样方式构成的 feed_dict 用于计算梯度
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        """
            更新 target 网络，update_target_net_op 定义在函数 _create_network 中
        """
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        """
            定义计算 Actor 和 Critic 损失所需要的计算流图
        """
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        # running averages
        # 分别定义用于规约 obs 和 goal 的 Normalizer 对象
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        # 用于存储一个批量样本的数据结构，为OrderedDict，执行后 batch_tf 如下:
        # OrderedDict([('g', <tf.Tensor 'ddpg/ddpg/StagingArea_get:0' shape=(?, 3) dtype=float32>), 
        #              ('o', <tf.Tensor 'ddpg/ddpg/StagingArea_get:1' shape=(?, 10) dtype=float32>), 
        #              ('u', <tf.Tensor 'ddpg/ddpg/StagingArea_get:2' shape=(?, 4) dtype=float32>), 
        #              ('o_2', <tf.Tensor 'ddpg/ddpg/StagingArea_get:3' shape=(?, 10) dtype=float32>), 
        #              ('g_2', <tf.Tensor 'ddpg/ddpg/StagingArea_get:4' shape=(?, 3) dtype=float32>), 
        #              ('r', <tf.Tensor 'ddpg/Reshape:0' shape=(?, 1) dtype=float32>)])
        # 定义的 batch_tf 变量将作为神经网络的输入

        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # 
        # 根据 ActorCritic.py 创建 main network
        # 在创建 ActorCritic 网络时，不需要显式的传参，利用 self.__dict__将DDPG类的对应参数直接赋值给 ActorCritic 的对应参数
        # print(self.main.__dict__)
        # {'inputs_tf': OrderedDict([('g', <tf.Tensor 'ddpg/ddpg/StagingArea_get:0' shape=(?, 3) dtype=float32>), ('o', <tf.Tensor 'ddpg/ddpg/StagingArea_get:1' shape=(?, 10) dtype=float32>), ('u', <tf.Tensor 'ddpg/ddpg/StagingArea_get:2' shape=(?, 4) dtype=float32>), ('o_2', <tf.Tensor 'ddpg/ddpg/StagingArea_get:3' shape=(?, 10) dtype=float32>), ('g_2', <tf.Tensor 'ddpg/ddpg/StagingArea_get:4' shape=(?, 3) dtype=float32>), ('r', <tf.Tensor 'ddpg/Reshape:0' shape=(?, 1) dtype=float32>)]), 
        # 'net_type': 'main', 'reuse': False, 'buffer_size': 1000000, 'hidden': 256, 'layers': 3, 'network_class': 'GHER.actor_critic:ActorCritic', 
        # 'polyak': 0.95, 'batch_size': 256, 'Q_lr': 0.001, 'pi_lr': 0.001, 'norm_eps': 0.01, 'norm_clip': 5, 'max_u': 1.0, 
        # 'action_l2': 1.0, 'clip_obs': 200.0, 'scope': 'ddpg', 'relative_goals': False, 'input_dims': {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}, 
        # 'T': 50, 'clip_pos_returns': True, 'clip_return': 49.996, 'rollout_batch_size': 2, 'subtract_goals': <function simple_goal_subtract at 0x7fcf72caa510>, 'sample_transitions': <function make_sample_her_transitions.<locals>._sample_her_transitions at 0x7fcf6e2ce048>, 
        # 'gamma': 0.98, 'info': {'env_name': 'FetchReach-v1'}, 'use_mpi': True, 'create_actor_critic': <class 'GHER.actor_critic.ActorCritic'>, 
        # 'dimo': 10, 'dimg': 3, 'dimu': 4, 'stage_shapes': OrderedDict([('g', (None, 3)), ('o', (None, 10)), ('u', (None, 4)), ('o_2', (None, 10)), ('g_2', (None, 3)), ('r', (None,))]), 'staging_tf': <tensorflow.python.ops.data_flow_ops.StagingArea object at 0x7fcf6e2dddd8>, 
        # 'buffer_ph_tf': [<tf.Tensor 'ddpg/Placeholder:0' shape=(?, 3) dtype=float32>, <tf.Tensor 'ddpg/Placeholder_1:0' shape=(?, 10) dtype=float32>, <tf.Tensor 'ddpg/Placeholder_2:0' shape=(?, 4) dtype=float32>, <tf.Tensor 'ddpg/Placeholder_3:0' shape=(?, 10) dtype=float32>, <tf.Tensor 'ddpg/Placeholder_4:0' shape=(?, 3) dtype=float32>, <tf.Tensor 'ddpg/Placeholder_5:0' shape=(?,) dtype=float32>], 
        # 'stage_op': <tf.Operation 'ddpg/ddpg/StagingArea_put' type=Stage>, 'sess': <tensorflow.python.client.session.InteractiveSession object at 0x7fcf6e2dde10>, 'o_stats': <GHER.normalizer.Normalizer object at 0x7fcf6e2ee940>, 'g_stats': <GHER.normalizer.Normalizer object at 0x7fcf6e2ee898>, 
        # 'o_tf': <tf.Tensor 'ddpg/ddpg/StagingArea_get:1' shape=(?, 10) dtype=float32>, 'g_tf': <tf.Tensor 'ddpg/ddpg/StagingArea_get:0' shape=(?, 3) dtype=float32>, 'u_tf': <tf.Tensor 'ddpg/ddpg/StagingArea_get:2' shape=(?, 4) dtype=float32>, 'pi_tf': <tf.Tensor 'ddpg/main/pi/mul:0' shape=(?, 4) dtype=float32>, 'Q_pi_tf': <tf.Tensor 'ddpg/main/Q/_3/BiasAdd:0' shape=(?, 1) dtype=float32>, '_input_Q': <tf.Tensor 'ddpg/main/Q/concat_1:0' shape=(?, 17) dtype=float32>, 'Q_tf': <tf.Tensor 'ddpg/main/Q/_3_1/BiasAdd:0' shape=(?, 1) dtype=float32>}

        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()

        # o_2, g_2 用来创建 target network
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']   # 由于 target 网络用于计算 target-Q 值，因此 o 和 g 需使用下一个状态的值 
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        # 计算Critic的target-Q值，需要用到Actor的target网络 和 Critic的target网络
        # target_Q_pi_tf 使用的是下一个状态 o_2 和 g_2
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range) 
        # Critic 的损失函数为 target_tf 与 Q_tf 的差的平方，注意梯度不通过target_tf进行传递
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))

        # Actor的损失函数为 main 网络中将actor的输出随后输入到critic网络中得到Q值的相反数
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        # Actor中加入正则
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        
        # 计算梯度 
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))     # 梯度和变量名进行对应
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')        # 将Actor和Critic网络的参数放在一起
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(            # target 初始化操作中，main网络参数直接赋值给target
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(          # target 更新操作中，需要将 main 网络和 target 网络按照参数 polyak 加权
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # # tensorboard可视化
        # tf.summary.scalar("Q_target-Q-mean", tf.reduce_mean(target_tf))
        # tf.summary.histogram("Q_target-Q", target_tf)
        # tf.summary.scalar("Q_Td-error-mean", tf.reduce_mean(target_tf - self.main.Q_tf))
        # tf.summary.histogram("Q_Td-error", target_tf - self.main.Q_tf)
        # tf.summary.scalar("Q_reward-mean", tf.reduce_mean(batch_tf['r']))
        # tf.summary.histogram("Q_reward", batch_tf['r'])
        # tf.summary.scalar("Q_loss_tf", self.Q_loss_tf)
        # tf.summary.scalar("pi_loss_tf", self.pi_loss_tf)
        # self.merged = tf.summary.merge_all()

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def tfboard_func(self, summary_writer, step):
        """
            tensorboard可视化
        """
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.tfboard_sample_tf, self.tfboard_sample_batch)))
        summary = self.sess.run(self.merged) 
        summary_writer.add_summary(summary, global_step=step)

        print("S"+str(step), end=",")

    def __getstate__(self):
        """
            Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    # -----------------------------------------
    def updata_loss_all(self, verbose=False):
        assert self.buffer.current_size > 0
        idxes = np.arange(self.buffer.current_size)
        print("--------------------------------------")
        print("Updata All loss start...")
        self.buffer.update_rnnLoss(idxes, verbose=verbose)
        print("Updata All loss end ...")


