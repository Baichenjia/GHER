import threading
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """
            buffer_shapes (dict of ints): the shape fo r all buffers that are used in the replay buffer
                buffer_shapes = {'o': (51, 10), 'u': (50, 4), 'g': (50, 3), 'info_is_success': (50, 1), 'ag': (51, 3)}
        """
        self.buffer_shapes = buffer_shapes            # {'o': (51, 10), 'u': (50, 4), 'g': (50, 3), 'info_is_success': (50, 1), 'ag': (51, 3)}
        self.size = size_in_transitions // T          # 除以T得到buffer中最多存储的周期个数  1E6/50=20000
        self.T = T                                    # 每个周期的样本数
        self.sample_transitions = sample_transitions  # 从经验池中采样的函数 

        # 每个key对应一个空的Numpy矩阵，key为 'o','ag','g','u'. val的 shape 为 [以周期计的容量 * (T/T+1 * 每种的维度)]
        # 具体而言：o (20000, 51, 10)  u (20000, 50, 4)  g (20000, 50, 3)  info_is_success (20000, 50, 1) ag (20000, 51, 3)
        self.buffers = {key: np.empty([self.size, *shape]) for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        # 线程锁. 多个 worker 需要操作同一个经验池，因此在存取样本时需要加锁
        self.lock = threading.Lock()

    @property
    def full(self):
        """
            返回经验池是否已满
        """
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """
            采样. 本函数调用了 her.py 中定义的采样函数，根据虚拟的 goal 重新计算了奖励
            本函数由 ddpg.py 中的 sample_batch 函数调用.
            Returns a dict {key: array(batch_size x shapes[key])}
        """

        # buffers 为从经验池中截取的 已经填充数据的部分
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        # o_2, ag_2 指的是 next_obs 和 next_ag. 在周期中，T=0时，o_2和ag_2对应的应该是T=1的值
        # 因此截取后， o_2 和 ag_2 是 o,ag 对应序号上的下一个状态的值
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # 根据 her 中定义的产生虚拟 goal 的方法重新计算奖励
        # buffers 存储当前经验池中所有数据，作为输入. batch_size
        transitions = self.sample_transitions(buffers, batch_size)  # self.batch_size=256

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch, verbose=False):
        """
            episode_batch: array(batch_size x (T or T+1) x dim_key)
            episode_batch是一个字典，keyh和val.shape分别为
                         o (2, 51, 10), u (2, 50, 4), g (2, 50, 3), ag (2, 51, 3), info_is_success (2, 50, 1)
        """

        # 提取每个键的第1维长度，执行后为 [2, 2, 2, 2, 2]
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]

        assert np.all(np.array(batch_sizes) == batch_sizes[0])   # np.all用于比较两个数组对应位置元素是否相等
        batch_size = batch_sizes[0]    # rollout 时代表 worker 的个数

        with self.lock:
            idxs = self._get_storage_idx(batch_size)   # idxs是一个list, 长度等于 batch_size

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]  # 在对应的key处存储对应的元素

            self.n_transitions_stored += batch_size * self.T  # 记录存储的元素个数（按transition记）

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        """
            经验池满后采用随机替换原则, 而不是采用去除最老样本的原则
            inc 代表待存储的元素个数. 
            返回一个List，表示这些元素应该存在经验池的哪些地方（序号）
            传参时 batch_size=inc，表示 rollout 中使用的 worker 个数，存储时，待存的数组规模为 inc*self.T*ndim
        """
        inc = inc or 1   # 如果 inc=None, 则变为1；否则不变
        assert inc <= self.size, "Batch committed to replay is too large!"   # self.size代表经验池容量
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:                         # 如果存储后经验池未满
            idx = np.arange(self.current_size, self.current_size+inc)  # arange后list，代表待存区域的序号
        elif self.current_size < self.size:                            # 如果存储后经验池满，则先存满，随后随机选择
            overflow = inc - (self.size - self.current_size)           # 溢出的样本个数
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx

