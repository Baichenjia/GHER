import tensorflow as tf
from GHER.util import store_args, nn

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            # input_tf 代表输入的 tensor，包括 obs, goal, action
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)

            # dimo, g, u 分别代表 obs, goal, action 的维度
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions

            # action 需要被规约的范围
            max_u (float): the maximum magnitude of actions; action outputs will be scaled accordingly

            # o_stats和g_stats均为Normalizer的对象，用于对状态进行规约. o_stats和g_stats中保存了
                    与 obs 或 goal 对应的 mean 和 std，并进行更新. 同时提供了Normalizer函数
            o_stats (GHER.Normalizer): normalizer for observations
            g_stats (GHER.Normalizer): normalizer for goals

            # 网络结构的控制
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers

        """
        # 分别提取出 obs, goal, action 对应的 tensor
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        # 分别对 obs, goal 的tensor 进行 normalize
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        # Actor 网络
        # obs 和 goal 进行连接，组成新的状态state的表示. 作为Actor的输入，输出动作
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor
        # input_pi 为网络的输入
        # max_u 对最终输出的动作范围进行规约
        # 最后一层激活函数为 tanh，内部激活函数均为 relu
        # 每层隐含层的神经元个数均为 self.hidden，网络层数为 self.layers
        # 最后一层神经元个数为 self.dimu（动作维度）
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        
        # Critic 网络
        # Q(s,a,g) 因此网络的输入为 o, g 和 动作u
        # 网络结构: 隐含层每层神经元个数相等，为self.layers，输出仅有1个节点，
        with tf.variable_scope('Q'):
 
            # for policy training
            # 在训练 actor 时，需要将 Actor 输出的动作作为 Critic 的输入
            # Actor的目标是最大化Critic的输出，因此损失为Critic输出的相反数，即为 -self.Q_pi_tf
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            
            # for critic training
            # 在训练 Critic 时，需要输入智能体真实执行的动作 self.u_tf
            # 真实执行的动作可能在 Actor 输出的基础上添加了噪声，且梯度不会传递到 Actor
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
