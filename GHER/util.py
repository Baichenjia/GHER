import os
import subprocess
import sys
import importlib
import inspect
import functools
import tensorflow as tf
import numpy as np
from GHER.common import tf_util as U


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def flatten_grads(var_list, grads):
    """
        Flattens a variables and their gradients.
        连接所有梯度成一个大的tensor,在第0维进行链接
    """
    return tf.concat([tf.reshape(grad, [U.numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """
        Creates a simple neural network
        创建一个全连接的神经网络，除了输出层之外激活函数均为relu
        layer_size 是一个list，决定每层的神经元个数
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes)-1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name+'_'+str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def install_mpi_excepthook():
    """
        ?? 多线程相关
    """
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook


def mpi_fork(n):
    """
        Re-launches the current script with workers
        Returns "parent" for original parent, "child" for MPI children
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        args = [
            "mpirun",
            "-np",
            str(n),
            "-bind-to",
            "core",
            sys.executable
        ]
        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
        该函数在 rollout.py 中 generate_rollout 的最后被调用

    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    # 转换前 episode 中每个 key 对应的变量的 shape 为
    # o (51, 2, 10), u (50, 2, 4), g (50, 2, 3), ag (51, 2, 3) info_is_success (50, 2, 1)
    # 其中 2 代表 worker 的个数，由参数 rollout_batch_size 决定

    # 转换后为
    # o (2, 51, 10), u (2, 50, 4), g (2, 50, 3), ag (2, 51, 3) info_is_success (2, 50, 1)

    return episode_batch


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """Reshapes a tensor (source) to have the correct shape and dtype of the target
    before broadcasting it with MPI.
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim-1)) + [-1]
    # print(shape)  # 当dim=3时，shape为[1,1,-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)


if __name__ == '__main__':
    # 测试 reshape_for_broadcasting
    a = tf.constant(np.zeros((3,4,5)))
    res = reshape_for_broadcasting([100, 100], a)
    sess = tf.Session()
    print(sess.run(res), res.get_shape())  # [[[100. 100.]]] (1, 1, 2)
