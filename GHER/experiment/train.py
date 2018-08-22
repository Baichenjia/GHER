# -*- coding: utf-8 -*-

import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI
import tensorflow as tf
from GHER import logger
from GHER.common import set_global_seeds
from GHER.common.mpi_moments import mpi_moments
import GHER.experiment.config as config
from GHER.rollout import RolloutWorker
from GHER.util import mpi_fork


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    # 保存网络参数的路径
    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1

    for epoch in range(n_epochs):
        print("Epoch=", epoch)
        # train
        rollout_worker.clear_history()

        for i in range(n_cycles):         # n_cycles=50
            episode = rollout_worker.generate_rollouts()  # 产生1个周期的样本
            # 调用DDPG的 store_episode 函数，进一步调用 replay_buffer 中的 store 函数
            policy.store_episode(episode, verbose=True)
            for j in range(n_batches):    # n_batches = 40
                policy.train()            # 定义在DDPG.train中，进行一次更新
            # 更新target-Q
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            epo_eval = evaluator.generate_rollouts()

        print("-----------------------------")
        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # 保存策略时文件读写与 tensorboard 中的 tf.summary 相互冲突，不能同时运行
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate > best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    

def launch(env_name, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, override_params={}, save_policies=True):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()
    # print("rank = ", rank)   # rank = 0

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)  # 创建目录用于记录日志

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS   # 字典，默认参数定义在 config.py 中
    params['env_name'] = env_name    # 添加参数 env_name
    params['replay_strategy'] = replay_strategy  # 添加参数 replay_strategy="future"
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    # 在本函数的输入中可以指定 override_params 来取代 params 中的特定参数
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)                # 将当前的所有参数设置写入文件
    # 该函数在 config.py 中，新增了 ddpg_params 键，将原有键更名为 "_"+键名
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' + 
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    # 执行后返回维度 dims = {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}
    dims = config.configure_dims(params)

    # 执行后返回 DDPG 类的实例化对象
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    # 以下参数用于控制在训练和测试中动作的选择
    rollout_params = {
        'exploit': False,
        'use_target_net': False,  # 控制动作选择时使用的是 main 网络还是 target 网络
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],  # 一般为False
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }
    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    # RolloutWorker 定义在 rollout.py 中
    # rollout_worker 的参数设置为：
    # { 'rollout_batch_size': 2, 'exploit': False, 'use_target_net': False, 
    #   'compute_Q': False, 'noise_eps': 0.2, 'random_eps': 0.3, 'history_len': 100, 
    #   'render': False, 'make_env': 函数, 'policy': DDPG类的对象, 'dims': {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}, 
    #   'logger': 类, 'use_demo_states': True, 'T': 50, 'gamma': 0.98, 'envs': [<TimeLimit<FetchReachEnv<FetchReach-v1>>>, 
    # 'info_keys': ['is_success'], 'success_history': deque([], maxlen=100), 'Q_history': deque([], maxlen=100), 'n_episodes': 0, 
    # 'g': array([[1.4879797 , 0.6269019 , 0.46735048], [1.3925381 , 0.8017641 , 0.49162573]], dtype=float32), 
    # 'initial_o': array([[ 1.3418437e+00,  7.4910051e-01,  5.3471720e-01,  1.8902746e-04, 7.7719116e-05,  3.4374943e-06, -1.2610036e-08, -9.0467189e-08, 4.5538709e-06, -2.1328783e-06],[ 1.3418437e+00,  7.4910051e-01,  5.3471720e-01,  1.8902746e-04, 7.7719116e-05,  3.4374943e-06, -1.2610036e-08, -9.0467189e-08, 4.5538709e-06, -2.1328783e-06]], dtype=float32), 
    # 'initial_ag': array([[1.3418437, 0.7491005, 0.5347172],

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)
    
    # 训练
    train(
        logdir=logdir, policy=policy,                                  # policy为DDPG类的对象
        rollout_worker=rollout_worker, evaluator=evaluator,            # rollout_worker和evaluator在 rollout.py 中定义 
        n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],  # n_epochs 为总训练周期数, n_test_rollouts=10
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],    # n_cycles=10, n_batches=40
        policy_save_interval=policy_save_interval, save_policies=save_policies)  # policy_save_interval=5, save_polices=True


@click.command()
@click.option('--env_name', type=str, default='FetchPush-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default="result/GHer-result/FetchPush/result/", help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=500, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=25, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['G-HER11-future', 'future', 'none']), 
    default='G-HER11-future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()
