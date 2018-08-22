from copy import deepcopy
import numpy as np
import json
import os
import gym
import gym.spaces
from GHER import logger
from GHER.ddpg import DDPG
from GHER.her import make_sample_her_transitions
import tensorflow as tf

from GHER.gmm_model.CONFIG import rnn_train_Config, rnn_eval_Config, rnn_sample_Config, rnn_meanStd_Config
from GHER.gmm_model.gmm_model import GMMModel
from GHER.gmm_model.gmm_train import GMMInput


def init_GMMModel():
    """
        导入已训练好的GMM模型
    """
    # config
    gmmTrain_config = rnn_train_Config()
    gmmEval_config = rnn_eval_Config()
    gmmSample_config = rnn_sample_Config()
    gmmmeanStd_config = rnn_meanStd_Config()

    with tf.name_scope("GMM_Model"):
        # session
        config_proto = tf.ConfigProto()
        # config_proto.gpu_options.per_process_gpu_memory_fraction = 0.45
        gmm_sess = tf.Session(config=config_proto)
        
        # train, eval, sample 三个模型在不同设置下重用权重.
        with tf.name_scope("Train"):
            with tf.variable_scope("Model") as scope:  # 训练
                gmmTrain = GMMModel(gmm_sess, config_str="train")

        with tf.name_scope("Eval"):
            with tf.variable_scope("Model", reuse=True) as scope:
                gmmEval = GMMModel(gmm_sess, config_str="eval")
        
        with tf.name_scope("Sample"):
            with tf.variable_scope("Model", reuse=True) as scope:
                gmmSample = GMMModel(gmm_sess, config_str="sample")

        with tf.name_scope("meanStd"):
            with tf.variable_scope("Model", reuse=True) as scope:
                gmmmeanStd = GMMModel(gmm_sess, config_str="meanStd")
    
    # load model
    print("\n\nLoad GMM Trained parameters...")
    gmmTrain.reload_model()
    print("Load done.\n\n")

    # 返回两个模型和两个config
    return gmmSample, gmmmeanStd, gmmSample_config, gmmmeanStd_config, gmmTrain, gmmTrain_config, gmmEval, gmmEval_config



DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
}


DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'GHER.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,   # 如果成立，则 goal = goal - achieved_goal
    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
}


CACHED_ENVS = {}
def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']
    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps       # T = 50 in Game FetchReach-v1
    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if type(kwargs['max_u']) == list else kwargs['max_u']  # 1.0
    kwargs['gamma'] = 1. - 1. / kwargs['T']    # gamma=49/50
    if 'lr' in kwargs:     # 学习率
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr'] 
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class', 'polyak', 
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]    # 复制一份用于DDPG
        kwargs['_' + name] = kwargs[name]  
        del kwargs[name]                    # 将这些参数的键更名为 "_"+name
    # 将DDPG的参数独立存储为一个键 "ddpg_params"
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    
    env = cached_make_env(params['make_env'])
    env.reset()
    def reward_fun(ag_2, g, info):  # vectorized
        # 当 ag_2 != g 时，奖励为-1; 当二者相等时奖励为 0
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }

    # 配置 her_params["replay_strategy"]="future"  her_params["replay_k"]=4
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]   # 参数传递给her_params后更名
        del params[name]
    
    # -------------------------------------------------
    # BAI. 导入 gmmSample 模型 并配置 her 参数
    # gmmInput = GMMInput()
    gmmSample, gmmmeanStd, gmmSample_config, gmmmeanStd_config, gmmTrain, gmmTrain_config, gmmEval, gmmEval_config = init_GMMModel()    
    
    her_params["gmmSample"] = gmmSample 
    her_params["gmmmeanStd"] = gmmmeanStd
    her_params["gmmSample_config"] = gmmSample_config
    her_params["gmmmeanStd_config"] = gmmmeanStd_config
    her_params["gmmTrain"] = gmmTrain
    her_params["gmmTrain_config"] = gmmTrain_config
    her_params["gmmEval"] = gmmEval
    her_params["gmmEval_config"] = gmmEval_config

    # -------------------------------------------------

    # 传入参数，返回 her._sample_her_transitions 函数
    # 传入的参数包括 replay_strategy='future', replay_k=4, reward_fun函数
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions   # 返回值是一个函数


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
    """
    """
    # 将参数传入 her， 返回 her._sample_her_transitions 函数
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']                                # = 0.98
    rollout_batch_size = params['rollout_batch_size']      # = 2
    ddpg_params = params['ddpg_params']  # = ddpg_params: {'buffer_size': 1000000, 'hidden': 256, 'layers': 3, 'network_class': 'GHER.actor_critic:ActorCritic', 'polyak': 0.95, 'batch_size': 256, 'Q_lr': 0.001, 'pi_lr': 0.001, 'norm_eps': 0.01, 'norm_clip': 5, 'max_u': 1.0, 'action_l2': 1.0, 'clip_obs': 200.0, 'scope': 'ddpg', 'relative_goals': False}

    input_dims = dims.copy()   # dims = {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}

    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return 当前值为50
                        'rollout_batch_size': rollout_batch_size,      # 2 
                        'subtract_goals': simple_goal_subtract,        # 对 goal 的预处理函数
                        'sample_transitions': sample_her_transitions,  # 为 her.py 中返回的函数
                        'gamma': gamma,
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    # 返回 DDPG 实例化的对象
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }

    # 在 FetchReach-v1 中，dims={'o': 10, 'u': 4, 'g': 3}
    # print(dims)

    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:     # 当 value 是标量时成立，此时将 value 扩展一个维度
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
    # dims = {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}


if __name__ == '__main__':
    init_GMMModel()