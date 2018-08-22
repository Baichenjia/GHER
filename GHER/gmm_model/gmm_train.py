#-*- coding: utf-8 -*-
__author__ = "ChenjiaBai"

import collections
import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt        
import itertools
from GHER.gmm_model.CONFIG import rnn_train_Config, rnn_eval_Config, rnn_sample_Config
from GHER.gmm_model.gmm_model import GMMModel


class GMMInput:
    def __init__(self, ename=None, shuffer=True):
        """
            o.shape= (50000, 51, 25)
            u.shape= (50000, 50, 4)
            g.shape= (50000, 50, 3)
            ag.shape= (50000, 51, 3)
            info_is_success.shape= (50000, 50, 1)
            o_2.shape= (50000, 50, 25)
            ag_2.shape= (50000, 50, 3)
        """
        self.config = config = rnn_train_Config()
        self.envname = envname = config.envname
        self.batch_size_in_episode = batch_size_in_episode = config.batch_size_in_episode
        self.seq_len = seq_len = config.seq_len   
        self.seq_window = seq_window = config.seq_window
        self.T = T = config.T
        self.batch_size = batch_size = config.batch_size 
        self._next_idx = 0
        self.shuffer = shuffer

        self._keys = ['g', 'ag', 'info_is_success']
        f = os.path.join("Buffer", envname, "train.pkl")
        f = f if os.path.exists(f) else "../gmm_model/"+f
        assert os.path.exists(f)
        self.data = pickle.load(open(f, "rb"))

        for k, v in self.data.items():
            print("key:", k, ",  val:", v.shape)
        
        print("load data from ", envname)
        self.storage = self.data['g'].shape[0]   
        self._shuffle_data()
        
        # 分开训练集和测试集. 分别占 4/5 和 1/5
        self.train_data, self.valid_data = {}, {}
        for key in self._keys:
            self.train_data[key] = self.data[key][:int(self.storage*(4/5.)), :, :]
            self.valid_data[key] = self.data[key][int(self.storage*(4/5.)):, :, :]
        self.train_storage, self.valid_storage = self.train_data['g'].shape[0], self.valid_data['g'].shape[0]
        print("训练集样本数: {}, 验证集样本数: {}".format(self.train_storage, self.valid_storage))

    def _shuffle_data(self):
        """
            将 self.data 中数据的顺序打乱
        """
        idx = np.random.permutation(self.storage)
        for key in self._keys:
            if self.shuffer:
                self.data[key] = self.data[key][idx, :, :]
            else:
                pass

    def _encode_sample(self, idxes, tv_data):
        """
            从经验池中将idxes取出. 随后提取指定的key，连接指定key，组成矩阵
        """
        # print("---", type(idxes), idxes.shape, idxes[0])
        batches = []
        for idx in idxes:
            # ag
            idx_ag = tv_data['ag'][idx]
            assert idx_ag.shape == (self.T+1, self.config.ag_len)
            # step 表示每个元素在周期中所属的位置
            idx_step = (np.arange(0, self.T+1) / float(self.T)).reshape(int(self.T)+1, 1) 

            # 以下为序列中不变化的量
            # goal
            idx_g = tv_data['g'][idx][-1]    # (3,)

            # 从末尾从前数，成功的时间步占整个周期的比例
            info_succ = tv_data['info_is_success'][idx, :, 0][::-1]
            info_succ_dic = [(k, list(v)) for k, v in itertools.groupby(info_succ)]
            idx_success_rate = 0. if info_succ_dic[0][0] == 0 else len(info_succ_dic[0][1])/float(self.T)
            
            # 周期结尾是否成功
            idx_done = float(tv_data['info_is_success'][idx, -1, 0]) 

            # 整个周期的平均位置与目标的距离 / 起点与目标的距离
            mean_ag = np.mean(tv_data['ag'][idx, -5:, :], axis=0)        # 最后5个step的平均位置 (3,)
            start_ag = tv_data['ag'][idx, 0, :]                          # 起始位置 (3,)
            idx_dis = np.linalg.norm(mean_ag-idx_g) / np.linalg.norm(start_ag-idx_g)  # 标量，相当于比例，越小越好
 
            # 按照顺序 ag(x,y,z), g(x,y,z), step, success_rate, success_first 进行整理
            train_idx = np.hstack([idx_ag,                                        # ag     (51, 3)
                                   np.tile(idx_g, (self.T+1, 1)),                 # g      (51, 3)
                                   idx_step,                                      # step   (51, 1)
                                   np.tile(idx_success_rate, (self.T+1, 1)),      # success_rate  (51, 1)
                                   np.tile(idx_done, (self.T+1, 1)),              # idx_done  (51, 1)
                                   np.tile(idx_dis, (self.T+1, 1))                # idx_dis  (51, 1)
                                ])                   
            assert train_idx.shape == (self.T+1, self.config.input_size)
            batches.append(train_idx)

        # 整合特征，处理成numpy
        batches_np = np.stack(batches)
        # assert batches_np.shape == (self.batch_size_in_episode, 51, 10)   
        self.batches_np = batches_np
        X = batches_np[:, :self.T, :]   
        Y = batches_np[:, 1:, :]         # 错位
        # assert X.shape == Y.shape == (self.batch_size_in_episode, 50, 10)

        X_batch = []
        Y_batch = []
        for i in range(0, self.T-self.seq_len+1, self.seq_window):   # 0,2,4,6,...,38,40
            x = X[:, i: i+self.seq_len]
            y = Y[:, i: i+self.seq_len]  
            # assert x.shape == y.shape == (self.batch_size_in_episode, self.seq_len, 10)  # shape=(50, 10, 9)
            X_batch.append(x)
            Y_batch.append(y)
        
        # vstack
        X_train = np.vstack(X_batch)     
        Y_train = np.vstack(Y_batch)    
        return X_train, Y_train, batches_np


    def create_batch(self, mode="train"):
        """
            mode 控制从 self.train_data 或 self.valid_data 中进行采样
            train 数据进行顺序采样, valid 随机采样
        """
        assert mode in ["train", "valid"]

        if mode == 'train':
            batch = self.batch_size_in_episode
            if self._next_idx + batch > self.train_storage:
                self._next_idx = 0
            idxes = np.arange(self._next_idx, self._next_idx + batch)
            self._next_idx = (self._next_idx + batch) % self.train_storage
            X, Y, _ = self._encode_sample(idxes, self.train_data)

        elif mode == 'valid':
            idxes = np.random.randint(0, self.valid_storage, 1000)
            X, Y, _ = self._encode_sample(idxes, self.valid_data)

        return X, Y


def TRAIN(epoch_num):
    # config
    train_config = rnn_train_Config()
    eval_config = rnn_eval_Config()
    sample_config = rnn_sample_Config()
    
    # build model
    with tf.name_scope("GMM_Model"):
        # session
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.per_process_gpu_memory_fraction = 0.7
        rnn_sess = tf.Session(config=config_proto)
        
        # train, eval, sample 三个模型在不同设置下重用权重.
        with tf.name_scope("Train"):
            with tf.variable_scope("Model") as scope:  # 训练
                rnn_m = GMMModel(rnn_sess, config_str="train")

        with tf.name_scope("Eval"):
            with tf.variable_scope("Model", reuse=True) as scope:
                rnn_eval = GMMModel(rnn_sess, config_str="eval")

        with tf.name_scope("Sample"):
            with tf.variable_scope("Model", reuse=True) as scope:
                rnn_sample = GMMModel(rnn_sess, config_str="sample")
        
    with tf.name_scope("GMM_Input"):
        gmmInput = GMMInput()
    
    rnn_sess.run(tf.global_variables_initializer())

    # 学习率
    init_lr = train_config.init_lr                    # 初始学习率
    decay = train_config.lr_decay                     # 学习率衰减

    # 循环
    min_loss = float("inf")
    train_losses, eval_losses = [], []
    for epoch in range(epoch_num):
        # 学习率更新
        rnn_sess.run(tf.assign(rnn_m.lr, init_lr * (decay ** epoch)))        
        
        # 提取一个批量数据，训练 train_steps 次
        train_cost = rnn_m.train_epoch(gmmInput, epoch, train_steps=8)   # train_steps修改
        train_losses.append(train_cost)

        # Eval
        X, Y = gmmInput.create_batch(mode="valid")
        eval_loss = rnn_eval.eval(X, Y)
        print("EVAL loss:", eval_loss, "\n------------") 
        eval_losses.append(eval_loss)

        # sample模型保存，用于产生样本
        if epoch % 20 == 0:
            savename = "model-"+str(epoch)+"-({:.4})".format(eval_loss)+".ckpt"
            rnn_m.save_model(savename)
            rnn_sample.save_model(savename)
    rnn_m.save_model("model-last.ckpt")
    
    print("average eval loss in all cycles: \t", np.mean(eval_losses))
    print("average eval loss in last 50 cycles: \t", np.mean(eval_losses[-50:]))
    

if __name__ == '__main__':
    TRAIN(epoch_num=2000)
    

