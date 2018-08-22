# -* cofing: utf-8 -*-
import numpy as np
import os

class rnn_train_Config(object):
    def __init__(self):

        # choose the different envname to train the Conditional RNN model
        self.envname = "FetchPush"
        # self.envname = "FetchReach"
        # self.envname = "FetchPickAndPlace"
        # self.envname = "FetchSlide"

        # 训练相关
        self.grad_clip = 1                      # clip the gradient of RNN
        self.T = 50                             # episode length  
        self.is_training = True                 # if training
        self.init_lr = 0.001                    # initial learning rate
        self.lr_decay = 0.99                    # learning reate decay

        # 数据相关
        self.seq_len = 40                       # seq_len
        self.seq_window = 1                     # sliding window size set to 1 or 2
        self.input_size = 10                    # input dims to RNN model
        self.ag_len = 3
        self.output_size = self.input_size      # output dims to RNN model
        self.batch_size_in_episode = 100        # batchsize
        self.batch_size = self.batch_size_in_episode * np.arange(0, self.T-self.seq_len+1, self.seq_window).shape[0]

        # RNN weight save path
        self.basepath = os.path.join("gmm_save", self.envname, "warm3-step40-config-all-data-same")
        self.save_path = os.path.join(self.basepath, "train")     # checkpoints
        
        # GMM
        self.num_mixture = 3                         # components in GMM
        self.include_rol = False                     # rol

        # warm_start
        self.warm_start = True                       # warm_up step
        self.warm_num = 3                            # n = 3
        
        # loss weight in sequence
        loss_weight = np.array([1.] * self.seq_len)  # same
        self.loss_weight = None
        if self.warm_start == False:
            self.loss_weight = loss_weight
        else:
            self.loss_weight = np.concatenate([[0.]*self.warm_num, loss_weight], axis=0)[:self.seq_len]
        assert self.loss_weight.shape == (self.seq_len,)

        # architecture of RNN
        self.num_layers = 1                    # LSTM layers
        self.keep_prob = 1.                    # dropout is not used
        self.hidden_size = 1000                # LSTM hidden units
        self.rnn_type = "lstm"                 # use "lstm", "gru" or "rnn"
        self.dense1_num = 500                  # fully connected layer units

        # output size of RNN model, include: pi, mu1, mu2, mu3, sigma1, sigma2, sigma3.
        self.NOUT = self.num_mixture * 7


# Evaluation config class
class rnn_eval_Config(rnn_train_Config):
    def __init__(self):
        super(rnn_eval_Config, self).__init__()
        self.keep_prob = 1 
        self.batch_size_in_episode = 1000 
        self.batch_size = self.batch_size_in_episode * np.arange(0, self.T-self.seq_len+1, self.seq_window).shape[0]
        self.is_training = False                               
        self.save_path = os.path.join(self.basepath, "eval")
        print(self.loss_weight)


# sample from RNN 
class rnn_sample_Config(rnn_train_Config):
    def __init__(self, all_batch=None):
        super(rnn_sample_Config, self).__init__()
        print("rnn_sample_Config:", 256 if all_batch is None else all_batch, "------")

        self.keep_prob = 1                             
        self.seq_len = 1                               # output one elements each time step
        self.is_training = False                       
        self.save_path = os.path.join(self.basepath, "sample")
        self.loss_weight = [1.]
        
        self.method = "max"                             # "max", "softmax", "random"
        self.all_batch = 256 if all_batch is None else all_batch

        # ratio of 3 different goals.
        self.ratio1 = 0.2                                   # ratio of original-goals
        self.ratio2 = 0.4                                   # ratio of HER-goals
        self.ratio3 = 1.-(self.ratio1+self.ratio2)          # ratio of RNN generated goals
        
        # use a specific policy level C^{gen} to sample 
        self.increase_level = True
        self.batch_size = self.batch_size_in_episode = self.all_batch-int(
                    self.all_batch*self.ratio1)-int(self.all_batch*self.ratio2)


# update the mean and std in normalizer class
class rnn_meanStd_Config(rnn_sample_Config):
    def __init__(self):
        all_batch = 100
        super(rnn_meanStd_Config, self).__init__(all_batch)


# 用于 her_sample.py 中进行测试
class rnn_test_Config(rnn_sample_Config):
    def __init__(self):
        super(rnn_test_Config, self).__init__()
        self.batch_size = self.batch_size_in_episode = 5
