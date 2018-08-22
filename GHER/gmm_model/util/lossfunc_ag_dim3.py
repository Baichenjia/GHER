# -*- coding: utf-8

import tensorflow as tf
import numpy as np

def lossfunc(z_pi, z_mu1, z_mu2, z_mu3, z_sigma1, z_sigma2, z_sigma3, x1_data, x2_data, x3_data, config):
    """
        z_pi, z_mu1, z_mu2, z_mu3, z_sigma1, z_sigma2, z_sigma3 shape = 
                                            (config.batch_size*config.seq_len, config.num_mixture)
        x1_data, x2_data, x3_data.shape = (config.batch_size * config.seq_len, 1)
    """
    tfd = tf.contrib.distributions
    num_mixture = int(z_pi.shape[-1])   # GMM混合单元个数  config.num_mixture
    batch_size = int(z_pi.shape[0])     # config.batch_size*config.seq_len
    s = 3                               # 坐标维度
    
    mvn_list = []
    for i in range(num_mixture):
        loc = tf.concat([z_mu1[:, i:i+1], z_mu2[:, i:i+1], z_mu3[:, i:i+1]], axis=1)      # (10, 3)
        assert loc.shape == (config.batch_size*config.seq_len, s)
        
        scale_diag = tf.concat([z_sigma1[:, i:i+1], z_sigma2[:, i:i+1], z_sigma3[:, i:i+1]], axis=1)  # (10, 3)
        assert scale_diag.shape == (config.batch_size*config.seq_len, s)

        # model
        mvn = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        
        # prob
        # print("\nMixture: ", i)
        # print("MVN.mean :", mvn.mean())                 # (10, 3)
        # print("MVN.stddev :", mvn.stddev())             # (10, 3)
        # print("MVN.batch_shape :", mvn.batch_shape)
        # print("MVN.prob: ", sess.run(mvn.prob(x)))      # (10,)

        mvn_list.append(mvn)

    # print("\n", mvn_list)

    # GMM 
    gmm = tfd.Mixture(cat=tfd.Categorical(probs=z_pi),
                      components=mvn_list,
                      name="lossfunc_gmm")

    # print("\nGMM:", gmm)
    x = tf.concat([x1_data, x2_data, x3_data], axis=1)   # (10, 3)
    assert x.shape == (config.batch_size*config.seq_len, s)

    result = gmm.prob(x)        # 计算概率
    result = -tf.log(tf.maximum(result, 1e-20))   # 计算损失
    assert result.shape == (config.batch_size*config.seq_len, )

    # 损失加权. 按照seq由前向后权重变小
    loss_flat = tf.reshape(result, (config.batch_size, config.seq_len))
    loss_flag_weight = tf.multiply(loss_flat, config.loss_weight)     # 加权
    assert loss_flag_weight.shape == (config.batch_size, config.seq_len)
    loss_tf = tf.reduce_mean(loss_flag_weight)

    return gmm, loss_tf

