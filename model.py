import logging
import os
import tensorflow as tf
import numpy as np
from utils.Linear_aesc import LinerNet_aesc
from utils.Conv_aesc import ConvNet_aesc
from utils.Attention_fusion import Attention_fusion
from utils.t_svd import t_SVD
from tqdm import tqdm
tf.compat.v1.disable_eager_execution()


def model(X, para, lr,
          L_dims=None, n_inputs=None, channels=None, kernels=None,
           data_name='hand'):

    view = len(X)
    sample_num = X[0].shape[0]
    Net_aescs = list()
    X_inputs = list()
    i = 0
    if L_dims is not None:
        for k in range(len(L_dims)):
            net_aesc = LinerNet_aesc(i+1, sample_num, L_dims[i], para)
            Net_aescs.append(net_aesc)
            x_input = tf.compat.v1.placeholder(np.float32, [None, L_dims[i][0]])
            X_inputs.append(x_input)
            i += 1
    if n_inputs is not None:
        for k in range(len(n_inputs)):
            net_aesc = ConvNet_aesc(i+1, sample_num, n_inputs[k], channels[k], kernels[k], para)
            Net_aescs.append(net_aesc)
            x_input = tf.compat.v1.placeholder(tf.float32, [None, n_inputs[k][0], n_inputs[k][1], 1])
            X_inputs.append(x_input)
            i += 1


    net_af = Attention_fusion(sample_num, view)

    C = np.zeros([sample_num, sample_num, view])
    G = np.zeros([sample_num, sample_num, view])
    G_inputs = list()
    C_inputs = list()
    for i in range(view):
        G_inputs.append(tf.compat.v1.placeholder(np.float32, [sample_num, sample_num]))
        C_inputs.append(tf.compat.v1.placeholder(np.float32, [sample_num, sample_num]))

    RHO = tf.compat.v1.placeholder(np.float32)

    loss_pre_list = list()

    aesc_C = list()
    loss_ae_list = list()
    loss_cnorm_list = list()
    loss_cexp_list = list()
    loss_c_g_list = list()
    for i in range(view):
        loss_p = Net_aescs[i].loss_pretrain(X_inputs[i])
        loss_pre_list.append(loss_p)
        if i == 0:
            loss_pre = loss_p
        else:
            loss_pre += loss_p

        aesc_C.append(Net_aescs[i].weights['c'])
        loss_aesc = list(Net_aescs[i].loss_aesc(X_inputs[i], G_inputs[i]))
        loss_ae_list.append(loss_aesc[0])
        loss_cnorm_list.append(loss_aesc[1])
        loss_cexp_list.append(loss_aesc[2])
        loss_c_g_list.append(loss_aesc[3])
        if i == 0:
            loss_aesc_all = loss_aesc[0] + para['c_norm']*loss_aesc[1] \
                            + para['c_express']*loss_aesc[2] + RHO/2*loss_aesc[3]
        else:
            loss_aesc_all += loss_aesc[0] + para['c_norm'] * loss_aesc[1] \
                             + para['c_express'] * loss_aesc[2] + RHO / 2 * loss_aesc[3]

    pre_opt = tf.compat.v1.train.AdamOptimizer(lr[0]).minimize(loss_pre)

    # af loss
    loss_kl, Cf = net_af.loss_kl(C_inputs)
    loss_kl = para['cf']*loss_kl
    loss_all = loss_aesc_all + loss_kl
    all_opt = tf.compat.v1.train.AdamOptimizer(lr[1]).minimize(loss_all, var_list=net_af.afpara)

    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())


    # init inner AEs
    base = os.path.dirname(os.path.realpath(__file__)) + '/pre_ae/'
    if not os.path.exists(base):
        os.makedirs(base)
    model_path = os.path.dirname(os.path.realpath(__file__)) + '/pre_ae/' + data_name
    saver.restore(sess, model_path+'/aescnet.ckpt')
    print('Restored Pretrain_ae model')

    # AESC-tSVD-AF
    rho = 0.0001
    max_rho = 10e12
    pho_rho = 1.5
    E = 1e-7
    print("#################       Train         ####################")
    for i in range(500):

        # aesc
        feed_dict = dict()
        for v in range(view):
            feed_dict[X_inputs[v]] = X[v]
            feed_dict[G_inputs[v]] = G[:, :, v]
            feed_dict[C_inputs[v]] = C[:, :, v]
        feed_dict[RHO] = rho
        _, loss_aesc_alld, \
        loss_aed, loss_cnormd, loss_cexpd, loss_c_gd, \
        C, cf, loss_kl_d = sess.run([all_opt, loss_aesc_all,
                           loss_ae_list, loss_cnorm_list, loss_cexp_list, loss_c_g_list,
                           aesc_C, Cf, loss_kl], feed_dict=feed_dict)

        for v in range(len(C)):
            C[v] = np.expand_dims(C[v], axis=2)
        C = np.concatenate(C, axis=2)

        # t-svd
        C_rot = C.transpose([0, 2, 1])
        G_rot = t_SVD(C_rot, 1 / rho)
        G = G_rot.transpose([0, 2, 1])
        rho = min(rho * pho_rho, max_rho)


        converge = 0
        for v in range(view):
            converge += np.max(np.abs(G[:, :, v] - C[:, :, v]))/view

        print("\nepoch:{} \nloss_ae:{} \nloss_c_norm:{} \nloss_c_express:{} \nloss_c_g:{} "
              "\nloss_kl:{} \nrho:{} \nconverge:{}".format(i + 1, loss_aed, loss_cnormd, loss_cexpd,
                                                           loss_c_gd,
                                                           loss_kl_d, rho, converge))

        if converge < E:
            logging.info("G converge: {:.8f}, end".format(converge))
            break

    tf.compat.v1.reset_default_graph()
    return cf
