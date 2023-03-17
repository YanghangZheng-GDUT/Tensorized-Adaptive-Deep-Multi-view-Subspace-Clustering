import tensorflow as tf

class Attention_fusion(object):

    def __init__(self, sample_num, v):
        self.sample_num = sample_num
        self.v = v
        with tf.compat.v1.variable_scope('attention_fusion'):
            self.weight = tf.compat.v1.Variable(tf.compat.v1.ones([v*sample_num, v]), name = 'attion_VE')
        self.afpara = tf.compat.v1.trainable_variables()

    def attention_fusion(self, G1, G2, weight):
        n_x = self.sample_num
        x1 = tf.concat([G1, G2], 1)
        p_nml = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(x1, weight)), axis=1)
        p = tf.math.l2_normalize(p_nml, axis=1)
        p1 = tf.reshape(p[:, 0], [n_x, 1])
        p2 = tf.reshape(p[:, 1], [n_x, 1])
        p1_broadcast = tf.tile(p1, [1, n_x])
        p2_broadcast = tf.tile(p2, [1, n_x])
        Coef_Fs = tf.multiply(p1_broadcast, G1) + tf.multiply(p2_broadcast, G2)  # hadamard product
        return Coef_Fs


    def attention_fusion_r(self, G, weight):
        n_x = self.sample_num
        x1 = tf.concat(G, 1)
        p_nml = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(x1, weight)), axis=1)
        p = tf.math.l2_normalize(p_nml, axis=1)

        for i in range(self.v):
            pi = tf.reshape(p[:, i], [n_x, 1])
            pi_broadcast = tf.tile(pi, [1, n_x])
            if i == 0:
                Coef_Fs = tf.multiply(pi_broadcast, G[i])  # hadamard product
            else:
                Coef_Fs += tf.multiply(pi_broadcast, G[i])  # hadamard product

        return Coef_Fs

    def loss_kl(self, G_inputs):
        Coef_F = self.attention_fusion_r(G_inputs, self.weight)

        sim_mat1 = 0.5 * (Coef_F + tf.transpose(Coef_F))
        sim_mat1 = sim_mat1 - tf.compat.v1.diag(tf.compat.v1.diag_part(sim_mat1))

        if tf.compat.v1.count_nonzero(tf.reduce_sum(sim_mat1, 1)) == self.sample_num:
            sim_mat2 = tf.divide(sim_mat1, tf.reduce_sum(sim_mat1, 1))
        else:
            sim_mat2 = sim_mat1
        sim_mat3 = (sim_mat2 - tf.compat.v1.diag(tf.compat.v1.diag_part(sim_mat2))) + tf.eye(self.sample_num)

        F_weight = sim_mat3 ** 2 / tf.reduce_sum(sim_mat3, 0)
        F_Aug = tf.transpose((tf.transpose(F_weight) / tf.reduce_sum(F_weight, 1)))

        KL_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        KL_div = KL_loss(F_Aug, Coef_F)
        return KL_div, Coef_F