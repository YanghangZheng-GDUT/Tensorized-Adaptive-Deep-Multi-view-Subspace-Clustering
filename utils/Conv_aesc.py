import tensorflow as tf



class ConvNet_aesc(object):
    def __init__(self, v, sample_num, n_input, channel_enc, kernel_enc, para):
        
        self.v = v
        self.sample_num = sample_num
        self.n_input = n_input
        self.channel_enc = channel_enc
        self.channel_dec = [i for i in reversed(channel_enc)]
        self.kernel_enc = kernel_enc
        self.kernel_dec = [i for i in reversed(kernel_enc)]
        self.num_layers = len(self.channel_enc)
        self.para_c_norm = para['c_norm']
        self.para_c_express = para['c_express']
        self.weights, self.netpara = self.init_weights()
        self.saver = tf.compat.v1.train.Saver(self.netpara)

    def init_weights(self):
        all_weights = dict()
        with tf.compat.v1.variable_scope('aesc-net'):
            for i in range(1, self.num_layers):
                all_weights['enc' + str(self.v) + '_w' + str(i)] = \
                    tf.compat.v1.get_variable(name="enc" + str(self.v) + "_w" + str(i),
                                              shape=[self.kernel_enc[i-1], self.kernel_enc[i-1],
                                                     self.channel_enc[i-1], self.channel_enc[i]],
                                              initializer=tf.compat.v1.keras.initializers.he_normal())

                all_weights['enc' + str(self.v) + '_b' + str(i)] = tf.compat.v1.Variable(
                    tf.zeros([self.channel_enc[i]], dtype=tf.float32))

            for i in range(1, self.num_layers):
                all_weights['dec' + str(self.v) + '_w' + str(i)] = \
                    tf.compat.v1.get_variable(name="dec" + str(self.v) + "_w" + str(i),
                                              shape=[self.kernel_dec[i-1], self.kernel_dec[i-1],
                                                     self.channel_dec[i], self.channel_dec[i-1]],
                                              initializer=tf.compat.v1.keras.initializers.he_normal())

                all_weights['dec' + str(self.v) + '_b' + str(i)] = tf.compat.v1.Variable(
                    tf.zeros([self.channel_dec[i]], dtype=tf.float32))

            all_weights['c'] = tf.compat.v1.Variable(1.0e-5 * tf.ones([self.sample_num, self.sample_num], tf.float32),
                                                       name='c' + str(self.v))
            aescnet = tf.compat.v1.trainable_variables()

        return all_weights, aescnet

    def encoder(self, x, weights):
        layer = x
        shapes = []
        for i in range(1, self.num_layers):
            shapes.append(layer.get_shape().as_list())
            layer = tf.nn.bias_add(tf.nn.conv2d(layer, weights['enc' + str(self.v) + '_w' + str(i)],strides=[1, 2, 2, 1],
                                                        padding='SAME'), weights['enc' + str(self.v) + '_b' + str(i)])
            layer = tf.nn.relu(layer)
        return layer, shapes

    def decoder(self, z_half, weights, shapes):
        layer = z_half
        shapes = shapes[::-1]
        for i in range(1, self.num_layers):
            layer = tf.add(tf.nn.conv2d_transpose(layer, weights['dec' + str(self.v) + '_w' + str(i)], tf.stack(
                [tf.shape(z_half)[0], shapes[i-1][1], shapes[i-1][2], shapes[i-1][3]]),
                                    strides=[1, 2, 2, 1], padding='SAME'), weights['dec' + str(self.v) + '_b' + str(i)])
            layer = tf.nn.relu(layer)
        return layer

    def loss_pretrain(self, x):
        z_half, shapes = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights, shapes)
        loss = tf.reduce_sum(tf.pow(tf.subtract(x, z), 2.0))
        return loss

    def loss_aesc(self, x, g):
        z_half, shapes = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights, shapes)
        c = self.weights['c']
        z_half = tf.reshape(z_half, [self.sample_num, -1])
        self_z = tf.matmul(c, z_half)

        loss_ae = tf.reduce_sum(tf.pow(tf.subtract(z, x), 2.0))
        loss_c_norm = tf.reduce_sum(tf.pow(c, 2.0))
        loss_c_express = tf.reduce_sum(tf.pow(tf.subtract(self_z, z_half), 2.0))
        loss_c_g = tf.reduce_sum(tf.pow(tf.subtract(c, g), 2.0))
        return loss_ae, loss_c_norm, loss_c_express, loss_c_g

    def loss_aesc_nog(self, x):

        z_half, shapes = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights, shapes)
        c = self.weights['c']
        z_half = tf.reshape(z_half, [self.sample_num, -1])
        self_z = tf.matmul(c, z_half)

        loss_ae = tf.reduce_sum(tf.pow(tf.subtract(z, x), 2.0))
        loss_c_norm = tf.reduce_sum(tf.pow(c, 2.0))
        loss_c_express = tf.reduce_sum(tf.pow(tf.subtract(self_z, z_half), 2.0))
        return loss_ae, loss_c_norm, loss_c_express








