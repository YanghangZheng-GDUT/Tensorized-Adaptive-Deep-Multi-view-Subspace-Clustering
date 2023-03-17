import tensorflow as tf



class LinerNet_aesc(object):
    def __init__(self, v, sample_num, dims_encoder, para):
        
        self.v = v
        self.sample_num = sample_num
        self.dims_encoder = dims_encoder
        self.dims_decoder = [i for i in reversed(dims_encoder)]
        self.num_layers = len(self.dims_encoder)
        self.para_c_norm = para['c_norm']
        self.para_c_express = para['c_express']
        self.weights, self.netpara = self.init_weights()
        self.saver = tf.compat.v1.train.Saver(self.netpara)

    def init_weights(self):
        all_weights = dict()
        with tf.compat.v1.variable_scope('aesc-net'):
            for i in range(1, self.num_layers):
                all_weights['enc' + str(self.v) + '_w' + str(i)] = tf.compat.v1.get_variable(
                    name="enc" + str(self.v) + "_w" + str(i),
                    shape=[self.dims_encoder[i-1],
                           self.dims_encoder[i]],
                    initializer=tf.compat.v1.keras.initializers.he_normal())

                all_weights['enc' + str(self.v) + '_b' + str(i)] = tf.compat.v1.Variable(
                    tf.zeros([self.dims_encoder[i]], dtype=tf.float32))

            for i in range(1, self.num_layers):
                all_weights['dec' + str(self.v) + '_w' + str(i)] = tf.compat.v1.get_variable(
                    name="dec" + str(self.v) + "_w" + str(i),
                    shape=[self.dims_decoder[i-1],
                           self.dims_decoder[i]],
                    initializer=tf.compat.v1.keras.initializers.he_normal())

                all_weights['dec' + str(self.v) + '_b' + str(i)] = tf.compat.v1.Variable(
                    tf.zeros([self.dims_decoder[i]], dtype=tf.float32))

            all_weights['c'] = tf.compat.v1.Variable(1.0e-5 * tf.ones([self.sample_num, self.sample_num], tf.float32),
                                                       name='c' + str(self.v))
            aescnet = tf.compat.v1.trainable_variables()

        return all_weights, aescnet

    def encoder(self, x, weights):
        layer = x
        for i in range(1, self.num_layers):
            layer = tf.add(tf.matmul(layer, weights['enc' + str(self.v) + '_w' + str(i)]),
                           weights['enc' + str(self.v) + '_b' + str(i)])
            layer = tf.nn.sigmoid(layer)
        return layer

    def decoder(self, z_half, weights):
        layer = z_half
        for i in range(1, self.num_layers):
            layer = tf.add(tf.matmul(layer, weights['dec' + str(self.v) + '_w' + str(i)]),
                           weights['dec' + str(self.v) + '_b' + str(i)])
            layer = tf.nn.sigmoid(layer)
        return layer

    def loss_pretrain(self, x):
        z_half = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights)
        loss = tf.reduce_sum(tf.pow(tf.subtract(x, z), 2.0))
        return loss # , z_half

    def loss_aesc(self, x, g):
        z_half = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights)
        c = self.weights['c']
        z_half = tf.reshape(z_half, [self.sample_num, -1])
        self_z = tf.matmul(c, z_half)

        loss_ae = tf.reduce_sum(tf.pow(tf.subtract(z, x), 2.0))
        loss_c_norm = tf.reduce_sum(tf.pow(c, 2.0))
        loss_c_express = tf.reduce_sum(tf.pow(tf.subtract(self_z, z_half), 2.0))
        loss_c_g = tf.reduce_sum(tf.pow(tf.subtract(c, g), 2.0))
        return loss_ae, loss_c_norm, loss_c_express, loss_c_g

    def loss_aesc_nog(self, x):

        z_half = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights)
        c = self.weights['c']
        z_half = tf.reshape(z_half, [self.sample_num, -1])
        self_z = tf.matmul(c, z_half)

        loss_ae = tf.reduce_sum(tf.pow(tf.subtract(z, x), 2.0))
        loss_c_norm = tf.reduce_sum(tf.pow(c, 2.0))
        loss_c_express = tf.reduce_sum(tf.pow(tf.subtract(self_z, z_half), 2.0))
        return loss_ae, loss_c_norm, loss_c_express








