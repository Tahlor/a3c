import tensorflow as tf

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, state_size):
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):

        # combine inputs before passing in
        # combined_input = tf.concat([h_t_prev, input], axis=1)

        W_f = tf.get_variable(name='W_f',
                              shape=(inputs.shape[1], self._state_size),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        b_f = tf.get_variable(name='b_f',
                              shape=self._state_size,
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        W_i = tf.get_variable(name='W_i',
                              shape=(inputs.shape[1], self._state_size),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        b_i = tf.get_variable(name='b_i',
                              shape=self._state_size,
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        W_c = tf.get_variable(name='W_c',
                              shape=(inputs.shape[1], self._state_size),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        b_c = tf.get_variable(name='b_c',
                              shape=self._state_size,
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        W_o = tf.get_variable(name='W_o',
                              shape=(inputs.shape[1], self._state_size),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        b_o = tf.get_variable(name='b_o',
                              shape=self._state_size,
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        # example vector shapes
        # INPUT SHAPE = [1,10]
        # HIDDEN SHAPE = [1,128]
        #   COMBINED-INPUTS SHAPE = [1,138]
        # CELL-STATE SHAPE = [1,128]
        # [1,138] x [138,128] = [1,128]
        f_t = tf.sigmoid(tf.matmul(inputs, W_f) + b_f)
        i_t = tf.sigmoid(tf.matmul(inputs, W_i) + b_i)
        c_tilde_t = tf.tanh(tf.matmul(inputs, W_c) + b_c)

        c_t = (f_t * state) + (i_t * c_tilde_t)
        o_t = tf.sigmoid(tf.matmul(inputs, W_o) + b_o)
        h_t = o_t * tf.tanh(c_t)

        return h_t, c_t
