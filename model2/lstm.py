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

    def __call__(self, input, h_t_prev, c_t_prev, scope=None):

        combined_input = tf.concat(h_t_prev, input)

        W_f = tf.get_variable(name='W_f',
                              shape=(self._state_size, combined_input.shape[0]),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        b_f = tf.get_variable(name='b_f',
                              shape=(self._state_size),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        W_i = tf.get_variable(name='W_i',
                              shape=(self._state_size, combined_input.shape[0]),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        b_i = tf.get_variable(name='b_i',
                              shape=(self._state_size),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        W_c = tf.get_variable(name='W_c',
                              shape=(self._state_size, combined_input.shape[0]),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        b_c = tf.get_variable(name='b_c',
                              shape=(self._state_size),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        W_o = tf.get_variable(name='W_o',
                              shape=(self._state_size, combined_input.shape[0]),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        b_o = tf.get_variable(name='b_o',
                              shape=(self._state_size),
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.variance_scaling_initializer())

        f_t = tf.sigmoid(tf.matmul(W_f, combined_input) + b_f)
        i_t = tf.sigmoid(tf.matmul(W_i, combined_input) + b_i)
        c_tilde_t = tf.tanh(tf.matmul(W_c, combined_input) + b_c)

        c_t = f_t * c_t_prev + i_t * c_tilde_t
        o_t = tf.sigmoid(tf.matmul(W_o, combined_input) + b_o)
        h_t = o_t * tf.tanh(c_t)

        return (c_t, h_t, h_t)