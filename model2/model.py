import tensorflow as tf
import tensorflow.contrib.layers as tfcl
import sys
sys.path.append("..")


def fc(inputs, num_nodes, name='0', activation=tf.nn.relu):
    weights = tf.get_variable('W_' + name,
                              shape=(inputs.shape[1], num_nodes),
                              dtype=tf.float32,
                              initializer=tfcl.variance_scaling_initializer())

    bias = tf.get_variable('b_' + name,
                           shape=[num_nodes],
                           dtype=tf.float32,
                           initializer=tfcl.variance_scaling_initializer())

    net_value = tf.matmul(inputs, weights) + bias
    return activation(net_value)


class Model:
    def __init__(self, input_size=10, layer_size=256):
        self.input_size = input_size
        self.layer_size = layer_size
        self.inputs_ph = None
        self.targets_ph = None # useless for 'real' model, just here for the proof of concept
        self.actions_op = None
        self.value_op = None
        self.loss_op = None
        self.optimizer = None
        self.saver = None
        self.graph = tf.Graph()

        self.build_network()

    def build_network(self):
        with self.graph.as_default():
            self.inputs_ph = tf.placeholder(tf.float32, shape=[1, self.input_size], name='inputs')
            self.targets_ph = tf.placeholder(tf.float32, shape=[1], name='targets')
            simple_network = fc(self.inputs_ph, self.layer_size)

            output = simple_network

            # Approach for a discrete action space, where we can either
            # buy or sell but don't specify an amount
            # logits = fc(output, 2, name='logits')
            # actions = tf.nn.softmax(logits)

            # Approach for a continuous space.
            # 'Action' is a real number in [-1,1], where
            # -1 means 'sell everything you have',
            # 0 means 'do nothing', and
            # 1 means 'buy everything you can'.
            # Exchange should know how to interpret this number.
            self.actions_op = fc(output, 1, name='action', activation=tf.nn.tanh)
            self.value_op = fc(output, 1, name='v')
            self.loss_op = tf.reduce_sum(self.targets_ph - self.actions_op, axis=1)
            self.optimizer = tf.train.RMSPropOptimizer(0.01).minimize(self.loss_op)

            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
