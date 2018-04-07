import tensorflow as tf
import tensorflow.contrib.layers as tfcl
from tensorflow.contrib.rnn import GRUCell
import tensorflow.contrib.legacy_seq2seq as seq2seq
import math
import sys
sys.path.append("..")
# import model.value
# import model.policy

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

def get_gru(num_layers, state_dim, reuse=False):
    with tf.variable_scope('gru', reuse=reuse):
        gru_cells = []
        for _ in range(num_layers):
            gru_cells.append(GRUCell(state_dim))

    return gru_cells


class Model:
    def __init__(self, input_size=10, num_layers=1, layer_size=256, trainable = True, discount = .9, naive=False):
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.inputs_ph = None
        self.targets_ph = None # useless for 'real' model, just here for the proof of concept
        self.actions_op = None
        self.value_op = None
        self.loss_op = None
        self.optimizer = None
        self.saver = None
        self.trainable = trainable
        self.graph = tf.Graph()
        self.discount = discount
        self.entropy_weight = 1e-4
        self.naive = naive
        self.build_network()


    def get_params(self):
        return {"input_size":self.input_size, "layer_size":self.layer_size, "trainable": self.trainable, "discount":self.discount}

    def build_network(self):
        with self.graph.as_default():
            self.inputs_ph = tf.placeholder(tf.float32, shape=[1, self.input_size], name='inputs')
            self.targets_ph = tf.placeholder(tf.float32, shape=[1], name='targets')

            inputs = tf.split(self.inputs_ph, self.input_size, axis=1)

            if self.naive:
                outputs = fc(self.inputs_ph, self.layer_size)
            else:
                gru_cells = get_gru(self.num_layers, self.layer_size)
                multi_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
                initial_state = multi_cell.zero_state(batch_size=1, dtype=tf.float32)

                with tf.variable_scope('rnn_decoder') as scope:
                    output_list, final_state = seq2seq.rnn_decoder(inputs, initial_state, multi_cell)
                    outputs = tf.concat(output_list, axis=0)

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
            self.actions_op = fc(outputs, 1, name='action', activation=tf.nn.tanh)
            self.value_op = fc(outputs, 1, name='v')
            self.loss_op = tf.reduce_sum(self.targets_ph - self.actions_op, axis=1)
            self.optimizer = tf.train.RMSPropOptimizer(0.01).minimize(self.loss_op)

            #with tf.Session(graph=self.graph) as sess:
            #    sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()

    def update_policy(self, chosen_actions, inputs):
        # Action -- needs to output
        #log(1 + exp(x))

        # Actions [n steps, # of actions, 2 (action, sd)]

        # Vector of continuous probabilites for each action
        # Vector of covariances for each action
        actions =
        sds = self.actions_op[:,:,1] # Actions op returns N X 1 action X 2
        action_vectors = actions[:,:,0] # n steps, by 1 action

        # Calculate entropy
        entropy = -1/2 * (tf.log(2*action_vectors * math.pi * sds ** 2) + 1) # N steps X # of actions

        # Advantages just an N list
        # Action Vectors N X # of actions
        # Entropy N X # of actions
        self.policy_losses = - (tf.log(action_vectors) * advantages + self.entropy_weight * entropy)
        self.policy_loss = tf.reduce_sum(self.policy_losses, name="policy_loss")

        #self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.policy_grads_and_vars = self.optimizer.compute_gradients(self.policy_loss)
        self.policy_grads_and_vars = [[grad, var] for grad, var in self.policy_grads_and_vars if grad is not None]
        self.policy_train_op = self.optimizer.apply_gradients(self.policy_grads_and_vars, global_step=tf.contrib.framework.get_global_step())

    def update_value(self, advantages):
        self.value_losses = (advantages)**2
        self.value_loss = tf.reduce_sum(self.value_losses, name="value_loss")

        #self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.value_grads_and_vars = self.optimizer.compute_gradients(self.value_loss)
        self.value_grads_and_vars = [[grad, var] for grad, var in self.value_grads_and_vars if grad is not None]
        self.value_train_op = self.optimizer.apply_gradients(self.value_grads_and_vars, global_step=tf.contrib.framework.get_global_step())

    tf.contrib.distributions.Normal(1,1).log_prob()

    def get_state(self):
        return self.last_input_state, self.gru_state

    def get_value(self, sess, input, gru_state = None):
        #session.run(self.global_model.actions_op, feed_dict={self.global_model.inputs_ph: hp_reshaped})
        with tf.Session() as sess:
            value = sess.run(self.value_op, feed_dict={self.input_ph: input, self.gru_state_input: gru_state})
        return value, gru_state

    def get_policy(self, sess, input, gru_state = None):
        with tf.Session() as sess:
            policy = sess.run(self.policy_op, feed_dict={self.input_ph: input, self.gru_state_input: gru_state})
        return policy, gru_state

