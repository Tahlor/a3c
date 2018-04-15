import tensorflow as tf
import tensorflow.contrib.layers as tfcl
from tensorflow.contrib.rnn import GRUCell
import tensorflow.contrib.legacy_seq2seq as seq2seq
import math
import sys
import numpy as np
sys.path.append("..")
# import model.value
# import model.policy

MAIN_INITIALIZER = tfcl.variance_scaling_initializer()
#tfcl.variance_scaling_initializer()
#tf.ones_initializer()

ACTION_ACTIVATION = tf.nn.sigmoid

def fc(inputs, num_nodes, name='0', activation=tf.nn.relu):
    with tf.variable_scope('fully_connected', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('W_' + name,
                                  shape=(inputs.shape[1], num_nodes),
                                  dtype=tf.float32,
                                  initializer=tfcl.variance_scaling_initializer())

        bias = tf.get_variable('b_' + name,
                               shape=[num_nodes],
                               dtype=tf.float32,
                               initializer=tfcl.variance_scaling_initializer())

        net_value = tf.matmul(inputs, weights) + bias
        if activation is None:
            return net_value
        else:
            return activation(net_value)

def fc_list(inputs, num_nodes, name='0', activation=tf.nn.relu, scope = ""):
    with tf.name_scope(scope):
        outputs = []
        for item in inputs:
            outputs.append(fc(item, num_nodes, name=name, activation=activation))

        return tf.concat(outputs, axis=1)

# This just converts down multiple batch samples -- e.g. a 5 batch, 1000 sequence game = [5000 batches X layer input size]
def fc_list2(inputs, num_nodes, batch_size, name='1', activation=tf.nn.relu):
    # batch_size = self.batch_size * self.seq_length
    temp_inputs = tf.reshape(inputs, [batch_size, -1])
    output_list = tf.contrib.layers.fully_connected(temp_inputs, num_nodes, weights_initializer=MAIN_INITIALIZER, scope = name, activation_fn = activation)
    return output_list

def get_gru(num_layers, state_dim, reuse=False):
    with tf.variable_scope('gru', reuse=reuse):
        gru_cells = []
        for _ in range(num_layers):
            gru_cells.append(GRUCell(state_dim))

    return gru_cells

class Model:
    def __init__(self, batch_size=1, inputs_per_time_step=2, seq_length=1000, num_layers=1, layer_size=32, trainable = True, discount = .8, naive=False):
        self.seq_length = seq_length
        self.batch_size = batch_size
        if not naive:
            self.input_size = inputs_per_time_step * seq_length
        else:
            self.input_size = inputs_per_time_step
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.number_of_actions = 1
        self.inputs_ph = None
        self.targets_ph = None # useless for 'real' model, just here for the proof of concept
        self.actions_op = None
        self.value_op = None
        self.loss_op = None
        # learning_rate = 0.00025
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate = .00025, decay=0.99, momentum=0.0, epsilon=1e-6)
        self.saver = None
        self.trainable = trainable
        self.graph = tf.Graph()
        self.discount = discount
        self.entropy_weight = 1e-4
        self.naive = naive
        self.network_output = None
        self.build_network()


    def get_params(self):
        return {"input_size":self.input_size, "layer_size":self.layer_size, "trainable": self.trainable, "discount":self.discount}

    def build_input_layer(self, name = "input_fc"):
        temp_inputs = tf.reshape(self.inputs_ph, [self.batch_size * self.seq_length, -1])
        output_list = tf.contrib.layers.fully_connected(temp_inputs, self.layer_size,
                                                        weights_initializer=MAIN_INITIALIZER,
                                                        biases_initializer=tf.zeros_initializer(),
                                                        scope=name)  # this is just a 256 node network instead of GRU
        output_list = tf.reshape(output_list, [self.batch_size, self.seq_length, -1])
        return output_list

    def build_network(self):
        with self.graph.as_default():
            if self.naive:
                self.inputs_ph = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_length, self.input_size], name='inputs')
            else:
                self.inputs_ph = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name='inputs')
            self.targets_ph = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name='targets')
            self.gru_state_ph = tf.placeholder(tf.float32, shape=[self.batch_size, self.layer_size], name='gru_state')
            self.policy_advantage = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_length], name='advantages')
            self.chosen_actions = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_length, self.number_of_actions], name='chosen_actions')
            self.discounted_rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_length], name='discounted_rewards')

            inputs = tf.split(self.inputs_ph, self.seq_length, axis=1)

            if self.naive:
                #self.inputs_ph 1 X SEQ X (eg. 10 input prices/sides)
                #output_list = fc_list(self.inputs_ph, self.layer_size)
                #tf.initializers.xavier_initializer()

                ## If we want different networks, do it here
                if True:
                    with tf.name_scope("naive_shared_network") as scope:
                        self.output_list = self.build_input_layer()
                        self.output_list_value = self.output_list
                        self.output_list_policy = self.output_list
                else: # Try separate networks
                    self.output_list_value = self.build_input_layer()
                    self.output_list_policy = self.build_input_layer()


                #inputs, num_nodes, batch_size
                actions_raw = fc_list2(inputs = self.output_list_policy, num_nodes = self.number_of_actions * 2, batch_size = self.batch_size * self.seq_length, name='action_fc', activation=None)
                self.value_op1 = fc_list2(inputs=self.output_list_value, num_nodes=1,
                                       batch_size=self.batch_size * self.seq_length, name='value_fc',
                                       activation=None) # this is (SEQ LEN * BATCH) X 1
                self.value_op = tf.reshape(self.value_op1,[self.batch_size, self.seq_length]) # model expects SEQ * 1
                #print(self.value_op.shape)
            else:
                with tf.name_scope("gru_shared_network") as scope:
                    gru_cells = get_gru(self.num_layers, self.layer_size)
                    self.multi_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
                    # initial_state = self.multi_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                    initial_state = tuple([self.gru_state_ph for _ in range(self.num_layers)])

                    with tf.variable_scope('rnn_decoder') as scope:
                        # network_output is a tuple of (output_list, final_state)
                        # note that (output_list) is really just the GRU state at each time step
                        # (e.g. the final element in output_list is equal to final_state)
                        self.network_output = seq2seq.rnn_decoder(inputs, initial_state, self.multi_cell)
                        self.output_list = self.network_output[0]
                        final_state = self.network_output[1]


                # Actions distribution: [batch_size x seq_length x number_of_actions x 2]
                # i.e. one mu and one standard deviation for each action at each step of each sequence
                actions_raw = fc_list(self.output_list, self.number_of_actions * 2, name='action', activation=None, scope="actions_fc")

                # Value: [batch_size x seq_length]
                # i.e. one value per step in the sequence, for all sequences
                self.value_op = fc_list(self.output_list, 1, name='value', activation=None, scope="value_fc")

            self.actions_op = tf.reshape(actions_raw, [self.batch_size, self.seq_length, self.number_of_actions, 2])
            self.action_mu = tf.subtract(self.actions_op[:, :, :, 0], .5) # something from -.5 to .5
            self.action_sd = tf.nn.softplus(self.actions_op[:, :, :, 1]) + 1e-2 # this should not be 0

            self.saver = tf.train.Saver()

    def update_policy(self):
        with tf.name_scope("policy_updates") as scope:
            # input placeholder = input
            # GRU states placeholder = states

            # Actions [batch size, t steps, # of actions, 2 (action, sd)]
            # chosen actions = [ batch size=1 * t ]
            # chosen rewards = [ batch size=1 * t ]
            # value_op = [batch_size, t]
            # actions_mus    = [batch, t, # of actions]

            # Vector of continuous probabilites for each action
            # Vector of covariances for each action
            action_dist = tf.contrib.distributions.Normal(self.action_mu, self.action_sd) # [batch, t, # of actions]

            # Get log prob given chosen actions -- this range goes in both directions...
            #log_prob = action_dist.log_prob(self.chosen_actions) # probability < 1 , so negative value here, [batch, t, # of actions]
            log_prob = tf.log(tf.nn.sigmoid(action_dist.prob(self.chosen_actions)))

            # Calculate entropy
            # use absolute value of action_mu so it doesn't go negative and blow up the log,
            # then if action_mu was negative, flip the sign on the entropy value
            # entropy = -1/2 * (tf.log(2*abs(self.action_mu) * math.pi * self.action_sd ** 2 + 1e-2) + 1) # N steps X # of actions; add .0001 to prevent inf
            mess = 2 * self.action_mu * math.pi * self.action_sd ** 2

            # Entropy goes negative for small SD, small action_mu; especially small SD

            # for i in range(entropy.shape[0]):
            #     entropy = tf.cond(self.action_mu[0][i][0] < 0, lambda: -entropy[0][i][0], lambda: entropy[0][i][0])
            entropy = action_dist.entropy() # [batch, t, # of actions], negative

            # Advantage function - exogenous to the policy network
            # advantage = tf.subtract(self.rewards, self.value_op, name='advantage')  #[ batch size=1 * t ]

            # Loss -- entropy is higher with high uncertainty -- ensures exploration at first,
            #  e.g. even if an OK path is found at first, high entropy => higher loss, so it will take
            #   that good path with a grain of salt

            # policy_advantage is negative, log_prob is positive

            ## Dynamic advantage
            #self.advantage = self.value_op - self.discounted_rewards

            advantage = self.policy_advantage # self.advantage is dynamic, we feed in self.policy_advantage
            #self.assert_op = tf.Assert(tf.less_equal(tf.reduce_max(log_prob), 1.), [log_prob])
            self.assert_op = tf.Assert(True, [True])
            self.policy_loss = -tf.reduce_mean(log_prob * advantage + entropy * self.entropy_weight) # policy advantage [batch, t]
            self.policy_grads_and_vars = self.optimizer.compute_gradients(self.policy_loss)
            self.policy_grads_and_vars = [[grad, var] for grad, var in self.policy_grads_and_vars if grad is not None]
            self.policy_train_op = self.optimizer.apply_gradients(self.policy_grads_and_vars, global_step=tf.train.get_global_step())
            self.policy_train_op = tf.Variable([0])
            self.policy_loss_summary = tf.summary.scalar('policy_loss_summary', self.policy_loss)
            self.policy_dict = {"entropy":entropy, "log_prob": log_prob, "policy_loss":self.policy_loss, "mess":mess, "actions": self.action_mu, "output_list": self.output_list}

            return self.policy_train_op

    def update_value(self):
        with tf.name_scope("value_updates") as scope:
            #self.value_losses = (self.value_op - self.discounted_rewards)**2
            self.value_losses = tf.squared_difference(self.value_op, self.discounted_rewards)
            self.value_loss = tf.reduce_mean(self.value_losses, name="value_loss")
            self.value_grads_and_vars = self.optimizer.compute_gradients(self.value_loss)
            self.value_grads_and_vars = [[grad, var] for grad, var in self.value_grads_and_vars if grad is not None]
            self.value_train_op = self.optimizer.apply_gradients(self.value_grads_and_vars, global_step=tf.train.get_global_step())
            #self.value_train_op = tf.Variable([0])
            self.value_loss_summary = tf.summary.scalar('value_loss_summary', self.value_loss)
            #self.merged = tf.summary.merge([self.value_loss_summary, self.policy_loss_summary])
            return self.value_train_op


    # tf.contrib.distributions.Normal(1.,1.).log_prob()

    def get_actions_states_values(self, sess, input_tensor, gru_state):
        states = [[]]
        if self.naive:
            actions, values = sess.run([self.actions_op, self.value_op], feed_dict={self.inputs_ph: input_tensor})
        else:
            actions, states, values = sess.run([self.actions_op, self.network_output, self.value_op], feed_dict={self.inputs_ph: input_tensor, self.gru_state_ph: gru_state})
        return actions, tuple(states[0]), values

    def get_state(self):
        return self.last_input_state, self.gru_state_ph

    def get_value(self, sess, input, gru_state):
        with tf.Session() as sess:
            value = sess.run(self.value_op, feed_dict={self.inputs_ph: input, self.gru_state_ph: gru_state})
        return value, gru_state

    def get_policy(self, sess, input, gru_state):
        with tf.Session() as sess:
            policy = sess.run(self.policy_op, feed_dict={self.inputs_ph: input, self.gru_state_ph: gru_state})
        return policy, gru_state
