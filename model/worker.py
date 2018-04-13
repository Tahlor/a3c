import tensorflow as tf
import numpy as np
from threading import Thread
from exchange import Exchange
from model.model import Model

# Normalize data somehow -- perhaps at the game level
# E.g. take all the steps needed to prime, all the steps post and normalize
# OR just train on percent changes between states -- maybe multiply by 1000 or something
GAME_LENGTH = 100
CASH = 10000
BTC = 0
DATA = r"./data/BTC-USD_SHORT.npy"
DATA = r"./data/BTC_USD_100_FREQ.npy"

# Each worker needs his own exchange -- needs to be some coordination to explore the exchange
# Train should have some logic to randomly move around the reinforcement space?
# Make some toy data

class Worker(Thread):
    def __init__(self, global_model, T, T_max, t_max=1000, states_to_prime = 1000, summary_writer=None):
        self.t = tf.Variable(initial_value=1, trainable=False)
        self.T = T
        self.T_max = T_max
        self.t_max = t_max
        self.global_model = global_model

        # Redundant
        self.deep_model = not global_model.naive
        self.naive = global_model.naive

        self.states_to_prime = states_to_prime
        #self.model = Model(**global_model.get_params()) # build model based on global model params

        # For now, just interface with main model
        self.model = self.global_model
        self.summary_writer = summary_writer

        # Each worker has an exchange; can be reset to any state
        self.exchange = Exchange(DATA, time_interval=60)


        # create thread-specific copy of global parameters
        # can load a single model because theta' and theta_v'
        # typically share a single network, and just munge the
        # output of that network differently to get action sample
        # and state value estimate
        # self.local_model = Model()
        # self.session = tf.Session(graph=self.local_model.graph)
        # self.loadNetworkFromSnapshot(model_file)

    def loadNetworkFromSnapshot(self, model_file):

        with tf.Session(graph=self.global_model.graph) as sess:
            self.global_model.saver.save(sess, model_file)

        self.local_model.saver.restore(self.session, model_file)

    def prime_gru(self, sess, input_tensor):
        initial_state = np.zeros([self.model.batch_size, self.model.layer_size])
        return sess.run(self.model.network_output, feed_dict={self.model.inputs_ph: input_tensor, self.model.gru_state_ph:initial_state})

    def play_game2(self, sess, turns=GAME_LENGTH, starting_state=1000):
        if self.exchange is None:
            self.exchange = Exchange(DATA, cash=10000, holdings=0, actions=[-1, 1])
        previous_value = self.exchange.cash
        self.exchange.goto_state(starting_state)
        chosen_actions = []
        rewards = []
        self.initial_gru_state = None
        # Prime e.g. LSTM
        if not self.deep_model:
            input_tensor = self.exchange.get_model_input_naive() # BATCH X SEQ X (Price, Side)
        else:
            # Prime GRU
            input_tensor = self.exchange.get_model_input(price_range=[starting_state - self.states_to_prime, starting_state], exogenous=True)
            self.initial_gru_state = self.prime_gru(sess, input_tensor)[1][0]
            input_tensor = self.exchange.get_model_input(price_range=[starting_state, starting_state + turns], exogenous=True)  # GAME LENGTH X INPUT SIZE

        # We could do the full GRU training in one shot if the input doesn't depend on our actions
        # When we calculate gradients, we can similarly do it in one batch
        self.input_tensor = input_tensor
        self.actions, self.state_sequence, self.values = self.model.get_actions_states_values(sess, input_tensor, self.initial_gru_state)  # returns GAME LENGTH X 1 X 2 [-1 to 1, sd]

        # final_state = [batch size, 256]

        for i in range(0, turns):
            # get action prediction
            action = self.actions[:, i, 0] # batch_size x 2
            mean = action[0][0]
            sd = action[0][1]
            if sd < 0:
                sd = -sd
            chosen_action = self.exchange.interpret_action(mean, sd)

            current_value = self.exchange.get_value()
            R = current_value - previous_value

            # Record actions
            chosen_actions.append(chosen_action)
            rewards.append(R)
            previous_value = self.exchange.get_value()

        self.chosen_actions = np.asarray(chosen_actions).reshape([self.model.batch_size, GAME_LENGTH, self.model.number_of_actions])
        self.rewards = np.asarray(rewards)

        # self.input_tensor, self.actions, self.states, self.values, self.chosen_actions, self.rewards


    def run(self, sess, coord):
        with sess.as_default(), sess.graph.as_default():
            #  Initial state
            # self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))

            try:
                while not coord.should_stop():
                    print("Running step #{}".format(str(self.T)))
                    # Copy Parameters from the global networks
                    #sess.run(self.copy_params_op)
                    # self.loadNetworkFromSnapshot()

                    # Collect some experience
                    #transitions, local_t, global_t = self.play_game(t_max, sess)

                    # Pre choose starting states without replacement:
                    # state_range = [int(x/exchange.game_length) for x in self.exchange.state_range]
                    # (np.random.choice(state_range[1]-state_range[0] , state_range[1]-state_range[0] , replace=False) + state_range[0] ) * exchange.game_length
                    print("Playing game for {} turns".format(self.t_max))
                    self.play_game2(sess, turns=self.t_max, starting_state=np.random.randint(*self.exchange.state_range))

                    if self.T_max is not None and next(self.T) >= self.T_max:
                        tf.logging.info("Reached global step {}. Stopping.".format(self.T))
                        print("Reached global step {}. Stopping.".format(self.T))
                        coord.request_stop()

                        return

                    # Update the global networks
                    print("Updating parameters")
                    self.update(sess)

            except tf.errors.CancelledError:
                return

    def update(self, sess):
        # Calculate reward
        # rewards, chosen_actions are: [batch, t]
        # states is batch x T x GRU SIZE
        # input_tensor is batch x T X INPUT_SIZE

        R = self.values[:,-1] # get value in last state

        # Accumulate gradients at each time step
        discounted_rewards = []
        policy_advantage = []

        for n, r in enumerate(self.rewards[::-1]):
            R = r + self.model.discount*R
            discounted_rewards.append(R)
            policy_advantage.append(R - self.values[:,n])

        self.discounted_rewards = np.asarray(discounted_rewards)[:, ::-1].T # batch, t , make it go forward again
        self.policy_advantage = np.asarray(policy_advantage).T
        self.update_policy(sess)
        self.update_values(sess)

    def update_policy(self, sess):
        policy_train_op = self.model.update_policy()
        policy_loss_summary = tf.summary.scalar('policy_loss', self.model.policy_loss)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        _, loss, policy_loss = sess.run([policy_train_op, policy_loss_summary, self.model.policy_loss_summary], feed_dict={self.model.inputs_ph: self.input_tensor, self.model.gru_state_ph: self.initial_gru_state,
                                                                  self.model.policy_advantage: self.policy_advantage, self.model.chosen_actions: self.chosen_actions})
        self.summary_writer.graph = self.model.graph
        self.summary_writer.add_summary(loss)
        print("Policy loss: {}".format(policy_loss))

    def update_values(self, sess):
        value_train_op = self.model.update_value()
        value_loss_summary = tf.summary.scalar('value_loss', self.model.value_loss)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        _, loss, value_loss = sess.run([value_train_op, value_loss_summary, self.model.policy_loss_summary], feed_dict={self.model.inputs_ph: self.input_tensor, self.model.gru_state_ph: self.initial_gru_state,
                                                                  self.model.discounted_rewards: self.discounted_rewards})
        self.summary_writer.graph = self.model.graph
        self.summary_writer.add_summary(loss)
        print("Values loss: {}".format(value_loss))
