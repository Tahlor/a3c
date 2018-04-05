import tensorflow as tf
import numpy as np
from threading import Thread
from exchange import Exchange
from model.model import Model

# Normalize data somehow -- perhaps at the game level
# E.g. take all the steps needed to prime, all the steps post and normalize
# OR just train on percent changes between states -- maybe multiply by 1000 or something
GAME_LENGTH = 1000
CASH = 10000
BTC = 0
DATA = r"../data/BTC-USD_SHORT.npy"
DATA = r"../data/BTC_USD_100_FREQ.npy"

# Each worker needs his own exchange -- needs to be some coordination to explore the exchange
# Train should have some logic to randomly move around the reinforcement space?
# Make some toy data

class Worker(Thread):
    def __init__(self, exchange, global_model, T, T_max, t_max=10):
        self.exchange = exchange
        self.t = tf.Variable(initial_value=1, trainable=False)
        self.T = T
        self.T_max = T_max
        self.t_max = t_max
        self.global_model = global_model
        #self.model = Model(**global_model.get_params()) # build model based on global model params

        # For now, just interface with main model
        self.model = self.global_model

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



    # placeholder implementation of the pseudocode
    # def run(self):
    #     while (self.T <= self.T_max):
    #         # TODO: reset gradients (maybe handled by TF?)
    #
    #         t_start = tf.Variable(initial_value=self.t.value(), trainable=False)
    #
    #         # TODO: get state s_t
    #         s_t = None
    #
    #         ### THIS MAY BE ALL WE NEED IN THE WORKER CLASS ###
    #         while self.t - t_start < self.t_max and not self.exchange.is_terminal_state():
    #             # TODO: take action a_t according to policy; should return reward and new state
    #             self.t += 1
    #             self.T += 1
    #
    #         R = 0 if self.exchange.is_terminal_state() else value(s_t, theta_prime_v)
    #
    #         for i in range(self.t.value()-1, t_start.value(), -1):
    #             R = self.exchange.get_reward(s_t) + self.gamma * R
    #             # TODO: gradients for theta_prime
    #             # TODO: gradients for theta_prime_v
    #
    #         #TODO: update theta and theta_v

    def play_game(self, sess, turns=GAME_LENGTH, starting_state=1000):
        if self.exchange is None:
            self.exchange = Exchange(DATA, cash=10000, holdings=0, actions=[-1, 1])
        starting_value = self.exchange.cash
        self.exchange.goto_state(starting_state)
        actions = []
        rewards = []
        values = []
        states = []
        # Prime e.g. LSTM
        historical_prices = self.exchange.get_price_history(current_id=starting_state, n=self.global_model.input_size,
                                                       freq=100)  # get 100 previous prices, every 100 steps
        hp_reshaped = historical_prices.reshape([1,10])

        # prime_lstm()

        for i in range(0, GAME_LENGTH):
            # get action prediction
            # action = np.random.randn() - .5
            value, action = self.model.get_both(sess, hp_reshaped)

            self.exchange.interpret_action(action)
            current_value = self.exchange.get_value()
            R = current_value - starting_value

            # Record actions
            values.append(value)
            actions.append(action)
            rewards.append(R)
            starting_value = self.exchange.get_value()
            states.append(self.model.get_state()) # returns hidden/cell states, need to combine with input state
        return actions, rewards, values

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            #  Initial state
            # self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    #sess.run(self.copy_params_op)
                    # self.loadNetworkFromSnapshot()

                    # Collect some experience
                    #transitions, local_t, global_t = self.play_game(t_max, sess)
                    actions, rewards = self.play_game(sess, turns=t_max)

                    if self.T_max is not None and next(self.T) >= self.T_max:
                        tf.logging.info("Reached global step {}. Stopping.".format(self.T))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(sess, actions, rewards, values)

            except tf.errors.CancelledError:
                return

    def update(self, sess, actions, rewards, values):
        # Calculate reward
        r = self.global_model.sample_value()

        # Accumlate gradients at each time step
        for n, r in enumerate(rewards[::-1]):
            R = r + self.global_model.discount*R
            advantage = (R - values[::-1][n])