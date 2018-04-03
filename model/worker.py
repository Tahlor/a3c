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

    def play_game(self, exchange=None, turns=GAME_LENGTH, starting_state=1000):
        if exchange is None:
            exchange = Exchange(DATA, cash=10000, holdings=0, actions=[-1, 1])
        starting_value = exchange.cash
        exchange.goto_state(starting_state)
        actions = []
        rewards = []

        # Prime e.g. LSTM
        historical_prices = exchange.get_price_history(current_id=starting_state, n=100,
                                                       freq=100)  # get 100 previous prices, every 100 steps
        # prime_lstm()

        for i in range(0, GAME_LENGTH):
            # get action prediction
            action = np.random.randn() - .5

            exchange.interpret_action(action)
            R = exchange.get_value() - starting_value

            # Record actions
            actions.append(action)
            rewards.append(R)
            starting_value = exchange.get_value()

        return actions, rewards

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            #  Initial state
            self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    #sess.run(self.copy_params_op)
                    self.loadNetworkFromSnapshot()

                    # Collect some experience
                    #transitions, local_t, global_t = self.play_game(t_max, sess)
                    actions, rewards = self.play_game(t_max, sess)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(actions, rewards, sess)

            except tf.errors.CancelledError:
                return

    def update(self, actions, rewards, sess):
        # Calculate reward
        r = self.global_model.sample_value()

        # Accumlate gradients at each time step
        for r in reverse(rewards):
            R = r + self.global_model.discount*R
            self.global_model.update_policy(R, rewards, actions)
            self.global_model.update_value(R, rewards)

