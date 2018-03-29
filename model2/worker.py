import tensorflow as tf
import numpy as np
from threading import Thread
from exchange import Exchange

# Normalize data somehow -- perhaps at the game level
# E.g. take all the steps needed to prime, all the steps post and normalize
# OR just train on percent changes between states -- maybe multiply by 1000 or something
GAME_LENGTH = 1000
CASH = 10000
BTC = 0
DATA = r"../data/BTC-USD_SHORT.npy"
DATA = r"../data/BTC_USD_100_FREQ.npy"


class Worker(Thread):
    def __init__(self, exchange, theta, theta_v, T, T_max, t_max=tf.Constant(10)):
        self.exchange = exchange
        self.t = tf.Variable(initial_value=1, trainable=False)
        self.T = T
        self.T_max = T_max
        self.t_max = t_max
        self.gamma = 0.9

        # TODO: create thread-specific copies of global parameters
        theta_prime = copyNetwork(theta)
        theta_prime_v = copyNetwork(theta_v)

    def run(self):
        while (self.T <= self.T_max):
            # TODO: reset gradients (maybe handled by TF?)

            t_start = tf.Variable(initial_value=self.t.value(), trainable=False)

            # TODO: get state s_t
            s_t = None

            ### THIS MAY BE ALL WE NEED IN THE WORKER CLASS ###
            while self.t - t_start < self.t_max and not self.exchange.is_terminal_state():
                # TODO: take action a_t according to policy; should return reward and new state
                self.t += 1
                self.T += 1

            R = 0 if self.exchange.is_terminal_state() else value(s_t, theta_prime_v)

            for i in range(self.t.value()-1, t_start.value(), -1):
                R = self.exchange.get_reward(s_t) + self.gamma * R
                # TODO: gradients for theta_prime
                # TODO: gradients for theta_prime_v

            #TODO: update theta and theta_v

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
