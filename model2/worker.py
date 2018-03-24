import tensorflow as tf
from threading import Thread

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
