import tensorflow as tf
import numpy as np
from model2.model import Model


class Policy:
    def __init__(self):
        self.model = Model()
        self.actions_op = self.model.get_actions_op()
        self.input_ph = self.model.get_input_ph()



    def sample_action(self, input):
        with tf.Session() as sess:
            actions_vector = sess.run(self.actions_op, feed_dict={self.input_ph:input})
            selected_action = np.argmax(actions_vector, axis=1)

        return 'buy' if selected_action == 0 else 'sell'