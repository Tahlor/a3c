import tensorflow as tf
import numpy as np
from model2.model import Model

class Value:
    def __init__(self):
        self.model = Model()
        self.value_op = self.model.get_value_op()
        self.input_ph = self.model.get_input_ph()


    def sample_value(self, input):
        with tf.Session() as sess:
            value = sess.run(self.value_op, feed_dict={self.input_ph:input})

        return value

