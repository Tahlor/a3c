import tensorflow as tf


class Policy:
    def __init__(self, model):
        self.model = model
        self.actions_op = self.model.actions_op
        self.inputs_ph = self.model.inputs_ph

    def sample_action(self, input):
        with tf.Session() as sess:
            actions_vector = sess.run(self.actions_op, feed_dict={self.inputs_ph: input})

        return actions_vector
