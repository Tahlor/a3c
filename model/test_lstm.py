import tensorflow as tf
import numpy as np
from model.lstm import LSTMCell

BATCH_SIZE = 1
SEQUENCE_LENGTH = 10
STATE_SIZE = 25

# build network
lstm = LSTMCell(STATE_SIZE)
src_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQUENCE_LENGTH+STATE_SIZE], name='comb_inputs')
# trg_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQUENCE_LENGTH], name='target')

c_0 = tf.constant([1.], dtype=tf.float32, shape=[BATCH_SIZE, STATE_SIZE], name='c_0')
# h_0 = tf.constant([1.], dtype=tf.float32, shape=[BATCH_SIZE, STATE_SIZE], name='h_0')
states = lstm(src_ph, c_0)

input = np.ones((BATCH_SIZE, SEQUENCE_LENGTH), dtype=np.float32)
h_0 = np.ones(shape=[BATCH_SIZE, STATE_SIZE], dtype=np.float32)
comb_input = np.concatenate((h_0, input), axis=1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )
# saver = tf.train.Saver()

h,c = sess.run(states, feed_dict={src_ph:comb_input})
print("h: " + str(h))
print("c: " + str(c))
