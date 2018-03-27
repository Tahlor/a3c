import tensorflow as tf
import numpy as np
from model2.lstm import LSTMCell

BATCH_SIZE = 1
SEQUENCE_LENGTH = 10
STATE_SIZE = 25

# build network
lstm = LSTMCell(STATE_SIZE)
src_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQUENCE_LENGTH], name='input')
trg_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQUENCE_LENGTH], name='target')

c_0 = tf.constant([1.], dtype=tf.float32, shape=[BATCH_SIZE, STATE_SIZE], name='c_0')
h_0 = tf.constant([1.], dtype=tf.float32, shape=[BATCH_SIZE, SEQUENCE_LENGTH], name='h_0')
h_1, c_1 = lstm(src_ph, h_0, c_0)

input = np.ones((BATCH_SIZE, SEQUENCE_LENGTH), dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )
# saver = tf.train.Saver()

sess.run(lstm, feed_dict={src_ph:input})
