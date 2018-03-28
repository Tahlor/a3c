import tensorflow as tf
import numpy as np

LAYER_SIZE = 256


def fc(inputs, num_nodes, name='0'):
    weights = tf.get_variable('W_' + name,
                              shape=(inputs.shape[1], num_nodes),
                              dtype=tf.float32)
                              # intializer=tf.contrib.layers.variance_scaling_initializer())

    bias = tf.get_variable('b_' + name,
                           shape=[num_nodes],
                           dtype=tf.float32)
                           # initializer=tf.contrib.layers.variance_scaling_initializer())

    return tf.nn.relu(tf.matmul(inputs, weights) + bias)


inputs = tf.placeholder(tf.float32, shape=[1, 10], name='inputs')
targets = tf.placeholder(tf.float32, shape=[1, 2], name='targets')
simple_network = fc(inputs, LAYER_SIZE)

output = simple_network

logits = fc(output, 2, name='logits')
actions = tf.nn.softmax(logits)
value = fc(output, 1, name='v')

loss = tf.reduce_sum(targets - actions, axis=1)

optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):

        input_vector = np.random.rand(1, 10)
        target_vector = np.random.rand(1, 2)
        if target_vector[0][0] > 0.5:
            target_vector[0][0] = 1
            target_vector[0][1] = 0
        else:
            target_vector[0][0] = 0
            target_vector[0][1] = 1

        av, vv, lv = sess.run([actions, value, loss], feed_dict={inputs:input_vector, targets:target_vector})

        print("Action vector:" + str(av))
        print("Value approx.: " + str(vv))
        print("Loss: " + str(lv))
        print('')
