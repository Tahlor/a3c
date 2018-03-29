import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from exchange import Exchange

# Normalize data somehow -- perhaps at the game level
# E.g. take all the steps needed to prime, all the steps post and normalize
# OR just train on percent changes between states -- maybe multiply by 1000 or something

LAYER_SIZE = 256
INPUT_SIZE = 10
GAME_LENGTH = 1000
CASH = 10000
BTC = 0
DATA = r"../data/BTC-USD_SHORT.npy"
DATA = r"../data/BTC_USD_100_FREQ.npy"


def play_game(exchange = None, turns = GAME_LENGTH, starting_state = 1000):
    if exchange is None:
        exchange = Exchange(DATA, cash=10000, holdings=0, actions=[-1, 1])
    starting_value = exchange.cash
    exchange.goto_state(starting_state)
    actions = []
    rewards = []

    # Prime e.g. LSTM
    historical_prices = exchange.get_price_history(current_id=starting_state, n=100, freq=100) # get 100 previous prices, every 100 steps
    #prime_lstm()

    for i in range(0, GAME_LENGTH):
        # get action prediction
        action = np.random.randn()-.5

        exchange.interpret_action(action)
        R = exchange.get_value() - starting_value

        # Record actions
        actions.append(action)
        rewards.append(R)
        starting_value = exchange.get_value()

    return actions, rewards

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


inputs = tf.placeholder(tf.float32, shape=[1, INPUT_SIZE], name='inputs')
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
