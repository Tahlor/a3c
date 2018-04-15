import tensorflow as tf
import numpy as np
from threading import Thread
from exchange import Exchange
from model.model import Model

## Problems:
# Final Value always less than initial value
# Is it learning?
# Are input prices correct?
# Policy loss is -inf


# Normalize data somehow -- perhaps at the game level
# E.g. take all the steps needed to prime, all the steps post and normalize
# OR just train on percent changes between states -- maybe multiply by 1000 or something
GAME_LENGTH = 100
CASH = 10000
BTC = 0
DATA = r"./data/BTC-USD_SHORT.npy"
DATA = r"./data/BTC_USD_100_FREQ.npy"

# Each worker needs his own exchange -- needs to be some coordination to explore the exchange
# Train should have some logic to randomly move around the reinforcement space?
# Make some toy data

class Worker(Thread):
    def __init__(self, global_model, T, T_max, t_max=1000, states_to_prime = 1000, summary_writer=None, data = DATA):
        self.previous = None # for testing if input is the same
        self.t = tf.Variable(initial_value=1, trainable=False)
        self.T = T
        self.T_max = T_max
        self.t_max = t_max
        self.global_model = global_model

        # Redundant
        self.deep_model = not global_model.naive
        self.naive = global_model.naive

        self.states_to_prime = states_to_prime
        #self.model = Model(**global_model.get_params()) # build model based on global model params

        # For now, just interface with main model
        self.model = self.global_model

        self.summary_writer = summary_writer

        # Each worker has an exchange; can be reset to any state
        self.exchange = Exchange(data, time_interval=1, game_length=self.t_max)


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

    def prime_gru(self, sess, input_tensor):
        initial_state = np.zeros([self.model.batch_size, self.model.layer_size])
        return sess.run(self.model.network_output, feed_dict={self.model.inputs_ph: input_tensor, self.model.gru_state_ph:initial_state})

    def play_game2(self, sess, starting_state=1000):
        self.exchange.reset()
        previous_value = self.exchange.get_value()
        self.exchange.goto_state(starting_state)
        chosen_actions = []
        rewards = []
        self.prices = []
        self.portfolio_values = []
        # Prime e.g. LSTM

        if self.naive:

            input_tensor = self.exchange.get_model_input_naive() # BATCH X SEQ X (Price, Side)
            self.initial_gru_state = np.zeros([self.model.batch_size, self.model.layer_size]) # just feed it some 0's

            # Make sure input is the same
            # if not self.previous is None:
            #     assert(self.previous.all() == input_tensor.all())
            #     print("Same input")
            # self.previous = input_tensor

        else:
            # Prime GRU
            input_tensor = self.exchange.get_model_input(price_range=[starting_state - self.states_to_prime, starting_state], exogenous=True)
            self.initial_gru_state = self.prime_gru(sess, input_tensor)[1][0]
            input_tensor = self.exchange.get_model_input(price_range=[starting_state, starting_state + self.t_max], exogenous=True)  # GAME LENGTH X INPUT SIZE

        self.input_tensor = input_tensor
        self.actions, self.state_sequence, self.values = self.model.get_actions_states_values(sess, input_tensor, self.initial_gru_state)  # returns GAME LENGTH X 1 X 2 [-1 to 1, sd]
        # final_state = [batch size, 256]

        #print(self.actions)
        for i in range(0, self.t_max):
            # get action prediction
            action = self.actions[:, i, 0] # batch_size x 2
            mean = action[0][0]
            sd = action[0][1]
            #print(mean, sd)
            chosen_action = self.exchange.interpret_action(mean, sd)
            #print(chosen_action)
            self.prices.append(self.exchange.current_price)
            self.exchange.get_next_state() # go to next state to find reward of that move
            current_value = self.exchange.get_value()
            self.portfolio_values.append(current_value)
            R = max(current_value - previous_value, 0)

            # Record actions
            chosen_actions.append(chosen_action)
            rewards.append(R)
            previous_value = current_value
            #self.exchange.get_status()

        self.chosen_actions = np.asarray(chosen_actions).reshape([self.model.batch_size, self.t_max, self.model.number_of_actions])
        self.rewards = np.asarray(rewards)

        if np.ndim(rewards) < 2: # if no batch dimension
            self.rewards = self.rewards[None,...]
        # self.input_tensor, self.actions, self.states, self.values, self.chosen_actions, self.rewards


    def run(self, sess, coord):
        with sess.as_default(), sess.graph.as_default():
            #  Initial state
            # self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))

            self.policy_train_op = self.model.update_policy()
            self.policy_loss_summary = tf.summary.scalar('policy_loss', self.model.policy_loss)

            self.value_train_op = self.model.update_value()
            self.value_loss_summary = tf.summary.scalar('value_loss', self.model.value_loss)

            self.summary_writer.graph = self.model.graph

            # Initialize model
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # Add graph
            # self.summary_writer.add_graph(self.summary_writer.graph)

            try:
                while not coord.should_stop():
                    count_string = str(self.T)
                    end_idx = count_string.find(',')
                    if end_idx == -1:
                        end_idx = count_string.find(')')
                    count_string = count_string[6:end_idx]

                    # Copy Parameters from the global networks
                    #sess.run(self.copy_params_op)
                    # self.loadNetworkFromSnapshot()

                    # Pre choose starting states without replacement:
                    # state_range = [int(x/exchange.game_length) for x in self.exchange.state_range]
                    # (np.random.choice(state_range[1]-state_range[0] , state_range[1]-state_range[0] , replace=False) + state_range[0] ) * exchange.game_length

                    #print("Playing game for {} turns".format(self.t_max))
                    starting_state = np.random.randint(*self.exchange.state_range)
                    self.play_game2(sess, starting_state=10)

                    # Update the global ne  tworks
                    #print("Updating parameters")
                    self.global_step = int(count_string)
                    self.update(sess)

                    # Write out profits
                    portfolio_value = self.exchange.get_value()-self.exchange.starting_cash
                    self.summary_writer.add_summary (self.log_scalar("portfolio", portfolio_value, self.global_step), self.global_step)


                    if int(count_string) % 100 == 0:
                        print("Finished step #{}, net worth {}, value loss {}, policy loss {}".format(int(count_string), self.exchange.get_value(), self.value_loss, self.policy_loss))
                        #print("Actions {}".format(self.chosen_actions))
                        #print("Action Mus {}".format(self.policy_loss_dict["actions"]))
                        #print("Network out {}".format(self.policy_loss_dict["output_list"][0,0:10]))

                    if self.T_max is not None and next(self.T) >= self.T_max:
                        tf.logging.info("Reached global step {}. Stopping.".format(self.T))
                        print("Reached global step {}. Stopping.".format(self.T))
                        coord.request_stop()

                        return

            except tf.errors.CancelledError:
                return

    def update(self, sess):
        # Calculate reward
        # rewards, chosen_actions are: [batch, t]
        # states is batch x T x GRU SIZE
        # input_tensor is batch x T X INPUT_SIZE

        # Accumulate gradients at each time step
        discounted_rewards = []
        policy_advantage = []

        rewards_swapped = np.transpose(self.rewards[::-1], [1,0]) # swap batch and seq axes, so SEX X BATCH; also reverse;
        values_swapped = np.transpose(self.values[::-1], [1, 0])

        # Copmletely ignore the stupid value net
        values_swapped = np.zeros((values_swapped.shape))
        R = values_swapped[0] # get value in last state
        R = 0
        for n, r in enumerate(rewards_swapped):
            R = r + self.model.discount*R
            discounted_rewards.append(R)
            policy_advantage.append(R - values_swapped[n])

        # Unreverse and transpose
        self.discounted_rewards = np.asarray(discounted_rewards[::-1]).transpose([1,0])
        self.policy_advantage = np.asarray(policy_advantage[::-1]).transpose([1,0])

        if self.global_step % 4000==0 and False:
            print("NEW RUN")
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            pass

        if False and self.global_step % 100 == 0:
            print(self.portfolio_values)
            print(self.rewards)
            print(self.discounted_rewards)
            import time
            time.sleep(2)
            #Stop
        self.update_policy(sess)

        if False:
            self.update_values(sess)
        else:
            self.value_loss = "N/A"

    def update_policy(self, sess):
        # self.policy_dict, self.model.policy_loss
        _, loss, self.policy_loss_dict, _ = sess.run([self.policy_train_op, self.policy_loss_summary, self.model.policy_dict, self.model.assert_op], feed_dict={self.model.inputs_ph: self.input_tensor, self.model.gru_state_ph: self.initial_gru_state,
            self.model.policy_advantage: self.policy_advantage, self.model.chosen_actions: self.chosen_actions})
        #_, loss = sess.run([self.policy_train_op, self.policy_loss_summary], feed_dict={self.model.inputs_ph: self.input_tensor, self.model.gru_state_ph: self.initial_gru_state,
            #self.model.policy_advantage: self.policy_advantage, self.model.chosen_actions: self.chosen_actions})
        self.summary_writer.add_summary(loss, self.global_step)

        #print(np.concatenate((pl["entropy"], pl["mess"]), axis=2))
        #print(pl["log_prob"])
        #print(pl["policy_loss"])

        self.policy_loss = self.policy_loss_dict ["policy_loss"]
        #print("Policy loss: {}".format(policy_loss))

    def update_values(self, sess):
        _, loss, self.value_loss = sess.run([self.value_train_op, self.value_loss_summary, self.model.value_loss],
                                       feed_dict={self.model.inputs_ph: self.input_tensor, self.model.gru_state_ph: self.initial_gru_state,self.model.discounted_rewards: self.discounted_rewards})
        #_, loss = sess.run([self.value_train_op, self.value_loss_summary],
                                        # feed_dict={self.model.inputs_ph: self.input_tensor, self.model.gru_state_ph: self.initial_gru_state,self.model.discounted_rewards: self.discounted_rewards})

        self.summary_writer.add_summary(loss, self.global_step)
        #print("Values loss: {}".format(self.value_loss))


    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        return summary