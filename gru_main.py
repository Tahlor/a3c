#import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell
import tensorflow.contrib.legacy_seq2seq as seq2seq
from exchange import Exchange
from utils import *
import argparse
import re
from gru_class import  mygru

# PARAMETERS
USE_ONE_GRU = True
RESTORE_PATH = ""
MY_GRU = True

parser = argparse.ArgumentParser()
parser.add_argument('--init', type=str, default=RESTORE_PATH,
                    help='initialize from old model')

parser.add_argument('--validate_only', type=str2bool, default=False,
                    help="don't train, just validate")

parser.add_argument('--new_folder', type=str, default="",
                    help="specify special folder")

args = parser.parse_args()

SAVE_FREQ = 1000 # for model checkpoints
PRINT_FREQ = 50

NEGATIVE_REWARD = False # naive
value_activation = None if NEGATIVE_REWARD else tf.nn.relu
#value_activation = None
PERMIT_SHORT = True
A_BOUND = [x*.9 for x in [-1, 1.0]]  # action bounds

ACTOR_NODES = 200 #200
CRITIC_NODES = 200 #100

OUTPUT_GRAPH = False # safe logs
RENDER = True  # render one worker
LOG_DIR = './log'  # savelocation for logs
N_WORKERS = int(multiprocessing.cpu_count())

if args.validate_only:
    N_WORKERS = 1

MAX_EP_STEP = 10000  # maxumum number of steps per episode
MAX_GLOBAL_EP = 1000000  # total number of episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 100  # sets how often the global net is updated (e.g. more often than 1 game)
GAMMA = 0.1  # discount factor
ENTROPY_BETA = 0.01  # entropy factor
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
N_LAYERS = 1 # number of layers in GRU implementation
USE_NAIVE = False

NUMBER_OF_NAIVE_INPUTS = 1
NAIVE_LOOKBACK = 10
NUMBER_OF_HOLDOUTS = 20

DATA = r"./data/BTC_USD_10_FREQ.npy"
main_exchange = Exchange(DATA, time_interval=1, game_length=MAX_EP_STEP, naive_price_history=NAIVE_LOOKBACK,naive_inputs=NUMBER_OF_NAIVE_INPUTS, permit_short=PERMIT_SHORT, naive=False)
global_state_manager = nextState(main_exchange.state_range, game_length=MAX_EP_STEP, hold_out_list=None,
                                  number_of_holdouts=NUMBER_OF_HOLDOUTS, no_random=True)
# print(main_exchange.vanilla_prices[STARTING_STATE:STARTING_STATE+MAX_EP_STEP+1])
#print(main_exchange.get_complete_state())
#print(main_exchange.get_model_input_naive(whiten=True))

N_S = (main_exchange.get_complete_state().shape)[0]  # number of states
N_A = 1  # number of actions

EXPLICIT_CPU = True
RESTORE_PATH = args.init

# PARAMETERS
SAVE_FOLDER = "./checkpoints"
if RESTORE_PATH!="":
    SAVE_FOLDER = RESTORE_PATH
else:
    SAVE_FOLDER = createLogDir(basepath=SAVE_FOLDER)
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


## Make a whole new folder
if args.new_folder != "":
    train_dir = args.new_folder
    SAVE_FOLDER = args.new_folder

    # Make folder if it does not exist
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    else:
        RESTORE_PATH = SAVE_FOLDER
else:
    train_dir = createLogDir(basepath=LOG_DIR)

checkpoint_path = os.path.join(SAVE_FOLDER, 'model.ckpt')
SUMMARY_WRITER = tf.summary.FileWriter(train_dir)



# Network for the Actor Critic
class ACNet(object):
    def __init__(self, scope, sess, globalAC=None):
        self.sess = sess
        self.actor_optimizer = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')  # optimizer for the actor
        self.critic_optimizer = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')  # optimizer for the critic

        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_state, self.c_state, self.a_params, self.c_params = self._build_net(scope)[-4:]  # parameters of actor and critic net
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')  # action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # v_target value

                mu, sigma, self.v, self.a_state, self.c_state, self.a_params, self.c_params = self._build_net(
                    scope)  # get mu and sigma of estimated action from neural net

                td = tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))


                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], tf.minimum(sigma + 1e-4, 1.5)

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):

                    if NEGATIVE_REWARD:
                        neg_rwd_mask = (tf.sign(self.v_target) - 1) / 2  # 0 for pos, -1 for neg
                        prob = tf.minimum(normal_dist.prob(self.a_his), .99)  # outputs (0,1]
                        log_prob = tf.log(abs(prob + neg_rwd_mask))  # invert probabilities with negative rewards (e.g. 99% becomes 1%); outputs (-inf, 0)
                    else:
                        prob = tf.minimum(normal_dist.prob(self.a_his), .99)
                        #log_prob = tf.log(prob)
                        log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0],
                                              A_BOUND[1])  # sample a action from distribution
                    self.A_mu = tf.squeeze(mu, axis=0)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss,
                                                self.a_params)  # calculate gradients for the network weights
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):  # update local and global network weights
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

            self.summary_dict = {"a_loss":self.a_loss, "c_loss": self.c_loss, "mu":mu, "sd":sigma}

    def get_gru(self, input, state_ph, num_layers, state_dim, reuse=False, activation=tf.nn.relu6, kernel_initializer=tf.random_normal_initializer, name='0'):
        with tf.variable_scope('gru_' + name, reuse=reuse):
            gru_cells = []
            for _ in range(num_layers):
                #GRUCell, mygru
                if MY_GRU:
                    gru_cells.append(mygru(5, 1, state_dim))
                else:
                    gru_cells.append(GRUCell(state_dim, activation=activation, kernel_initializer=kernel_initializer))


            gru = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
            output, final_state = seq2seq.rnn_decoder(input, tuple(state_ph for _ in range(1)), gru)

        return final_state[0]

    def _build_net(self, scope):  # neural network structure of the actor and critic
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            if USE_NAIVE:
                l_a = tf.layers.dense(self.s, ACTOR_NODES, tf.nn.relu6, kernel_initializer=w_init, name='la')
            else:
                self.a_gru_state_ph = tf.placeholder(tf.float32, shape=[None, ACTOR_NODES], name='a_gru_state')
                l_a = self.get_gru([self.s], self.a_gru_state_ph, N_LAYERS, ACTOR_NODES, activation=tf.nn.relu6, kernel_initializer=w_init, name='la')

            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')  # estimated action value
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init,
                                    name='sigma')  # estimated variance
        with tf.variable_scope('critic'):
            if USE_NAIVE:
                l_c = tf.layers.dense(self.s, CRITIC_NODES, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            else:
                if USE_ONE_GRU:
                    self.c_gru_state_ph = self.a_gru_state_ph
                    l_c = l_a
                else:
                    self.c_gru_state_ph = tf.placeholder(tf.float32, shape=[None, CRITIC_NODES], name='c_gru_state')
                    l_c = self.get_gru([self.s], self.c_gru_state_ph, N_LAYERS, CRITIC_NODES, activation=tf.nn.relu6,
                                       kernel_initializer=w_init, name='lc')

            v = tf.layers.dense(l_c, 1, value_activation, kernel_initializer=w_init, name='v')  # estimated value for state
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, l_a, l_c, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        return self.sess.run([self.summary_dict, self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, a_state, c_state = None, sample = True):  # run by a local
        s = s[np.newaxis, :]
        a = self.A if sample else self.A_mu
        if c_state is None:
            feed_dict = {self.s: s, self.a_gru_state_ph: a_state}
            a_return, a_state_return = self.sess.run([a, self.a_state], feed_dict)
            c_state_return = a_state_return
        else:
            feed_dict = {self.s: s, self.a_gru_state_ph: a_state, self.c_gru_state_ph: c_state}
            a_return, a_state_return, c_state_return = self.sess.run([a, self.a_state, self.c_state],feed_dict)
        return a_return[0], a_state_return, c_state_return



# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, name, globalAC, sess):
        self.env = main_exchange.copy() # make environment for each worker
        self.name = name
        self.AC = ACNet(name, sess, globalAC)  # create ACNet for each worker
        self.sess = sess
        self.state_manager = nextState(main_exchange.state_range, game_length=MAX_EP_STEP, hold_out_list=None,
                                  number_of_holdouts=NUMBER_OF_HOLDOUTS, no_random=True)
        STARTING_STATE = self.state_manager.get_next()
        print(STARTING_STATE)
        self.env.reset(STARTING_STATE)

    def work(self):
        global global_rewards, global_episodes, profits, profits_above_baseline
        total_step = 1
        a_state = np.zeros([1, ACTOR_NODES])
        c_state = np.zeros([1, CRITIC_NODES])
        if USE_ONE_GRU:
            c_state = None
        buffer_s, buffer_a, buffer_r, buffer_a_state, buffer_c_state = [], [], [], [], []
        while not coord.should_stop() and global_episodes < MAX_GLOBAL_EP:
            start = self.state_manager.get_next()
            end = start + MAX_EP_STEP
            s_ = self.env.reset(start)
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                # Render one worker
                #if self.name == 'W_0' and RENDER:
                #    self.env.render()

                a, a_state, c_state = self.AC.choose_action(s, a_state, c_state)  # estimate stochastic action based on policy

                # Return State, Reward, _, _
                s, r, done, info = self.env.step(a)  # make step in environment
                done = True if ep_t == MAX_EP_STEP - 1 else False

                if not NEGATIVE_REWARD and False:
                    r = max(r, 0)

                ep_r += r
                # save actions, states and rewards in buffer
                buffer_s.append(s)
                buffer_a.append(a)
                #buffer_r.append((r + 8) / 8)  # normalize reward
                buffer_r.append(r)
                buffer_a_state.append(np.copy(a_state))
                buffer_c_state.append(np.copy(c_state))

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.c_gru_state_ph: c_state})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target, buffer_a_state, buffer_c_state = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target), np.vstack(buffer_a_state), np.vstack(buffer_c_state)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.a_gru_state_ph: buffer_a_state,
                        self.AC.c_gru_state_ph: buffer_c_state
                    }
                    summary_dict, _, _ = self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r, buffer_a_state, buffer_c_state = [], [], [], [], []
                    self.AC.pull_global()  # get global parameters to local ACNet

                s = s_
                total_step += 1

                if done:
                    buy_and_hold = 10000 * self.env.vanilla_prices[end]/self.env.vanilla_prices[start] - 10000
                    profit = self.env.get_profit()
                    profits.append(profit)
                    profits_above_baseline.append(profit-buy_and_hold)

                    if len(global_rewards) < 5:  # record running episode reward
                        global_rewards.append(ep_r)
                    else:
                        global_rewards.append(ep_r)
                        global_rewards[-1] = (np.mean(global_rewards[-5:]))  # smoothing

                    if global_episodes % PRINT_FREQ == 0:

                        print(
                            self.name,
                            "Ep:", global_episodes,
                            "| Ep_r: {:4.1f}, Profit: {:4.1f}, Policy Loss {:4.1f}, Value loss {:4.1f}, Mu {:4.1f}, SD {:4.1f}, State {}, Above Baseline {:4.0f}".format(global_rewards[-1], profit, summary_dict["a_loss"], summary_dict["c_loss"], summary_dict["mu"][0,0], summary_dict["sd"][0,0], self.state_manager.get_current_game(), profits_above_baseline[-1])
                        )
                        if False and global_episodes % 1000 == 0:
                            print ("sd", summary_dict["sd"][:,0])
                            print ("Mu", summary_dict["mu"][:,0])

                    if global_episodes % 20 == 0:
                        log(SUMMARY_WRITER, "profit", profit, global_episodes)
                        log(SUMMARY_WRITER, "a_loss", summary_dict["a_loss"], global_episodes)
                        log(SUMMARY_WRITER, "c_loss", summary_dict["c_loss"], global_episodes)
                        log(SUMMARY_WRITER, "sd", summary_dict["sd"][0, 0], global_episodes)
                        log(SUMMARY_WRITER, "profit_over_baseline", profits_above_baseline[-1], global_episodes)

                    global_episodes += 1
                    # Save model every ~10k; put this after global episode counter to avoid collisions
                    if global_episodes % SAVE_FREQ == 1 and global_episodes > 1:
                        saver.save(sess, checkpoint_path, global_step=global_episodes)

                    break
def graph():
    plt.plot(np.arange(len(global_rewards)), global_rewards)  # plot rewards
    plt.xlabel('step')
    plt.ylabel('total moving reward')
    plt.show()

    plt.plot(np.arange(len(profits)), profits)  # plot profits
    plt.xlabel('step')
    plt.ylabel('total profits')
    plt.show()

def run_validation(sess, globalAC):
    print("Running validation...")
    env = main_exchange.copy()  # make environment for each worker
    AC = ACNet("validation", sess, globalAC)  # create ACNet for each worker
    sess.run(tf.global_variables_initializer())
    buy_and_hold_list = []
    bot_profit_list = []
    #a_state = sess.run([AC.a_state])
    #c_state = sess.run([AC.c_state])
    a_state = np.zeros([1, ACTOR_NODES])
    c_state = np.zeros([1, CRITIC_NODES])
    if USE_ONE_GRU:
        c_state = None
    for holdout in range(0, NUMBER_OF_HOLDOUTS):
        if holdout % 10 == 0:
            print("Running game {}...".format(holdout))
        start = global_state_manager.get_validation()
        end = start + MAX_EP_STEP
        s = env.reset(start-1000) # prime it for 1,000
        buy_and_hold_gain = 10000 * (env.vanilla_prices[end]/env.vanilla_prices[start]) - 10000
        buy_and_hold_list.append(buy_and_hold_gain)

        for ep_t in range(2000):
            a,a_state,c_state = AC.choose_action(s, a_state=a_state, c_state=c_state, sample = False)  # estimate stochastic action based on policy

            # Return State, Reward, _, _
            s, r, done, info = env.step([a])
            if env.state == start:
                env.cash = 10000 # reset
        bot_profit_list.append(env.get_profit())

    print("Total profit (learned): {}".format(sum(bot_profit_list)))
    print("Total profit (buy and hold): {}".format(sum(buy_and_hold_list)))
    print(buy_and_hold_list)
    print(bot_profit_list)

if __name__ == "__main__":
    global_rewards = []
    profits = []
    global_episodes = 0
    profits_above_baseline = []
    sess = tf.Session()

    if EXPLICIT_CPU:
        with tf.device("/cpu:0"):
            global_ac = ACNet(GLOBAL_NET_SCOPE, sess)  # we only need its params
            workers = []
            # Create workers
            for i in range(N_WORKERS):
                i_name = 'W_%i' % i  # worker name
                workers.append(Worker(i_name, global_ac, sess))
    else:
        global_ac = ACNet(GLOBAL_NET_SCOPE, sess)  # we only need its params
        workers = []
        # Create workers
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, global_ac, sess))

        coord = tf.train.Coordinator()

    if not args.validate_only:
        sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    if RESTORE_PATH != "":
        ckpt = tf.train.get_checkpoint_state(RESTORE_PATH )
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_episodes = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        except:
            print("Could not restore")
            import traceback
            traceback.print_exc()
            STOP


    if not args.validate_only: # don't do full training
        if OUTPUT_GRAPH:  # write log file
            SUMMARY_WRITER.add_graph(sess.graph)

        worker_threads = []
        for worker in workers:  # start workers
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
            #workers[1].get_status()
        coord.join(worker_threads)  # wait for termination of workers
    else:
        state_manager = nextState(main_exchange.state_range, game_length=1000, hold_out_list=[150000*x for x in range(1,21)],
                                  number_of_holdouts=NUMBER_OF_HOLDOUTS)
    run_validation(sess, global_ac)
    # graph()
