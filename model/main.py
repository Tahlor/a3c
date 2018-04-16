import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from exchange import Exchange
from utils import *

# PARAMETERS
NEGATIVE_REWARD = False # naive
value_activation = None if NEGATIVE_REWARD else tf.nn.relu
#value_activation = None
PERMIT_SHORT = True
A_BOUND = [x*.9 for x in [-1, 1.0]]  # action bounds

ACTOR_NODES = 200 #200
CRITIC_NODES = 100 #100

OUTPUT_GRAPH = False # safe logs
RENDER = True  # render one worker
LOG_DIR = './log'  # savelocation for logs
N_WORKERS = multiprocessing.cpu_count()  # number of workers
MAX_EP_STEP = 100  # maxumum number of steps per episode
MAX_GLOBAL_EP = 10000  # total number of episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = MAX_EP_STEP/2  # sets how often the global net is updated (e.g. more often than 1 game)
GAMMA = 0.1  # discount factor
ENTROPY_BETA = 0.01  # entropy factor
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic

NUMBER_OF_NAIVE_INPUTS = 1
NAIVE_LOOKBACK = 8

NUMBER_OF_HOLDOUTS = 30

DATA = r"./data/BTC_USD_100_FREQ.npy"
main_exchange = Exchange(DATA, time_interval=1, game_length=MAX_EP_STEP, naive_price_history=NAIVE_LOOKBACK,naive_inputs=NUMBER_OF_NAIVE_INPUTS, permit_short=PERMIT_SHORT, naive= True)
state_manager = nextState(main_exchange.state_range, game_length=MAX_EP_STEP, hold_out_list = None, number_of_holdouts=NUMBER_OF_HOLDOUTS)
STARTING_STATE = state_manager.get_next()
print(STARTING_STATE)
main_exchange.reset(STARTING_STATE)

print(main_exchange.vanilla_prices[STARTING_STATE:STARTING_STATE+MAX_EP_STEP+1])
print(main_exchange.get_complete_state())
#print(main_exchange.get_model_input_naive(whiten=True))

N_S = (main_exchange.get_complete_state().shape)[0]  # number of states
N_A = 1  # number of actions

train_dir = createLogDir(basepath=LOG_DIR)
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
                self.a_params, self.c_params = self._build_net(scope)[-2:]  # parameters of actor and critic net
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')  # action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # v_target value

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(
                    scope)  # get mu and sigma of estimated action from neural net

                td = tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))


                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], tf.minimum(sigma + 1e-4, 2)

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

    def _build_net(self, scope):  # neural network structure of the actor and critic
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, ACTOR_NODES, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')  # estimated action value
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init,
                                    name='sigma')  # estimated variance
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, CRITIC_NODES, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, value_activation, kernel_initializer=w_init, name='v')  # estimated value for state
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        return self.sess.run([self.summary_dict, self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})[0]


# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, name, globalAC, sess):
        self.env = main_exchange.copy() # make environment for each worker
        self.name = name
        self.AC = ACNet(name, sess, globalAC)  # create ACNet for each worker
        self.sess = sess

    def work(self):
        global global_rewards, global_episodes, profits, state_manager
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not coord.should_stop() and global_episodes < MAX_GLOBAL_EP:
            s = self.env.reset(state_manager.get_next())
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                # Render one worker
                #if self.name == 'W_0' and RENDER:
                #    self.env.render()

                a = self.AC.choose_action(s)  # estimate stochastic action based on policy

                # Return State, Reward, _, _
                s_, r, done, info = self.env.step(a)  # make step in environment
                done = True if ep_t == MAX_EP_STEP - 1 else False

                if not NEGATIVE_REWARD and False:
                    r = max(r, 0)

                ep_r += r
                # save actions, states and rewards in buffer
                buffer_s.append(s)
                buffer_a.append(a)
                #buffer_r.append((r + 8) / 8)  # normalize reward
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    summary_dict, _, _ = self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()  # get global parameters to local ACNet

                s = s_
                total_step += 1

                if done:
                    profit = self.env.get_profit()
                    profits.append(profit)

                    if len(global_rewards) < 5:  # record running episode reward
                        global_rewards.append(ep_r)
                    else:
                        global_rewards.append(ep_r)
                        global_rewards[-1] = (np.mean(global_rewards[-5:]))  # smoothing

                    if global_episodes % 100 == 0:

                        print(
                            self.name,
                            "Ep:", global_episodes,
                            "| Ep_r: {:4.1f}, Profit: {:4.1f}, Policy Loss {:4.1f}, Value loss {:4.1f}, SD {:4.1f}, State {}".format(global_rewards[-1], profit, summary_dict["a_loss"], summary_dict["c_loss"], summary_dict["sd"][0,0], state_manager.get_current_game())
                        )
                        if False and global_episodes % 1000 == 0:
                            print ("sd", summary_dict["sd"][:,0])
                            print ("Mu", summary_dict["mu"][:,0])

                    log(SUMMARY_WRITER, "profit", profit, global_episodes)
                    log(SUMMARY_WRITER, "a_loss", summary_dict["a_loss"], global_episodes)
                    log(SUMMARY_WRITER, "c_loss", summary_dict["c_loss"], global_episodes)
                    log(SUMMARY_WRITER, "sd", summary_dict["sd"][0, 0], global_episodes)
                    global_episodes += 1
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

    env = main_exchange.copy()  # make environment for each worker
    AC = ACNet("validation", sess, globalAC)  # create ACNet for each worker

    total_profit = 0
    buy_and_hold = 0
    buy_and_hold_list = 0
    for holdout in range(0, NUMBER_OF_HOLDOUTS):
        start = state_manager.get_validation()
        end = start + MAX_EP_STEP
        s = env.reset(start)
        buy_and_hold += 10000 * (env.vanilla_prices[end]/env.vanilla_prices[start])
        
        for ep_t in range(MAX_EP_STEP):
            a = AC.choose_action(s)  # estimate stochastic action based on policy

            # Return State, Reward, _, _
            s_, r, done, info = env.step(a)
            total_profit += r
    print("Total profit (learned) (): {}".format())
    print("Total profit (): {}".format())

if __name__ == "__main__":
    global_rewards = []
    profits = []
    global_episodes = 0

    sess = tf.Session()

    with tf.device("/cpu:0"):
        global_ac = ACNet(GLOBAL_NET_SCOPE, sess)  # we only need its params
        workers = []
        # Create workers
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, global_ac, sess))

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

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
    graph()

