import sys
import os
import tensorflow as tf
import numpy as np
import shutil
import threading
import multiprocessing
from inspect import getsourcefile
from model.model import Model
from model.worker import Worker
import itertools
import archipack

current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from exchange import Exchange

# Define Parameters

# Game parameters
CASH = 10000
BTC = 0
DATA = r"./data/BTC-USD_SHORT.npy"
NAIVE_M0DEL = False
GAME_MAX_LENGTH = 1000
EPOCHS = 2
MODEL_DIR = "../tmp/"
TAYLOR = False

if os.environ["COMPUTERNAME"] == 'DALAILAMA':
    DATA = ".\data\BTC_USD_100_FREQ.npy"
    NAIVE_M0DEL = True
    GAME_MAX_LENGTH = 20
    EPOCHS = 100000
    MODEL_DIR = "./tmp"
    TAYLOR = True

tf.flags.DEFINE_string("model_dir", MODEL_DIR, "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", "exchange_v1.0", "Name of game")
tf.flags.DEFINE_integer("t_max", GAME_MAX_LENGTH, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", EPOCHS, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 300, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")
tf.flags.DEFINE_boolean("naive", NAIVE_M0DEL, "Use naive MLP.")
tf.flags.DEFINE_integer("naive_lookback", 10, "Number of back prices to look at.")
tf.flags.DEFINE_integer("num_input_types", 2, "E.g. prices, side, timestampe etc.")
tf.flags.DEFINE_string("data_path", DATA, "Path to .npy input file")
FLAGS = tf.flags.FLAGS

# Set the number of workers
NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = 1


MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Optionally empty model directory
if FLAGS.reset:
  shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

# Initialize saver
train_dir = os.path.join(MODEL_DIR, "train")
if TAYLOR:
    train_dir = archipack.createLogDir(basepath=train_dir)
summary_writer = tf.summary.FileWriter(train_dir)
if not os.path.exists(train_dir):
  os.makedirs(train_dir)

# saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=3)

# Initialize model (value and policy nets)
m = Model(seq_length=FLAGS.t_max, naive=FLAGS.naive, inputs_per_time_step=(FLAGS.naive_lookback * FLAGS.num_input_types if FLAGS.naive else FLAGS.num_input_types))

# Keep track of steps
global_step = tf.Variable(0, name="global_step", trainable=False)
T = itertools.count(1)

# Create workers
workers = []
for worker_id in range(NUM_WORKERS):

    # Write summary for worker 0
    worker_summary_writer = None
    if worker_id == 0:
        worker_summary_writer = summary_writer

    # Initialize new workers
    worker = Worker(global_model=m, T=T, T_max=FLAGS.max_global_steps, t_max=FLAGS.t_max, states_to_prime=FLAGS.t_max, summary_writer=worker_summary_writer, data = FLAGS.data_path)
    workers.append(worker)

# Have each worker somewhat randomly hop around to different dates
with tf.Session(graph=m.graph) as sess:
    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists -- do this HERE OR IN MODEL?
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
      print("Loading model checkpoint: {}".format(latest_checkpoint))
      m.saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for worker in workers:
      worker_fn = lambda worker=worker: worker.run(sess, coord)
      t = threading.Thread(target=worker_fn)
      t.start()
      worker_threads.append(t)

    # Start a thread for policy eval task
    #monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    #monitor_thread.start()

    # Wait for all workers to finish
    coord.join(worker_threads)
    summary_writer.close()

print('DONE')
