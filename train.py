import sys
import os
import tensorflow as tf
import numpy as np
import shutil
import threading
import multiprocessing
from inspect import getsourcefile
from model2.model import Model
from model2.worker import Worker

current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from exchange import Exchange

tf.flags.DEFINE_string("model_dir", "/tmp/", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", "exchange_v1.0", "Name of game")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 300, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")
FLAGS = tf.flags.FLAGS


# Define Parameters

# Game parameters
CASH = 10000
BTC = 0
DATA = r"./data/BTC-USD_SHORT.npy"

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
summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))
saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=3)

# Initialize model (value and policy nets)
m = Model()

loss = tf.reduce_sum(m.targets_ph - m.actions_op, axis=1)
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

# Keep track of steps
global_step = tf.Variable(0, name="global_step", trainable=False)

# Create workers
workers = []
for worker_id in range(NUM_WORKERS):

    # Write summary for worker 0
    worker_summary_writer = None
    if worker_id == 0:
        worker_summary_writer = summary_writer

    # Initialize new workers
    worker = Worker()
    workers.append(worker)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists -- do this HERE OR IN MODEL?
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
      print("Loading model checkpoint: {}".format(latest_checkpoint))
      saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for worker in workers:
      worker_fn = lambda worker=worker: worker.run(sess, coord, FLAGS.t_max)
      t = threading.Thread(target=worker_fn)
      t.start()
      worker_threads.append(t)

    # Start a thread for policy eval task
    #monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    #monitor_thread.start()

    # Wait for all workers to finish
    coord.join(worker_threads)


