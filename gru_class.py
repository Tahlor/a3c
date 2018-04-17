from tensorflow.python.ops.rnn_cell import BasicLSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn_cell import RNNCell
import tensorflow as tf
import numpy as np
class mygru( RNNCell ):

     # input_size = vocab size for first cell
    def __init__( self, input_size, b_s, s_d = 128, scope_name = "cell", reuse = None):
        #super(BasicLSTMCell, self).__init__(_reuse=reuse)
        #super(mygru, self).__init__(_reuse=reuse)
    
        with tf.variable_scope(scope_name) as scope: 
            i = tf.random_uniform_initializer(-1e-3, 1e-3)
            self.s_d = s_d
            bs = b_s
            #s_d = 128 # state dimension  
            #b_s = 64  # batch size
            #v_s = 10  # vocab size
            
            # z = bs, sd
            # r = bs, sd
            # u = sd, sd
            # h = bs, sd
            # x = bs, input_size
            # output = bs, s_d
            
            self.Wz = tf.get_variable("Wz", [input_size, s_d], np.float32, initializer=i)   
            self.Wr = tf.get_variable("Wr", [input_size, s_d], np.float32, initializer=i)
            self.Wh = tf.get_variable("Wh", [input_size, s_d], np.float32, initializer=i)   
            self.Uz = tf.get_variable("Uz", [s_d, s_d], np.float32, initializer=i)   
            self.Ur = tf.get_variable("Ur", [s_d, s_d], np.float32, initializer=i)
            self.Uh = tf.get_variable("Uh", [s_d, s_d], np.float32, initializer=i)   
            self.bz = tf.get_variable("bz", [s_d], np.float32, initializer=i)
            self.br = tf.get_variable("br", [s_d], np.float32, initializer=i)   
            self.bh = tf.get_variable("bh", [s_d], np.float32, initializer=i)   
        
    @property
    def state_size(self):
    	#return (self.s_d, self.s_d)
        return self.s_d
    @property
    def output_size(self):
    	return (self.s_d)
 
    def __call__( self, inputs, state, scope=None ):
        x = inputs
        #h_prev = state[0]
        h_prev = state
        # bunch of matrix mult
        z = tf.sigmoid(tf.matmul(x, self.Wz) + tf.matmul(h_prev, self.Uz)  + self.bz)
        r = tf.sigmoid(tf.matmul(x, self.Wr) + tf.matmul(h_prev, self.Ur)  + self.br)
        h = z * h_prev + (1-z) * tf.tanh(tf.matmul(x, self.Wh) + tf.matmul(r * h_prev, self.Uh)  + self.bh)
        #return h, (h, h)        
        return h, h
    
# GRU - spits out [batch size, state dimension]
# x - [batch size, vocab size]
# output [bs, state_dim]
# fc => [bs, vocab size]
