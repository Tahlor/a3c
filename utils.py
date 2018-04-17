import tensorflow as tf
import os, random
import fnmatch
import numpy as np

def log_scalar(tag, value, step):
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

def log(writer, tag, value, step):
    summary = log_scalar(tag, value, step)
    writer.add_summary(summary, step)

def createLogDir(basepath = "./tf_logs", name = "", force_numerical_ordering = True):
    n = 1

    # Add padding
    if name != "" and name[0] != " ":
        name = " " + name

    # Check for existence
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    if force_numerical_ordering:
        while find_files(basepath, str(n) + " *") or os.path.exists(os.path.join(basepath, str(n)    )) :
            n += 1
    else:
        while os.path.exists(os.path.join(basepath, str(n) + name )):
            n += 1

    # Create
    logdir = os.path.join(basepath, str(n) + name)
    os.makedirs(logdir)
    training_accuracy_list = []
    print(logdir)
    return logdir


def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    #matching_files = [n for n in fnmatch.filter(os.listdir(base), pattern) if os.path.isfile(os.path.join(base, n))]
    matching_files_and_folders = fnmatch.filter(os.listdir(base), pattern)
    return len(matching_files_and_folders)>0


class nextState():
    def __init__(self, state_range, game_length, hold_out_list = None, number_of_holdouts = 0, no_random = False):
        self.number_of_holdouts = number_of_holdouts
        self.state_range = state_range
        self.game_numbers = int((state_range[1] - state_range[0]) / game_length)-1 # subtract one just in case
        self.game_length = game_length
        self.hold_out_list = hold_out_list
        if hold_out_list is None:
            self.hold_out_list = np.random.choice(self.game_numbers, self.number_of_holdouts, replace=False)
        self.no_random = no_random
        self.reset()
        self.hold_out_game_number = 0

    def reset(self): # first valid state may not be 0, add it back in; multiply game index by game_length
        if self.no_random:
            randy = random.Random()
            offset = randy.randint(self.state_range[0], self.state_range[1] - self.game_length)
            self.game_list = np.asarray([offset + i for i in range(0, self.game_numbers)])
        else:
            self.game_list = self.state_range[0] + np.random.choice(self.game_numbers, self.game_numbers , replace=False) * self.game_length
        self.current_game_idx = 0

    def get_next(self):
        self.current_game_idx += 1

        # Don't do holdouts
        while self.current_game_idx in self.hold_out_list:
            self.current_game_idx += 1

        if self.current_game_idx >= len(self.game_list):
            self.reset()
        if self.no_random:
            return self.game_list[self.current_game_idx]
        else:
            return self.game_list[self.current_game_idx] + np.random.randint(0, self.game_length)

    def get_validation(self):
        self.hold_out_game_number = (self.hold_out_game_number + 1 ) % self.number_of_holdouts
        return self.hold_out_list[self.hold_out_game_number]

    def get_current_game(self):
        return self.game_list[self.current_game_idx]