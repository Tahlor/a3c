import tensorflow as tf
import os
import fnmatch

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
