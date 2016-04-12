import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN

# Parameters
# ==================================================

# Files and directories
tf.flags.DEFINE_string("pos_file", None, "File containing positive examples.")
tf.flags.DEFINE_string("neg_file", None, "File containing negative examples.")
tf.flags.DEFINE_string("word2vec_file", None, "File of saved word2vec model.")
tf.flags.DEFINE_string("vocab_file", None, "Path to vocabulary file.")
tf.flags.DEFINE_string("checkpoint_dir", None, "Directory for saving checkpoints.")

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_length", 200, "The length of a sequence of words (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: \"1,2,3\")")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 10)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if not FLAGS.pos_file or not FLAGS.neg_file:
    print("--pos_file and --neg_file must be specified.")
    sys.exit(1)

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x, y, word2id, id2word = data_helpers.load_data(FLAGS.vocab_file, FLAGS.pos_file, 
    FLAGS.neg_file, FLAGS.sequence_length, 110000)
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
x_train, x_dev = x_shuffled[:-10000], x_shuffled[-10000:]
y_train, y_dev = y_shuffled[:-10000], y_shuffled[-10000:]
print("Vocabulary Size: {:d}".format(len(word2id)))
print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=FLAGS.sequence_length,
            num_classes=2,
            vocab_size=len(word2id),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            word2vec_file=FLAGS.word2vec_file, 
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        if FLAGS.checkpoint_dir:
            checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "model")
            if not os.path.exists(FLAGS.checkpoint_dir):
                os.makedirs(FLAGS.checkpoint_dir)
        
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)

        def dev_step(x_batch, y_batch):
            """
            Evaluates model
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                dev_step(x_dev, y_dev)
            if current_step % FLAGS.checkpoint_every == 0 and FLAGS.checkpoint_dir:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}".format(path))
