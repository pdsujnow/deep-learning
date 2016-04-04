from csv import reader
import tensorflow as tf
import sys

flags = tf.app.flags
flags.DEFINE_string("model_file", None, "Path to model file.")
flags.DEFINE_string("vocab_file", None, "Path to vocabulary file.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
FLAGS = flags.FLAGS

def main(_):
  """Load a word embedding."""
  if not FLAGS.model_file or not FLAGS.vocab_file:
    print("--model_file --vocab_file and must be specified.")
    sys.exit(1)

  # get the word to id mapping
  word2id = {}
  with open(FLAGS.vocab_file, "r") as file:
    for i, line in enumerate(reader(file, delimiter=" ")):
      word2id[line[0]] = i

  # load word embeddings
  with tf.Graph().as_default(), tf.Session() as session:
    #with tf.device("/cpu:0"):
    
    w_in = tf.Variable(tf.zeros([len(word2id), FLAGS.embedding_size]), 
      trainable=False, name="w_in")
    saver = tf.train.Saver({"w_in": w_in})
    saver.restore(session, FLAGS.model_file)

    tensor = tf.concat(0, [w_in.value(), tf.zeros([2, FLAGS.embedding_size])])
    embeddings = tf.Variable(tensor, trainable=True, name="embeddings")

    word_ids = tf.constant([[0, 1, 2], [3, 4, 71291]])
    word_emb = tf.nn.embedding_lookup(embeddings, word_ids)

    #word_emb = tf.Print(word_emb, [word_emb[0]])

    init = tf.initialize_variables([embeddings])
    session.run(init)

    word_emb = session.run(word_emb)
    print word_emb

if __name__ == "__main__":
  tf.app.run()
