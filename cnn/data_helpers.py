import numpy as np
import re
import itertools
from collections import Counter
from csv import reader
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from 
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(pos_file, neg_file):
    """
    Loads MR polarity data from files, splits the data into words and generates 
        labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(pos_file).readlines()[:15000])
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_file).readlines()[:15000])
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    #x_text = [s.split(" ") for s in x_text]
    tknzr = TweetTokenizer()
    x_text = [tknzr.tokenize(s) for s in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="UNK"):
    """
    Pads all sentences to the same length. The length is defined by the longest 
        sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(vocab_file, sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    stopset = set(stopwords.words('english'))

    # Load vocabulary file
    id2word = [] # Mapping from index to word
    word2id = {} # Mapping from word to index
    with open(vocab_file, "r") as file:
        for i, line in enumerate(reader(file, delimiter=" ")):
            #if len(line[0]) >= 3 and line[0] not in stopset:
                id2word.append(line[0])
                word2id[line[0]] = i
    # Add frequent words not in vocabulary
    #word_counts = Counter(itertools.chain(*sentences))
    #for i, x in enumerate(word_counts.most_common()):
        #if (x[1] > 100 and len(x[0]) >= 3 and x[0] not in word2id 
            #and x[0] not in stopset): 
            #id2word.append(x[0])
            #word2id[x[0]] = len(id2word) - 1

    return [word2id, id2word]


def build_input_data(sentences, labels, word2id):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[word2id[word] if word in word2id else 0 for word in sentence]
        for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(vocab_file, pos_file, neg_file):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """

    # Load and preprocess data
    sentences, labels = load_data_and_labels(pos_file, neg_file)
    sentences_padded = pad_sentences(sentences)

    word2id, id2word = build_vocab(vocab_file, sentences)
    x, y = build_input_data(sentences_padded, labels, word2id)

    return [x, y, word2id, id2word]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
