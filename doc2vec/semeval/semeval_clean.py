# -*- coding: utf-8 -*-
from string import punctuation
import re
from nltk.tokenize import TweetTokenizer


def write_to_file(file_name, sents):
    with open(file_name, 'w') as f:
        for sent in sents:
            #print sent
            #f.write(sent.encode('utf-8'))
            try:
                f.write(sent)
            except UnicodeEncodeError:
                continue
            f.write('\n')


def clean_tweet(tweet):
    tknzr = TweetTokenizer()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet.lower())
    tweet = ' '.join(tweet.split())
    words = tknzr.tokenize(tweet)
    words = [''.join(c for c in s if c not in punctuation) for s in words]
    words = [s for s in words if s]
    sent = " ".join(words)
    return sent


def clean_data(input_file_name,
               pos_output_file_name,
               neg_output_file_name,
               neu_output_file_name,
               count_neutral=False):
    with open(input_file_name, 'r') as f:
        content = f.readlines()

    pos_sents = []
    neg_sents = []
    neu_sents = []
    for line in content:
        try:
            tweet_pair = line.split('\t')
            tweet = tweet_pair[1]
            sent = clean_tweet(tweet)
            # print sent
            pol = tweet_pair[0].replace('"', '')
            if pol == 'positive':
                pos_sents.append(sent)
            elif pol == 'negative':
                neg_sents.append(sent)
            elif pol == 'neutral':
                if count_neutral:
                    neu_sents.append(sent)
            else:
                raise RuntimeError('Unknown polarity %s' % pol)
        except UnicodeDecodeError:
            continue

    write_to_file(pos_output_file_name, pos_sents)
    write_to_file(neg_output_file_name, neg_sents)
    if count_neutral:
        write_to_file(neu_output_file_name, neu_sents)


if __name__ == '__main__':
    clean_data('../../semeval/train.tsv', 'train_pos.txt', 'train_neg.txt', 'train_neu.txt')
    clean_data('../../semeval/test.tsv', 'test_pos.txt', 'test_neg.txt', 'test_neu.txt')
