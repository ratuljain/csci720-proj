from os import listdir
from os.path import join, isfile
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import PlaintextCorpusReader

from helpers import remove_stop_words, get_top_N_words

# switch to 'Gutenberg/text' for all the books.
DATA_DIR = '/Users/ratuljain/PycharmProjects/CSCI720proj/Gutenberg/test'
file_list = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
corpus = PlaintextCorpusReader(DATA_DIR, '.*.txt')
files = corpus.fileids()


# All the text cleaning logic goes here. Add the helper methods in helper.py
def clean_text(text: str):
    remove_stop_words(text)

    return text


# add all the features extraction features here. Add the helper methods in helper.py
def get_features(corpus: PlaintextCorpusReader, fname: str):
    feature_dict = {}

    author, title = Path(fname).stem.split('___')
    words = corpus.words(fname)
    sentences = corpus.sents(fname)
    words_per_sentence = np.array([len(nltk.word_tokenize(' '.join(s))) for s in sentences])

    feature_dict['author'] = author
    feature_dict['title'] = title
    feature_dict['words'] = len(words)
    feature_dict['sentence'] = len(sentences)
    feature_dict['avg_words_per_sentence'] = words_per_sentence.mean()
    feature_dict['sentence_len_variation'] = words_per_sentence.std()
    feature_dict['top_words'] = get_top_N_words(text)

    print(feature_dict)

    return feature_dict


for fname in corpus.fileids():
    author, book_name = Path(fname).stem.split('___')
    text = corpus.raw(fname)
    cleaned_text = clean_text(text)
    get_features(corpus, fname)
