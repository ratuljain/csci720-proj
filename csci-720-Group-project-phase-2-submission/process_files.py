import statistics
import string
from os import listdir
from os.path import join, isfile
from pathlib import Path


import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader

from helpers import remove_stop_words, get_top_N_words

# switch to 'Gutenberg/text' for all the books.
DATA_DIR = './Gutenberg/test'
file_list = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
corpus = PlaintextCorpusReader(DATA_DIR, '.*.txt', encoding='utf8')
files = corpus.fileids()


# All the text cleaning logic goes here. Add the helper methods in helper.py
def clean_text(text: str):
    remove_stop_words(text)

    return text


# add all the features extraction features here. Add the helper methods in helper.py
def get_features(corpus: PlaintextCorpusReader, fname: str):
    feature_dict = {}

    author, title = Path(fname).stem.split('___')
    text = corpus.raw(fname)
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
    # additional features



    unique_word = set()
    for sent in nltk.sent_tokenize(text):
        token_text = nltk.word_tokenize(sent)
        tagged_words = nltk.pos_tag(token_text)
        # print(tagged)
        # if tagged not in unique_word:
        #     unique_word.add(tagged)


    feature_dict['avg_word_len'] = ''
    # feature_dict['stop_words_per_sentence'] =
    # feature_dict['stop_words_count'] = ''
    # feature_dict['special_char_count'] = ''
    # feature_dict['num_unique_words'] = ''
    # feature_dict['uppercase_word_count'] = ''

    # feature_dict['article_count_per_sentence'] = ''
    # feature_dict['article_count'] = ''
    # feature_dict['verb_count'] = ''
    # feature_dict['nouns_count'] = ''
    # feature_dict['pron_count'] = ''
    feature_dict['adj_count'] = ''
    feature_dict['adv_count'] = ''

    unique_word = set()
    stop_words_list = []
    stop_words_count = []
    stop_words_per_sentence = 0.0
    stop_words_count = 0.0

    comma_per_sentence_list = []
    comma_per_sentence_count = []
    comma_count = 0.0
    comma_per_sentence = 0.0

    special_char_per_sentence_list = []
    special_char_per_sentence_count = []
    special_char_count = 0.0
    special_char_per_sentence = 0.0

    feature_dict = {}
    all_sentences = nltk.sent_tokenize(text)
    all_words = []

    for sent in nltk.sent_tokenize(text):
        #     token_word = nltk.word_tokenize(sent)
        #     all_words.append(token_word)
        #     all_tagged_word.append(nltk.pos_tag(token_word))

        token_word = nltk.word_tokenize(sent)
        tagged_word = nltk.pos_tag(token_word)
        print(sent)
        print(token_word)
        #     print(tagged_word)

        #     stop_words_per_sentence = len([w for w in sent if w in stopwords.words('english')])
        for word in token_word:
            if word not in unique_word and word not in stopwords.words('english'):
                unique_word.add(word)
            if word in stopwords.words('english'):
                stop_words_list.append(word)
            if word in string.punctuation:
                if word == ',':
                    comma_per_sentence_list.append(word)
                else:
                    special_char_per_sentence_list.append(word)

        stop_words_count.append(len(stop_words_list))
        comma_per_sentence_count.append(len(comma_per_sentence_list))

    stop_words_per_sentence = statistics.mean(stop_words_count)
    stop_words_count = sum(stop_words_count)
    print(stop_words_per_sentence)
    print(stop_words_count)
    feature_dict['stop_words_per_sentence'] = stop_words_per_sentence
    feature_dict['stop_words_count'] = stop_words_count

    comma_per_sentence = statistics.mean(comma_per_sentence_count)
    comma_count = sum(comma_per_sentence_count)
    feature_dict['comma_per_sentence'] = comma_per_sentence
    feature_dict['comma_count'] = comma_count

    feature_dict['num_unique_words'] = len(unique_word)

    print(feature_dict)

    return feature_dict

feature_list = []
for fname in corpus.fileids():
    author, book_name = Path(fname).stem.split('___')
    text = corpus.raw(fname)
    cleaned_text = clean_text(text)
    get_features(corpus, fname)




