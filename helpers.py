import nltk


def remove_stop_words(text: str):
    """
    Removes all the stop words in the text
    """
    pass


def get_top_N_words(text: str, n=10):
    """
    Get top n most frequently used words
    """
    tokens = nltk.word_tokenize(text)
    freq_dist = nltk.FreqDist(tokens)

    top_n_words_tup = freq_dist.most_common(n)
    top_n_words = [tup[0] for tup in top_n_words_tup]

    return top_n_words


def bag_of_words():
    pass
