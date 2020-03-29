import glob, os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
import statistics
import string

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

DATA_DIR = r"Gutenberg/test1/"
DATABASE_USER = 'root'
DATABASE_USER_PASSWORD = 'password'
DATABASE_HOST = 'localhost'
DATABASE_NAME = 'Authorship_Attribution'
DATABASE_CONNECTION = "mysql+pymysql://" + DATABASE_USER + ":" + DATABASE_USER_PASSWORD + "@" + DATABASE_HOST + "/" + DATABASE_NAME


def pre_processing():
    author, file_path = get_author_and_file_path()
    authors, text = get_authors_and_text_list(author, file_path)
    return authors, text


def get_author_and_file_path():
    author, text = [], []
    DATA_DIR = r"Gutenberg/test1/"
    for file in glob.glob(f"{DATA_DIR}*.txt"):
        author.append(file.split("/")[-1].split("__")[0])
        text.append(file)
    return author, text


def get_author_and_text(author, file):
    authors, text = [], []
    with open(file, encoding='utf-8') as f:
        data = f.read()

    data = data.strip()
    data = data.split("\n\n")
    for para in data:
        authors.append(author)
        para = para.replace("\n", "")
        text.append(para)
    return authors, text


def get_authors_and_text_list(author, file_path):
    authors, text = [], []
    for auth, file in zip(author, file_path):
        name, sentence = get_author_and_text(auth, file)
        authors.extend(name)
        text.extend(sentence)
    return authors, text


def get_features(paragraph):
    feature_set = []
    unique_word = set()
    stop_words = 0
    commas = 0
    special_char = 0
    uppercase = 0
    articles = 0
    nouns = 0
    verbs = 0
    pronouns = 0

    token_word = nltk.word_tokenize(paragraph)
    tagged_word = nltk.pos_tag(token_word)

    para = (len(paragraph))
    sent = paragraph.count('.')
    sent_max_len = len(max(paragraph.split('.')))
    words = (len(token_word))

    for word, tag in tagged_word:
        if word not in unique_word and tag not in ('AT', 'DT') and \
                word not in stopwords.words('english') and word not in string.punctuation:
            unique_word.add(word)
        if word in stopwords.words('english'):
            stop_words += 1
        if word in string.punctuation:
            if word == ',':
                commas += 1
            else:
                special_char += 1
        if word.isupper():
            uppercase += 1
            # if tag in ('AT', 'DT'):
            #     articles += 1
            # if tag in ('NNP', 'NOUN'):
            #     nouns += 1
            # if tag in ('VBD', 'VERB'):
            #     verbs += 1
            # if tag in ('PRP', 'PRON'):
            #     pronouns += 1

    feature_set.append(para)
    feature_set.append(sent)
    feature_set.append(sent_max_len)
    feature_set.append(words)
    feature_set.append(len(unique_word))
    feature_set.append(stop_words)
    feature_set.append(commas)
    feature_set.append(special_char)
    feature_set.append(uppercase)
    #     feature_set.append(articles)
    #     feature_set.append(nouns)
    #     feature_set.append(verbs)
    #     feature_set.append(pronouns)
    return feature_set


def connect_to_database():
    engine = create_engine(DATABASE_CONNECTION)
    return engine


def disconnect_to_database():
    pass


def write_to_database(engine, df):
    df.to_sql(con=engine, name='author1', if_exists='replace', chunksize=5000)


def read_from_database(engine):
    df1 = pd.DataFrame()
    df1 = pd.read_sql('SELECT * FROM author1', con=engine)
    return df1


def train_naive_bayes_model(X_train, y_train):
    gnb = MultinomialNB()
    gnb.fit(X_train, y_train)
    return gnb


def naive_bayes(X_train, y_train, X_test, y_test):
    gnb = MultinomialNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    # accuracy = gnb.score(X_test, y_test)
    print(f"Accuracy: {100 * accuracy_score(y_test, y_pred):2.4f}%")
    conf = confusion_matrix(y_test, y_pred)
    print(conf)
    plt.imshow(conf, cmap='binary', interpolation='None')
    plt.show()


def kmeans_print_optimal_k(X):
    wass = []
    for i in range(1, 11):
        KM = KMeans(n_clusters=i, max_iter=500)
        KM.fit(X)
        wass.append(KM.inertia_)

    plt.plot(range(1, 11), wass, color='green', linewidth='3')
    plt.xlabel("K")
    plt.ylabel("Sqaured Error (wass)")
    plt.show()


def kmeans(X, y):
    # print no of optimal K
    kmeans_print_optimal_k(X)

    X11 = X.to_numpy()
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X11)
    y_kmeans = kmeans.predict(X11)
    # accuracy = gnb.score(X_test, y_test)
    plt.scatter(X11[:, 0], X11[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=50, alpha=0.5)


def kmeans_classification(X, y):
    X = X.to_numpy()
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)

    labelEncoder = LabelEncoder()
    labelEncoder.fit(y)
    y_actual_encoded = labelEncoder.transform(y)

    X_actual, y_actual = shuffle(X, y_actual_encoded, random_state=0)

    accuracy_count = 0
    for i in range(len(X_actual)):
        X_predict = X_actual.to_numpy()[i]
        X_predict = X_predict.reshape(-1, len(X_predict))
        y_predict = kmeans.predict(X_predict)
        if y_actual[i] == y_predict[0]:
            accuracy_count += 1

    print(f"Accuracy: {100 * accuracy_count / len(X):2.4f}%")

    # max scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans.fit(X_scaled)


    accuracy_count = 0
    for i in range(len(X_actual)):
        X_predict = X_actual.to_numpy()[i]
        X_predict = X_predict.reshape(-1, len(X_predict))
        y_predict = kmeans.predict(X_predict)
        if y_actual[i] == y_predict[0]:
            accuracy_count += 1

    print(f"Accuracy: {100 * accuracy_count / len(X):2.4f}%")


def authorship_attribution():
    authors, sentences = pre_processing()
    # Create pandas data frame
    df = pd.DataFrame()
    df['authors'], df['sentences'] = authors, sentences
    del authors
    del sentences

    df['features'] = df['sentences'].apply(lambda sentence: get_features(sentence))
    df[['para_len', 'sent', 'sent_max_len', 'word', 'unique_word', 'stop_words', 'comma', 'special', 'uppercase']] = \
        pd.DataFrame(df.features.values.tolist(), index=df.index)
    del df['features']

    connection = connect_to_database()
    write_to_database(connection, df)

    # Read from DB
    # df1 = pd.DataFrame()
    # df1 = read_from_database(connection)

    X, y = df.iloc[:, 3:], df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Naive Bayes
    naive_bayes(X_train, y_train, X_test, y_test)

    kmeans(X, y)
    kmeans_classification(X, y)


if __name__ == '__main__':
    authorship_attribution()
