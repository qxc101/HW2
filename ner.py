import nltk
from nltk.corpus import conll2002, wordnet
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import openai
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import requests
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def build_gazetteer():
    entity_tag = "LOC"
    spanish_gazetteer = set()
    for sentence in list(conll2002.iob_sents('esp.train')):
        for token, pos, ne in sentence:
            if ne.startswith(f"B-{entity_tag}") or ne.startswith(f"I-{entity_tag}"):
                spanish_gazetteer.add(token.lower())

    return spanish_gazetteer


# spanish_gazetteer = build_gazetteer()


def is_in_gazetteer(word):
    normalized_word = word.lower()
    return normalized_word in spanish_gazetteer


def word_shape(word):
    shape = []

    for char in word:
        if char.isupper():
            shape.append("X")
        elif char.islower():
            shape.append("x")
        elif char.isdigit():
            shape.append("d")
        else:
            shape.append("-")

    return "".join(shape)


def word_identity(word):
    tokens = nltk.word_tokenize(word)
    pos_tag = nltk.pos_tag(tokens)[0][1]
    return pos_tag


def bigram(word):
    word = word.lower()
    normalized_word = word.translate(str.maketrans("áéíóúüñ", "aeiouun"))
    bigrams = [normalized_word[i:i + 2] for i in range(len(normalized_word) - 1)]
    return bigrams


def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [
        (o + 'word', word),
        # (o + 'word_len', len(word)),
        (o + 'word_shape', word_shape(word)),
        # (o + 'word_is_in_gazetteer', is_in_gazetteer(word)),
        # (o + 'word_identity', bigram(word)),
        # TODO: add more features here.
    ]
    return features
    

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    
    return dict(features)


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            # print("sent,i ", sent," ", i)
            feats = word2features(sent,i)
            # print("feats ", feats)
            # print("labels ", sent[i][-1])
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    print("finished feature")
    # model = MLPClassifier(n_jobs=-1, verbose=True)
    # model = LogisticRegression(n_jobs=-1, max_iter=500, verbose=True)
    # model = RandomForestClassifier(n_jobs=-1, verbose=True)
    # pipeline = Pipeline([
    #     ('clf', LinearSVC())
    # ])
    # param_grid = {
    #     'clf__C': [0.1, 1, 10],
    #     'clf__penalty': ['l1', 'l2'],
    #     'clf__loss': ['hinge', 'squared_hinge']
    # }
    # grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
    # grid_search.fit(X_train, train_labels)
    # best_params = grid_search.best_params_
    # print("Best parameters found by GridSearchCV:", best_params)
    model = LinearSVC(dual=True,penalty='l2', loss='squared_hinge', max_iter=5000, verbose=True)
    # model = SGDClassifier(verbose=True)
    model.fit(X_train, train_labels)
    print("finished training")
    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in test_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("test_results.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py  results.txt")






