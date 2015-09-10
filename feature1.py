#!/usr/bin/python

""" feature1.py
    -----------
    @author = Ankai Lou
"""

import os
import sys
import string
import nltk
from tfidf import tfidf
from collections import Counter

def generate_dataset(documents, lexicon):
    """ function: generate_dataset
        --------------------------
        select features from @lexicon for feature vectors
        generate dataset of feature vectors for @documents

        :param documents: list of well-formatted, processable documents
        :param lexicon:   list of word stems for selecting features
    """
    print('Generating dataset @ dataset1.csv...')

    # UNCOMMENT WHEN DEBUGGING
    # print(documents)
    # print(lexicon)

    # document = { 'topics' : [], 'places' : [],
    #              'words' : { 'title' : [], 'body' : []} }

    # list of dictionaries {}
    # each dictionary := { "id_no" : {} } pairs
    # each sub-dictionary := { "word" : score } pairs
    weights = dict()

    # perform tfidf
    m = tfidf()

    # add documents to tfidf objects
    print('Adding documents for TF-IDF...')
    for i, document in enumerate(documents):
        m.addDocument(i, document['words']['title']+document['words']['body'])
        weights[i] = dict()

    # generate dictionary of { "word", "score" } pairs for each document
    print('Generating weight scores for words...')
    for word in lexicon['title'] | lexicon['body']:
        print('Generating weights for word:', word)
        m.get_similarities(word, weights)

    # generate feature list
    print('Selecting features for the feature vector...')
    features = set()
    for doc, doc_dict in weights.iteritems():
        scorer = Counter(doc_dict)
        for term in scorer.most_common(5):
            if term[1] > 0.0:
                features.add(term[0])

    # UNCOMMENT WHEN DEBUGGING
    print(features)
    print(len(features))

    # TODO: select largest 1000 values
    # TODO: select words (<= 1000) for values => make list

    # TODO: generate feature vector for each document

    # TODO: write rows vectors to dataset1.csv
    dataset = open("dataset1.csv", "w")

    dataset.close()

    print('Finished generating dataset @ dataset1.csv!')
