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

###############################################################################
########## global variables for single-point of control over change ###########
###############################################################################

datafile = 'dataset1.csv'

###############################################################################
############### function for printing dataset to .csv document ################
###############################################################################

def generate_csv(documents, features, weights):
    """ function: generate_csv
        ----------------------
        print feature vectors & class labels to .csv file

        :param documents: dictionary of document objects
        :param features: sorted list of features to represent
    """
    dataset = open(datafile, "w")
    dataset.write('id\t')
    for feature in features:
        dataset.write(feature)
        dataset.write('\t')
    dataset.write('class-label:topics\t')
    dataset.write('class-lable:places\t')
    dataset.write('\n')
    # feature vector for each document
    for i, document in enumerate(documents):
        # document id number
        dataset.write(str(i))
        dataset.write('\t')
        # each tf-idf score
        for feature in features:
            dataset.write(str(weights[i][feature]))
            dataset.write('\t')
        # topics/places class labels
        dataset.write(str(document['topics']))
        dataset.write(str(document['places']))
        dataset.write('\n')
    dataset.close()

###############################################################################
###################### function(s) for feature selection ######################
###############################################################################

def select_features(weights):
    """ function: select_features
        -------------------------
        generated reduced feature list for vector generation

        :param weights: dictionary from results of the tf-idf calculations
        :returns: sorted list of terms representing the selected features
    """
    features = set()
    for doc, doc_dict in weights.iteritems():
        scorer = Counter(doc_dict)
        for term in scorer.most_common(5):
            if term[1] > 0.0:
                features.add(term[0])
    # sort set into list
    return sorted(features)

###############################################################################
############## function(s) for generating weighted tf-idf scores ##############
###############################################################################

def generate_weights(documents, lexicon):
    """ function: generate_weights
        --------------------------
        perform tf-idf to generate importance scores for words in documents

        :param document: list of documents to use in calculations
        :returns: dictionary of dictionaries: {"id_" : {"word" : score,...}}
    """
    # weight = { 'document' : { 'word' : score,... },... }
    weights = dict()
    m = tfidf()
    print('Adding documents for TF-IDF...')
    for i, document in enumerate(documents):
        m.addDocument(i, document['words']['title']+document['words']['body'])
        weights[i] = dict()
    # generate dictionary of { "word", "score" } pairs for each document
    print('Generating weight scores for words; This WILL take time...')
    for word in lexicon['title'] | lexicon['body']:
        # UNCOMMENT FOR SANITY
        # print('Generating weights for word:', word)
        m.get_similarities(word, weights)
    return weights

###############################################################################
################ main function for generating refined dataset #################
###############################################################################

def generate_dataset(documents, lexicon):
    """ function: generate_dataset
        --------------------------
        select features from @lexicon for feature vectors
        generate dataset of feature vectors for @documents

        :param documents: list of well-formatted, processable documents
        :param lexicon:   list of word stems for selecting features
    """
    print '\nGenerating dataset @', datafile
    weights = generate_weights(documents, lexicon)

    # generate feature list
    print 'Selecting features for the feature vectors @', datafile
    features = select_features(weights)

    # write vectors to dataset1.csv
    print 'Writing feature vector data @', datafile
    generate_csv(documents, features, weights)
    print 'Finished generating dataset @', datafile
