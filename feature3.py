#!/usr/bin/python

""" feature3.py
    -----------
    @author = Ankai Lou
"""

import os
import sys
import string
import nltk
import tfidf

def generate_dataset(document, lexicon):
    """ function: generate_dataset
        --------------------------
        select features from @lexicon for feature vectors
        generate dataset of feature vectors for @documents

        :param document: list of well-formatted, processable documents
        :param lexicon:  list of word stems for selecting features
    """
    print('Generating Dataset @ dataset.csv')
    print('Finished Generating Dataset @ dataset.csv')
