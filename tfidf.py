#!/usr/local/python-2.7.5/bin/python

""" tfidf.py
    --------
    @author = Ankai Lou
"""

import sys
import os
import math

###############################################################################
##### class for term frequency - inverse document frequency functionality #####
###############################################################################

class tfidf:
    def __init__(self):
        """ function: constructor
            ---------------------
            :param documents: store word occurrences per document
            :param occurrences: store word occurrences for all documents
        """
        self.documents = dict([])
        self.occurrences = dict({})

    ###########################################################################
    ########## function(s) for collecting information on document set #########
    ###########################################################################

    def addDocument(self, name, words):
        """ function: addDocument
            ---------------------
            add document @doc_name to dictionary
            construct corpus from @list_of_words

            :param name: name to identify document
            :param words: list of terms in document
        """
        doc_dict = dict([])
        occurrence = False
        for w in words:
            # occurrences per documents
            doc_dict[w] = doc_dict.get(w, 0.0) + 1.0
            if not occurrence:
                # documents with the word
                self.occurrences[w] = self.occurrences.get(w, 0.0) + 1.0
                occurrence = True
        # normalizing tf by document length
        length = float(len(words))
        for key in doc_dict:
            doc_dict[key] = doc_dict[key] / length
        # add the normalized document to the corpus
        self.documents[name] = doc_dict

    ###########################################################################
    ############### general function for computing tf-idf score ###############
    ###########################################################################

    def get_similarities(self, word, weights, type='normal', scaling=1.0):
        """ function: get_similarities
            --------------------------
            generator function for feature vectors

            :param word: term to calculate tf-idf score
            :param weights: dictionary to write scores
        """
        # compute list of similarities
        num_docs = len(self.documents)
        for doc, doc_dict in self.documents.iteritems():
            score = 0.0
            if self.occurrences.has_key(word) and doc_dict.has_key(word):
                if type == 'normal':
                    score += self.normal(word, doc_dict[word], num_docs)
                elif type == 'smooth':
                    score += self.smooth(word, doc_dict[word], num_docs)
                else:
                    print type, 'is not a valid td-idf function'
                    sys.exit(1)
                score *= scaling
            weights[doc][word] = score

    ###########################################################################
    ############ function(s) for computing tf-idf in different ways ###########
    ###########################################################################

    def normal(self, word, freq, num_docs):
        """ function: normal
            ----------------
            tfidf using raw frequency & inverse frequency

            :param word: term in tf-idf calculation
            :param freq: number of times word appears in document
            :param num_docs: total number of documents in the set
        """
        tf = 0.5 + (0.5 * freq)
        idf = math.log(num_docs / self.occurrences[word], num_docs)
        return tf * idf

    def smooth(self, word, freq, num_docs):
        """ function: smooth
            ----------------
            tfidf using smooth raw frequency & inverse frequency

            :param word: term in tf-idf calculation
            :param freq: number of times word appears in document
            :param num_docs: total number of documents in the set
        """
        tf = math.log(1 + freq)
        idfN = (num_docs - self.occurrences[word]) / self.occurrences[word]
        idf = math.log(1 + idfN)
        return tf * idf
