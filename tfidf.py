#!/usr/bin/python

""" tfidf.py
    --------
    @author = Ankai Lou
"""

import sys
import os
import math

class tfidf:
    def __init__(self):
        """ function: constructor
            ---------------------
            :param documents: store word occurrences per document
            :param occurrences: store word occurrences for all documents
        """
        self.documents = dict([])
        self.occurrences = dict({})

    def addDocument(self, name, words):
        """ function: addDocument
            ---------------------
            add document @doc_name to dictionary
            construct corpus from @list_of_words

            :param name: name to identify document
            :param words: list of terms in document
        """
        # calculations
        doc_dict = dict({})
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

    def get_similarities(self, word, weights):
        """ function: similarities
            ----------------------
            list of all the [docname, similarity_score]
            pairs relative to a list of terms
        """

        # computing the list of similarities
        sims = []
        num_docs = len(self.documents)
        for doc, doc_dict in self.documents.iteritems():
            score = 0.0
            if self.occurrences.has_key(word) and doc_dict.has_key(word):
                tf = 0.5 + (0.5 * doc_dict[word])
                idf = math.log( num_docs / self.occurrences[word] )
                score = tf * idf
                # UNCOMMENT WHEN DEBUGGING
                # print(score)
            weights[doc][word] = score
