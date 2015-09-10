#!/usr/bin/python

""" preprocess.py
    -------------
    @author = Ankai Lou
"""

import os
import sys
import string
import nltk
import threading # will potentially use multi-threading
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup

###############################################################################
###### modules for feature selection & feature vector dataset generation ######
###############################################################################

import feature1
import feature2
import feature3

###############################################################################
################ function(s) for generating document objects ##################
###############################################################################

def init_document():
    """ function: init_document
        -----------------------
        initialize new empty document skeleton

        :returns: dictionary @document of document fields
            @dictionary['topics'] is a list representing topic class labels
            @dictionary['places'] is a list representing place class labels
            @dictionary['words'] is a dictionary
            @dictionary['words']['title'] is a list for the title text terms
            @dictionary['words']['body'] is a list for the body text terms
    """
    document = { 'topics' : [], 'places' : [], 'words' : dict([]) }
    document['words']['title'] = []
    document['words']['body']  = []
    return document

def populate_class_label(document, article):
    """ function: populate_class_label
        ------------------------------
        extract topics/places from @article and fill @document

        :param document: formatted dictionary object representing a document
        :param article:  formatted parse tree built from unformatted data
            @article is a 'reuter' child of the original file parsetree
    """
    for topic in article.topics.children:
        document['topics'].append(topic.text.encode('ascii', 'ignore'))
    for place in article.places.children:
        document['places'].append(place.text.encode('ascii', 'ignore'))

def populate_word_list(document, article):
    """ function: populate_word_list
        ----------------------------
        extract title/body words from @article, preprocess, and fill @document

        :param document: formatted dictionary object representing a document
        :param article:  formatted parse tree built from unformatted data
            @article is a 'reuter' child of the original file parsetree
    """
    text = article.find('text')
    title = text.title
    body = text.body

    if title != None:
        document['words']['title'] = tokenize(title.text)
    if body != None:
        document['words']['body'] = tokenize(body.text)

def tokenize(text):
    """ function: tokenize
        ------------------
        generate list of tokens given a block of @text;

        :param text: string representing text field (title or body)
        :returns: list of strings of tokenized & sanitized words
    """
    # encode unicode to string
    ascii = text.encode('ascii', 'ignore')
    # remove digits
    no_digits = ascii.translate(None, string.digits)
    # remove punctuation
    no_punctuation = no_digits.translate(None, string.punctuation)
    # tokenize
    tokens = nltk.word_tokenize(no_punctuation)
    # remove stopwords - assume 'reuter'/'reuters' are also irrelevant
    irrelevant = stopwords.words('english') + ['reuters', 'reuter']
    no_stop_words = [w for w in tokens if not w in stopwords.words('english')]
    # filter out non-english words
    eng = [y for y in no_stop_words if wordnet.synsets(y)]
    # lemmatization process
    lemmas = []
    lmtzr = WordNetLemmatizer()
    for token in eng:
        lemmas.append(lmtzr.lemmatize(token))
    # stemming process
    stems = []
    stemmer = PorterStemmer()
    for token in lemmas:
        stems.append(stemmer.stem(token).encode('ascii','ignore'))
    # remove short stems
    terms = [x for x in stems if len(x) >= 4]
    return terms

def generate_document(text):
    """ function: generate_document
        ---------------------------
        extract class labels & tokenized (and sanitized) title/body text

        :param text: parsetree of 'reuter' child in original parsetree
        :returns: dictionary representing fields of single document entity
    """
    document = init_document()
    populate_class_label(document, text)
    populate_word_list(document, text)
    # UNCOMMENT WHEN DEBUGGING
    # print(document)
    return document

###############################################################################
############ function(s) for generating parse tree from .sgm files ############
###############################################################################

def generate_tree(text):
    """ function: generate_tree
        -----------------------
        extract well-formatted tree from poorly-formatted sgml @text

        :param text: string representing sgml text for a set of articles
        :returns: parsetree @tree of the structured @text
    """
    return BeautifulSoup(text, "html.parser")

###############################################################################
########## function(s) for generating parse trees & document objects ##########
###############################################################################

def parse_documents():
    """ function: parse_document
        ------------------------
        extract list of Document objects from token list

        :returns: list of document entities generated by generate_document()
    """
    documents = []
    # generate well-formatted document set for each file
    for file in os.listdir('data'):
        # open 'reut2-XXX.sgm' file from /data directory
        data = open(os.path.join(os.getcwd(), "data", file), 'r')
        text = data.read()
        data.close()
        tree = generate_tree(text.lower())
        # separate segments & generate documents
        for reuter in tree.find_all("reuters"):
            document = generate_document(reuter)
            documents.append(document)
        print "Finished extracting information from file:", file
    return documents

###############################################################################
################## main function - single point of execution ##################
###############################################################################

def main(argv):
    """ function: main
        --------------
        sanitize input files into well-formatted, processable objects
        generate dataset (feature vectors, class labels) for .sgm file set:

        :param argv: command line arguments - no purpose at the moment
    """
    # generate list of document objects for feature selection
    print('Generating document objects. This may take some time...')
    documents = parse_documents()

    # generate lexicon of unique words for feature reduction
    print('Document generation complete. Building lexicon...')
    lexicon = { 'title' : set(), 'body' : set() }
    for document in documents:
        for term in document['words']['title']:
            lexicon['title'].add(term)
        for term in document['words']['body']:
            lexicon['body'].add(term)

    # UNCOMMENT WHEN DEBUGGING
    # print(len(lexicon['title']))
    # print(len(lexicon['body']))

    # generate dataset 1 w tfidf (using feature1 module)
    feature1.generate_dataset(documents, lexicon)

    # generate dataset 2 w tfidf (using feature2 module)
    feature2.generate_dataset(documents, lexicon)

    # generate dataset 3 w tfidf (using feature3 module)
    feature3.generate_dataset(documents, lexicon)

if __name__ == "__main__":
    main(sys.argv[1:])
