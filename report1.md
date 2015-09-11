Report 1 - Feature Vectors
==========================

## Text Extraction & Processing

### Approach

The same methodology for extracting the text from the files is used for all datasets:

* For each file, the `BeautifulSoup4` library was used to generate a parse-tree from the SGML using the built-in Python `html.parser` library:
    * For each parse-tree, article blocks - delimited by `<reuters>` - were separated into strings.
    * For each article, the text in `<topics>` and `<places>` delimited by `<d>` were used as class labels; the text in `<title>` and `<body>` were extracted for tokenization.

* For each title/body text block, the `NLTK` and `string` libraries were used for tokenization:     
    * For each field, digits & unicode symbols were replaced by `None` using the `string` library.
    * For each field, punctuation symbols were replaced by `None` using the `string` library.
    * For each field, text blocks were tokenized into lists using `nltk.word_tokenize()`.
    * For the tokens, `stopwords` from `nltk.corpus` were filtered from the lists.
    * For the tokens, non-English words were filtered via `nltk.corpus.wordnet.synsets()`.
    * For the tokens, lemmatization was used via `nltk.stem.wordnet.WordLemmatizer()`.
    * For the tokens, tokens were stemmed via `nltk.stem.porter.PorterStemmer()`.
    * For each stem list, the word stems shorter than 4 characters were filtered from the list

The final output of the text extraction & processing phase is a list of documents:

```
documents = [document = {'topics' : [], 'places' : [], 'words' : {'title' : [], 'body' : []}}]
```

From this list, a lexicon was generated for all unique words in titles and body fields:

```
lexicon = { 'title' = set() : 'body' = set() }
```

This concludes the text extraction $ processing phase and prepared the file input for feature selection.

### Rationale

Several portions of the text processing & tokenization phase were selected for specific reasons:

* Digits, unicode characters, and punctuation symbols were removed from the text because digits & meta-characters provide less valuable information to article context than actual words.
* Stopwords were removed from the text because words such as `the` are frequently present yet provide no contextual value. Though the tf-idf process would inevitably filter stopwords during the weighting phase, removing stopwords at tokenization removes several polynomial time calculations during the tf-idf calculations in linear time - a desirable improvement in performance.
* The non-English words were filtered from the text because the stemmer used the English Porter stemmer; therefore, stemming non-English words is likely to produce erroneous data and artificially inflate the size of the lexicon - which will increase the runtime.
* The tokens were lemmatized to reduce the dataset by removing tenses before stemming.
* The tokens were stemmed to reduce the dataset further by minimizing the size of the lexicon.
* The stems shorter than 4 characters were filtered because sufficiently short stems appear frequently in articles yet provide little importance to classification - similar to stop words.

This minification of the tokens & lexicon ensure a minimum number of calculations during the selection phase, while not losing valuable information or context. The same feature reduction process was used for all feature vectors because filtering out low-value words & stems from the data is low-risk/high-reward for data quality and runtime. A unified text processing methodology also reduced the runtime.

## Generating Feature Vector 1

### Approach

The first approach is the simplest and most naive approach to feature selection. No weighting was used for different word groups (e.g. title/body, part-of-speech, bigrams/trigrams, etc). For weight score generation: a term frequency-inverse document frequency score for each `(document, word)` pair was calculated using the simple formula:

> tf = 0.5 + (0.5 * ( # times word appears in document / # of words in the document ))

> idf = log_{# of documents}( # of documents / # of documents that contain the word)

> tf-idf score = scaling * tf * idf

where `scaling` is an arbitrary non-zero floating-point (default=1.0) for weighting specific groups of words. These calculations return a 2d matrix of `(document, word)` scores.

For the feature selection phase: for each document, the top 5 words with the highest non-zero tf-idf score were added to the feature set. The size of the feature set was approximately ~2500. The final dataset is a 20000x2500 matrix with two additional columns for topic and place class-labels. The results for feature vector 1 can be accessed after running the code via the command:

```
> less dataset1.csv
```

### Rationale

A naive approach was employed first to provide a baseline of quality and performance. The naive approach provides the fastest implementation, an introduction to feature selection, and recyclable code.

The tf-idf calculation was normalized the tf-score and idf-score to the interval [0,1] - 0 meaning completely unimportant; 1 meaning absolute importance. This provides a metric for measuring word importance to a document without being skewed by document size or an abundance of low-value words. Given a scaling constant of 1.0, the final tf-idf score for each (document,word) pair is in the [0,1] interval.

Feature selection filtered the top 5 words for each document. The top five words for each document (assuming quality tf-idf weighting) provides a sufficient metric for representing a document by class. The feature set of ~2500 was not reduced further to 500-1000 because removing 60% of the words would inevitably filter out important information on certain types of articles. 

## Generating Feature Vector 2

### Approach

The second approach for feature selection is built on the code for dataset1.csv. The second attempt at feature vector construction alters the weighting of different word groups, the calculation of the tf-idf score, and the feature selection methodology.

The ti-idf score for each `(document, word)` pair in this approach weighted words present in the title AND body 1.25x (25% boost), words that only appear the in title 1.1x (10% boost), and words that only appear in the body no additional weight. 

For the formula used for the tf-idf score calculation: the term frequency (tf) score was altered to use log normalization instead of raw frequency; and the inverse document frequency (idf) score was changed to use probabilistic inverse frequency over raw inverse frequency. The equations are provided below:

> tf = log_2(1 + # times word appears in document / # of words in the document)

> idf = log_2( 1 + ((# of documents - # of documents containing the word) / # of documents that containing word))

> tf-idf score = scaling * tf * idf

For feature selection, the selection process was changed to averaging the columns of the tf-idf matrix, and selecting the words corresponding to the largest 1000 values. The final dataset is a 20000x1000 matrix with two additional columns for topic and place class-labels. The results for feature vector 2 can be accessed after running the code via the command:

```
> less dataset2.csv
```

### Rationale

The weighting title words higher than body words is based on the assumption that the purpose of the title is to summarize the body text. The implication being that title words provide more compact & valuable information to the article's class labels than body words. Words present in both the title & body are given an even stronger weighting (1.25x vs 1.1x) because words present in the title and re-emphasized in the body text are very strongly tied to the class labels of the article.

The application of a weight on title words over body wordsdid not significantly change the feature vector for dataset2.csv from dataset1.csv. Therefore, the tf-idf calculation formula was also altered. The log normalization and probabilistic inverse frequency are used in order to decrease the raw tf-score for larger frequency values to counterbalance the additional weighting of title words. The tf-idf scores were not normalized to the interval [0,1] because the 1.1x and 1.d25x weightings cannot be normalized to [0,1].

The result of selecting the top 5 words for each article to construct the feature set generated a set of over 10000 words - which is far too large for this project. Therefore, a different feature selection method was used to select exactly 1000 words. The averaging of the word column was employed to select the most important words to ALL documents to counterbalance the 1000 word constraint.

## GeneratingFeature Vector 3

### Approach

filter tags verb/nouns/etc

use scikit-learn tf-idf

use same feature selection as feature vector 2

```
> less dataset3.csv
```

### Rationale