Report 1 - Feature Vectors Methodology
======================================

## Text Extraction & Processing

### Approach

Preceding the feature selection & dataset generation phase for each feature vector, the same methodology for extracting the relevant text from the files is employed for all three datasets. The goal of the text extraction & processing phase was to:

* For each file, the `BeautifulSoup4` library was used to generate a well-formatted parse-tree from the SGML using the built-in Python `html.parser` library, this led into the following text extraction process:
    * For each parse-tree, text blocks for distinct articles - delimited by `<reuters>` tags - were separated into a list of `strings`.
    * For each article, the text in the `<topics>` and `<places>` tags - delimited by `<d>` - were extracted as the class labels.
    * For each article, the text in the `<title>` and `<body>` tags were extracted into `strings` for the tokenization process:

* For each title/body text block, the `NLTK` and `string` libraries were used for filtering & tokenization:     
    * For each text field, the digits & unicode characters were replaced by `None` using the `string` library.
    * For each text field, the punctuation characters were replaced by `None` using the `string` library.
    * For each text field, the text chunks were tokenized into lists of terms using the `nltk.word_tokenize()` function.
    * For each token list, the `stopwords` import from `nltk.corpus` were filtered from the lists.
    * For each token list, the `non-english` words were filtered using the `nltk.corpus.wordnet.synsets()` function.
    * For each token list, the tokens were 'lemmatized' using the `nltk.stem.wordnet.WordLemmatizer` package.
    * For each token list, the lemmatized tokens were stemmed using the `nltk.stem.porter.PorterStemmer()` package.
    * For each stem list, the word stems shorter than 4 characters were filtered from the list

The final output of the text extraction & processing phase is a list of documents:

```
documents = []
```

For each `document` dictionary element in this list, the key-value pairs include:

```
document = { 'topics' : [], 'places' : [], 'words' : { 'title' : [], 'body' : [] } }
```

From this list of documents, a lexicon was generated for all unique words that appear in the title and body fields.

```
lexicon = { 'title' = set() : 'body' = set() }
```

This concludes the text extraction $ processing phase and prepared the file input for feature selection.

### Rationale

Several portions of the text processing & tokenization phase were non-essential and included for specific reasons:

* The digits, unicode characters, and punctuation symbols were removed from the original text because digits & meta-characters provide less valuable information to article classification as opposed to actual words.
* The stop words were removed from the text because words such as `the` and `was` provide minimal contextual value yet appear frequently. Though the tf-idf process would inevitably filter out stopwords during the weight & feature selection phase, removing stop words at the tokenization phase removes several thousand $n^2$ time calculations during the tf-idf calculations in linear time.
* The non-English words were filtered from the text because the stemming process used the English version of the Porter stemmer; therefore, stemming non-English words is highly likely to produce erroneous data and inflate the size of the lexicon.
* The tokens were lemmatized to reduce the dataset by grouping similar words such as `foot` and `feet` before stemming
* The tokens were stemmed using the Porter Stemmer to reduce the dataset further by minimizing the size of the lexicon
* The stems shorter than 4 characters were filtered from the dataset because one-letter or other sufficiently short stems appear frequently in articles yet provide little importance to classification - similar to stop words.

This minification of the token lists & lexicons ensure the minimum number of calculation during the selection phase while not losing any valuable information. The same feature reduction process was used across all 3 feature vector sets because of the low-risk, high-reward nature of filtering out low-value words & stems from the data. A unified text processing methodology also reduced the runtime of the code.

## Generating Feature Vector 1

### Approach

The first approach is the simplest and most naive approach to feature selection.

### Rationale

The simplest approach was employed first to provide a quality baseline for later feature vectors and to provide a foundation for more precise selection processes.

## Generating Feature Vector 2

### Approach

### Rationale

## GeneratingFeature Vector 3

### Approach

### Rationale