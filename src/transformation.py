import numpy as np
import re
import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from hunspell import Hunspell
import string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import logging as log

h = Hunspell('es-CR', hunspell_data_dir='/code/include/huspell/dic/es-CR')
spanishStemmer = SnowballStemmer("spanish", ignore_stopwords=True)
correctedToken = pd.read_json('/code/include/correction.json', orient='index')
comments = pd.read_csv('/code/include/proccesed_comments.csv', index_col='id')
variables = comments.drop('primary_category', axis=1)
response = comments['primary_category'].values


def predict(model, comment, creator_department, resource_type):
    if comment == "":
        return 'Ninguno', ["0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%"]

    test_predict = pd.DataFrame(data=[
        {
            'creator_department': creator_department,
            'resource_type': resource_type,
            'comment': comment
        }
    ], index=[1])

    preprocess = ColumnTransformer([
        ('preprocess', Pipeline(steps=[
            ('LowerCaser', LowerCaser()),
            ('PuntuationRemover', PuntuationRemover()),
            ('Striper', Striper()),
            ('TokenCleaner', TokenCleaner()),
            ('WordCorrector', WordCorrector()),
            ('StopWordsRemover', StopWordsRemover()),
            ('SingleLetterRemover', SingleLetterRemover()),
            ('Stemmer', Stemmer()),
        ]), ['tokens']),
    ], remainder='passthrough')
    test_predict['tokens'] = test_predict. \
        apply(lambda row: word_tokenize(row['comment']), axis=1)
    proccessed = preprocess.fit_transform(test_predict.reset_index())

    proccessed = pd.DataFrame(data=proccessed, columns=[
        "tokens",
        "index",
        "creator_department",
        "resource_type",
        "comment"
    ])

    proccessed['comment'] = proccessed['tokens'].apply(lambda x: ' '.join(x))
    to_predict = proccessed[[
        'creator_department',
        'resource_type',
        'comment']]

    label = model.predict(to_predict)[0]
    prob = model.predict_proba(to_predict)[0]

    return label.capitalize(), ['{}%'.format(x) for
                                x in np.round(np.array([p * 100
                                                        for p in prob]), 2)]


def fit_smv():
    svm_optimal_pipeline = Pipeline(steps=[
        ('all', FeatureUnion(transformer_list=[
            ('cat_feature', Pipeline(steps=[
                ('selector', CategoricalFeatureSelector()),
                ('encoding', OneHotEncoder())
            ])),
            ('txt_feature', Pipeline(steps=[
                ('selector', TxtFeatureSelector()),
                ('vectorizer',
                 TfidfVectorizer(ngram_range=(1, 1), binary=False)),
            ]))
        ])),
        ('fect_selec', SelectKBest(chi2, k=1200)),
        ('model', SVC(kernel='linear', gamma="scale", C=2, probability=True))
    ])

    svm_optimal_pipeline.fit(variables[[
        'creator_department',
        'resource_type',
        'comment']], response)

    return svm_optimal_pipeline


# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names=None):
        self._feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # select a particular set of features to transform

    def transform(self, X, y=None):
        return X[self._feature_names]

    def get_params(self, deep=True):
        return {}


class TxtFeatureSelector(FeatureSelector):
    def transform(self, X, y=None):
        return X['comment']


class CategoricalFeatureSelector(FeatureSelector):
    def transform(self, X, y=None):
        return X[['creator_department', 'resource_type']]


class OneHotVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        sparse_matrix = self.vectorizer.fit_transform(documents)
        return [freq.toarray()[0] for freq in sparse_matrix]


class Preprocessing(BaseEstimator, TransformerMixin):
    # not used
    def fit(self, X, y=None):
        return self

    # return tokens column as received
    def convert(self, X, CallBack):
        return pd.DataFrame(
            X.apply(lambda row: CallBack(self.getToken(row)), axis=1),
            columns=['tokens'])

    def getToken(self, row):
        return row['tokens']


class LowerCaser(Preprocessing):
    # returns numpy.ndarray
    def toLower(self, tokens):
        return np.char.lower(tokens)

    # Convert the column to lower case
    def transform(self, X, y=None):
        return self.convert(X=X, CallBack=self.toLower)


class PuntuationRemover(Preprocessing):
    # returns numpy.ndarray
    def removePuntuation(self, tokens):
        result = np.empty((0, 0), dtype=str, order='C')
        for token in tokens:
            if token not in string.punctuation:
                result = np.append(result, token)
        return result

    # remove puntuation symbols from the tokens
    def transform(self, X, y=None):
        return self.convert(X=X, CallBack=self.removePuntuation)


class Striper(Preprocessing):
    # returns numpy.ndarray
    def strip(self, tokens):
        result = np.empty((0, 0), dtype=str, order='C')
        for token in tokens:
            token = token.strip()
            if token != "":
                result = np.append(result, token)
        return result

    # strip each token removing empty spaces
    def transform(self, X, y=None):
        return self.convert(X=X, CallBack=self.strip)


class StopWordsRemover(Preprocessing):
    # returns numpy.ndarray
    def removeStopWords(self, tokens):
        result = np.empty((0, 0), dtype=str, order='C')
        for token in tokens:
            if token in stopwords.words('spanish'):
                continue
            # remove the subjet of the sentences
            if str(token).lower() in ['leonardo.quintanilla', 'hulihealth.com',
                                      'dulce.rodriguez', 'liz.barrantes',
                                      'hulilabs.com', 'doctor', 'doctora', 'dr',
                                      'dra', 'secretaria', 'secretario',
                                      'doctores', 'doctoras', 'cemim', 'http',
                                      'https']:
                continue
            result = np.append(result, token)
        return result

    # remove stops words from the sequence of tokens
    def transform(self, X, y=None):
        return self.convert(X=X, CallBack=self.removeStopWords)


class TokenCleaner(Preprocessing):
    # returns numpy.ndarray
    def tokenCleaner(self, tokens):
        result = np.empty((0, 0), dtype=str, order='C')
        for token in tokens:
            token = re.sub('[^A-Za-z]+', '', token)
            if token.isalpha():
                result = np.append(result, token)
        return result

    # remove special characters from word
    def transform(self, X, y=None):
        return self.convert(X=X, CallBack=self.tokenCleaner)


class WordCorrector(Preprocessing):
    # returns numpy.ndarray
    def correctWord(self, tokens):
        result = np.empty((0, 0), dtype=str, order='C')
        for token in tokens:
            token = str(token)
            if not h.spell(token):
                if token in correctedToken.index:
                    token = correctedToken.loc[token, :][0]
                else:
                    correction = h.suggest(token)
                    # take the first suggestion as the corrected value
                    if len(correction) != 0:
                        token = correction[0]
            # some of the token are identify to different words so we are separating those words
            separatedTokens = word_tokenize(token)
            result = np.append(result, separatedTokens)
        return result

    # correct words in the sentence
    def transform(self, X, y=None):
        return self.convert(X=X, CallBack=self.correctWord)


class Stemmer(Preprocessing):
    # returns numpy.ndarray
    def stemm(self, tokens):
        result = np.empty((0, 0), dtype=str, order='C')
        for token in tokens:
            result = np.append(result, spanishStemmer.stem(str(token)))
        return result

    # Get the root part of the word to reduce dimensionality
    def transform(self, X, y=None):
        return self.convert(X=X, CallBack=self.stemm)


class SingleLetterRemover(Preprocessing):
    # returns numpy.ndarray
    def removeSingleLetter(self, tokens):
        result = np.empty((0, 0), dtype=str, order='C')
        for token in tokens:
            if len(str(token)) > 1:
                result = np.append(result, token)
        return result

    # remove all single letter token
    def transform(self, X, y=None):
        return self.convert(X=X, CallBack=self.removeSingleLetter)
