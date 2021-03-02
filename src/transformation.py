from sklearn.base import BaseEstimator, TransformerMixin
from nltk.cluster import KMeansClusterer
from sklearn.cluster import DBSCAN

#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names=None):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    # select a particular set of features to transform
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 
    
    def get_params(self, deep=True):
        return {}

class TxtFeatureSelector(FeatureSelector):
    def transform( self, X, y = None ):
        return X['comment'] 

class CategoricalFeatureSelector(FeatureSelector):
    def transform( self, X, y = None ):
        return X[['creator_department', 'resource_type']] 

class OneHotVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
    
    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents):
        sparse_matrix = self.vectorizer.fit_transform(documents)
        return [freq.toarray()[0] for freq in sparse_matrix]

class KmeansCustom(BaseEstimator, TransformerMixin):
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance
    
    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        model = KMeansClusterer(self.k, self.distance, repeats=10, avoid_empty_clusters=True)
        return model.cluster(documents, True)

class DBScanCustom(BaseEstimator, TransformerMixin):
    def __init__(self, eps, min_samples, metric):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit(documents)
        return model.labels_ 