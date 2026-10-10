"""Ronda 009: estrategias genuinamente distintas de la 007/008.

- NBSVM (Wang & Manning 2012): re-ponderación de TF-IDF por log-odds de Naive Bayes
  aprendidos de las etiquetas de entrenamiento, con regresión logística uno-contra-
  resto por clase. Todo scikit-learn; sin lexicones ni datos externos.
- El canal char_wb y la regularización L1 se implementan en experimento_009.py.

Ningún componente consulta eval; el vocabulario, los conteos NB y el IDF se aprenden
dentro del fold.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression


class NBSVM(BaseEstimator, ClassifierMixin):
    """NB-SVM uno-contra-resto: LogisticRegression sobre TF-IDF * log-odds(NB)."""

    def __init__(self, ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                 max_features=160000, C=4.0, alpha=1.0, max_iter=2500, random_state=42):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features
        self.C = C
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.tfidf_ = TfidfVectorizer(ngram_range=self.ngram_range, min_df=self.min_df,
                                      sublinear_tf=self.sublinear_tf,
                                      max_features=self.max_features).fit(X)
        Xt = self.tfidf_.transform(X)
        cnt = CountVectorizer(vocabulary=self.tfidf_.vocabulary_, binary=True).fit(X)
        C = cnt.transform(X)
        self.ratios_ = []
        self.models_ = []
        for c in self.classes_:
            mask = (y == c)
            n_c = np.asarray(C[mask].sum(axis=0)).ravel().astype(np.float64)
            n_notc = np.asarray(C[~mask].sum(axis=0)).ravel().astype(np.float64)
            r = np.log((n_c + self.alpha) / (n_notc + self.alpha))
            Xr = Xt.multiply(r)
            m = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                   random_state=self.random_state)
            m.fit(Xr, mask.astype(int))
            self.ratios_.append(r)
            self.models_.append(m)
        return self

    def decision_function(self, X):
        Xt = self.tfidf_.transform(X)
        cols = []
        for r, m in zip(self.ratios_, self.models_):
            Xr = Xt.multiply(r)
            cols.append(m.decision_function(Xr))
        return np.column_stack(cols)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        d = self.decision_function(X)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
