"""Ronda 008: ponderación del bloque de tercios con normalización conjunta.

Separa la representación 007 en dos vectorizadores (base = 005 exacta; thirds =
solo etiquetas `third0/1/2:palabra`) y recombina `[base, w * thirds]` con
renormalización L2 de toda la fila. Esto permite mover el peso `w` del bloque de
tercios como único factor, con `w=1.0` como control técnico del propio split.

La referencia exacta de la 007 (un solo TF-IDF con tercios dentro) se construye en
experimento_008 con `TextFeatures007b(thirds=True)`; aquí solo se implementan los
candidatos "split". El vocabulario y el IDF de cada bloque se aprenden dentro del
fold. Ningún analizador consulta etiquetas ni eval.
"""
import re
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.features_007b import clean, discourse_tokens


def thirds_tokens(text):
    text = clean(text)
    raw = re.findall(r'\b\w+\b', text)
    for i, w in enumerate(raw):
        yield 'third' + str(min(2, 3 * i // max(1, len(raw)))) + ':' + w


class BaseAnalyzer:
    def __call__(self, text):
        return discourse_tokens(text, global_max_ngram=2, mid_bucket=False,
                                edge_bigrams=False, thirds=False)


class ThirdsAnalyzer:
    def __call__(self, text):
        return thirds_tokens(text)


class SplitThirdsFeatures(BaseEstimator, TransformerMixin):
    """Bloques base y thirds vectorizados por separado y recombinados con peso."""

    def __init__(self, third_weight=1.0, min_df=3, max_features=160000):
        self.third_weight = third_weight
        self.min_df = min_df
        self.max_features = max_features

    def fit(self, X, y=None):
        self.base_ = TfidfVectorizer(analyzer=BaseAnalyzer(), min_df=self.min_df,
                                     sublinear_tf=True, max_features=self.max_features).fit(X)
        self.third_ = TfidfVectorizer(analyzer=ThirdsAnalyzer(), min_df=self.min_df,
                                      sublinear_tf=True, max_features=self.max_features).fit(X)
        return self

    def transform(self, X):
        B = self.base_.transform(X)
        T = self.third_.transform(X)
        C = sp.hstack([B, self.third_weight * T], format='csr')
        return normalize(C, norm='l2', copy=False)
