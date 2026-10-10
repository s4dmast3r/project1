"""Ronda 012: rasgos densos de resumen de sentimiento (diagnóstico de polaridad).

El diagnóstico mostró que los errores positivo<->negativo de la 009 tienen ~2x
más marcadores de negación/opinión y más conectores de contraste que los aciertos:
son reseñas de aspecto mixto. El modelo lineal sobre TF-IDF disperso no expresa
fácilmente "esta reseña tiene a la vez señal positiva y negativa".

Estos rasgos densos lo hacen explícito. El "léxico" interno de polaridad se aprende
DENTRO del fold a partir de las etiquetas de entrenamiento (razón log-odds NB de
cada palabra entre positivo y negativo), NO es un lexicón externo. Los marcadores de
negación y contraste son listas fijas de NLP clásico, no reglas de etiqueta.
"""
import re
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

NEGATORS = {'no', 'nunca', 'sin', 'ni', 'jamas', 'tampoco'}
CONTRAST = re.compile(r'\b(?:pero|aunque|sin embargo|no obstante|aun asi|eso si|por otro lado|de paso|por cierto)\b')

from src.features_007b import clean


class DenseSentimentFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, k=150, alpha=1.0, use_lexicon=True):
        self.k = k
        self.alpha = alpha
        self.use_lexicon = use_lexicon

    def fit(self, X, y=None):
        y = np.asarray(y)
        if self.use_lexicon and y is not None and 'positivo' in set(y) and 'negativo' in set(y):
            cv = CountVectorizer(analyzer='word', min_df=3, binary=True).fit(X)
            C = cv.transform(X)
            ypos = (y == 'positivo')
            yneg = (y == 'negativo')
            npos = np.asarray(C[ypos].sum(axis=0)).ravel() + self.alpha
            nneg = np.asarray(C[yneg].sum(axis=0)).ravel() + self.alpha
            r = np.log(npos / nneg)
            vocab = cv.get_feature_names_out()
            order = np.argsort(r)
            self.pos_words_ = set(vocab[order[-self.k:]][r[order[-self.k:]] > 0].tolist())
            self.neg_words_ = set(vocab[order[:self.k]][r[order[:self.k]] < 0].tolist())
        else:
            self.pos_words_ = set()
            self.neg_words_ = set()
        return self

    def transform(self, X):
        out = []
        for text in X:
            t = clean(text)
            words = re.findall(r'\b\w+\b', t)
            pos = sum(1 for w in words if w in self.pos_words_)
            neg = sum(1 for w in words if w in self.neg_words_)
            negm = sum(1 for w in words if w in NEGATORS)
            contrast = len(CONTRAST.findall(t))
            n_clauses = max(1, len([s for s in re.split(r'[.!?;]+', t) if s.strip()]))
            mixed = float(min(pos, neg))
            out.append([pos, neg, negm, contrast, n_clauses, mixed])
        return np.asarray(out, dtype=float)


class CombinedFeatures(BaseEstimator, TransformerMixin):
    """007 dispersa + rasgos densos de sentimiento (escalados), como una matriz dispersa."""

    def __init__(self, k=150, alpha=1.0, use_lexicon=True):
        self.k = k
        self.alpha = alpha
        self.use_lexicon = use_lexicon

    def fit(self, X, y=None):
        from src.features_007b import TextFeatures007b
        self.sparse_ = TextFeatures007b(thirds=True, min_df=3, max_features=160000).fit(X)
        self.dense_ = DenseSentimentFeatures(k=self.k, alpha=self.alpha,
                                             use_lexicon=self.use_lexicon).fit(X, y)
        self.scaler_ = StandardScaler().fit(self.dense_.transform(X))
        return self

    def transform(self, X):
        S = self.sparse_.transform(X)
        D = self.scaler_.transform(self.dense_.transform(X))
        return sp.hstack([S, sp.csr_matrix(D)], format='csr')
