"""Características de la versión 004: réplica parametrizada del canal discursivo 003.

Reproduce exactamente `features_003.discourse` cuando se usan los valores por
defecto (keep_epilogue=False, use_connector=True, max_ngram=3, min_df=3,
char_weight=0.5, max_features=160000). Los parámetros adicionales permiten
ablaciones acotadas sin tocar el módulo 003 (necesario para cargar modelo_003).

Ningún parámetro ni analizador consulta etiquetas ni eval; solo texto.
"""
import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

EPILOGUE = re.compile(r'^(?:en fin|cada quien|para gustos|queda a criterio|no me queda claro|ni tan bueno|sopesalo|para quien|eso es todo)\b')
CONNECTOR = re.compile(r'\b(?:eso si|aun asi|sin embargo|no obstante|con todo|por otro lado|por cierto|de paso|aunque|pero|al final(?: de cuentas)?|y (?=el |la |los |las ))\b')


def clean(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(text).lower()) if not unicodedata.combining(c))


def discourse_tokens(text, keep_epilogue=False, use_connector=True, max_ngram=3):
    """Generador idéntico a features_003.discourse salvo por los conmutadores."""
    text = clean(text)
    sentences = [s.strip(' ,:') for s in re.split(r'[.!?;]+', text) if s.strip(' ,:')]
    epilogue = bool(sentences and EPILOGUE.search(sentences[-1]))
    raw = re.findall(r'\b\w+\b', text)
    for n in (1, 2):
        for j in range(len(raw) - n + 1):
            yield 'global:' + '_'.join(raw[j:j + n])
    # En 003 el cierre se excluye del cuerpo de los canales posicionales.
    body = sentences[:-1] if (epilogue and not keep_epilogue) else sentences
    if use_connector:
        chunks = []
        for sentence in body:
            chunks.extend(x.strip(' ,:') for x in CONNECTOR.split(sentence) if x.strip(' ,:'))
    else:
        chunks = body
    for i, chunk in enumerate(chunks):
        tokens = re.findall(r'\b\w+\b', chunk)
        pos = ['start' + str(min(i, 3)), 'end' + str(min(len(chunks) - i - 1, 3))]
        for n in range(1, max_ngram + 1):
            for j in range(len(tokens) - n + 1):
                term = '_'.join(tokens[j:j + n])
                for p in pos:
                    yield p + ':' + term
                    yield 'epilogue' + str(epilogue) + ':' + p + ':' + term


class DiscourseAnalyzer:
    def __init__(self, keep_epilogue=False, use_connector=True, max_ngram=3):
        self.keep_epilogue = keep_epilogue
        self.use_connector = use_connector
        self.max_ngram = max_ngram

    def __call__(self, text):
        return discourse_tokens(text, self.keep_epilogue, self.use_connector, self.max_ngram)


class TextFeatures(BaseEstimator, TransformerMixin):
    """FeatureUnion con canal discursivo y canal de caracteres, como en 003."""

    def __init__(self, keep_epilogue=False, use_connector=True, max_ngram=3,
                 min_df=3, char_weight=0.5, max_features=160000):
        self.keep_epilogue = keep_epilogue
        self.use_connector = use_connector
        self.max_ngram = max_ngram
        self.min_df = min_df
        self.char_weight = char_weight
        self.max_features = max_features

    def fit(self, X, y=None):
        ordered = TfidfVectorizer(
            analyzer=DiscourseAnalyzer(self.keep_epilogue, self.use_connector, self.max_ngram),
            min_df=self.min_df, sublinear_tf=True, max_features=self.max_features)
        parts = [('ordered', ordered)]
        weights = {'ordered': 1.0}
        if self.char_weight > 0:
            char = TfidfVectorizer(strip_accents='unicode', analyzer='char',
                                   ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                   max_features=70000)
            parts.append(('chars', char))
            weights['chars'] = self.char_weight
        self.transformer_ = FeatureUnion(parts, transformer_weights=weights)
        self.transformer_.fit(X)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)
