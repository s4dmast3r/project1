"""Ronda 007b: variantes ESTRUCTURALES de la representación 005.

Con todos los parámetros por defecto reproduce exactamente el analizador usado
por la versión 005 (features_004.discourse_tokens con max_ngram=1, canal global
1-2 gramos, sin canal de caracteres). Cada parámetro activa UN cambio estructural
del espacio de características, ninguno probado por las rondas 006/007:

- global_max_ngram: profundidad de n-gramas del canal GLOBAL (005 usa 2). Probar
  3 permite capturar frases completas de sentimiento ("vale_la_pena",
  "no_lo_recomiendo") sin tocar el canal posicional.
- mid_bucket: etiqueta 'mid' para palabras que quedan lejos de ambos extremos de
  su bloque (>=3 desde el inicio y desde el final). El bucketing start/end actual
  hace ambiguas las palabras centrales de bloques largos.
- edge_bigrams: bigramas posicionales SOLO en los dos bordes de cada bloque
  ("no_me", "me_encanto", "pero_la"). Compromiso entre 004 (bigramas posicionales
  en todas partes: peor) y 005 (ninguno).
- thirds: etiqueta de tercio del texto completo (first/mid/last) para cada palabra.

Ningún parámetro consulta etiquetas ni eval; solo texto. Las hipótesis de la ronda
y su contraste previo quedan en reports/plan_007b.json.
"""
import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from src.rulefeat import rule_tokens

EPILOGUE = re.compile(r'^(?:en fin|cada quien|para gustos|queda a criterio|no me queda claro|ni tan bueno|sopesalo|para quien|eso es todo)\b')
CONNECTOR = re.compile(r'\b(?:eso si|aun asi|sin embargo|no obstante|con todo|por otro lado|por cierto|de paso|aunque|pero|al final(?: de cuentas)?|y (?=el |la |los |las ))\b')


def clean(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(text).lower()) if not unicodedata.combining(c))


def discourse_tokens(text, global_max_ngram=2, mid_bucket=False, edge_bigrams=False, thirds=False, rules=False):
    """Replica features_004.discourse_tokens(max_ngram=1) en sus valores por defecto."""
    text = clean(text)
    sentences = [s.strip(' ,:') for s in re.split(r'[.!?;]+', text) if s.strip(' ,:')]
    epilogue = bool(sentences and EPILOGUE.search(sentences[-1]))
    raw = re.findall(r'\b\w+\b', text)
    for n in range(1, global_max_ngram + 1):
        for j in range(len(raw) - n + 1):
            yield 'global:' + '_'.join(raw[j:j + n])
    if thirds:
        for i, w in enumerate(raw):
            yield 'third' + str(min(2, 3 * i // max(1, len(raw)))) + ':' + w
    body = sentences[:-1] if epilogue else sentences
    chunks = []
    for sentence in body:
        chunks.extend(x.strip(' ,:') for x in CONNECTOR.split(sentence) if x.strip(' ,:'))
    for i, chunk in enumerate(chunks):
        tokens = re.findall(r'\b\w+\b', chunk)
        pos = ['start' + str(min(i, 3)), 'end' + str(min(len(chunks) - i - 1, 3))]
        for j in range(len(tokens)):
            term = tokens[j]
            for p in pos:
                yield p + ':' + term
                yield 'epilogue' + str(epilogue) + ':' + p + ':' + term
            if mid_bucket and len(tokens) >= 7 and 3 <= j <= len(tokens) - 4:
                yield 'mid:' + term
                yield 'epilogue' + str(epilogue) + ':mid:' + term
        if edge_bigrams and len(tokens) >= 2:
            for j in (0, len(tokens) - 2):
                gram = tokens[j] + '_' + tokens[j + 1]
                for p in pos:
                    yield p + ':' + gram
                    yield 'epilogue' + str(epilogue) + ':' + p + ':' + gram

    if rules:
        yield from rule_tokens(text)


class DiscourseAnalyzer:
    def __init__(self, global_max_ngram=2, mid_bucket=False, edge_bigrams=False, thirds=False, rules=False):
        self.global_max_ngram = global_max_ngram
        self.mid_bucket = mid_bucket
        self.edge_bigrams = edge_bigrams
        self.thirds = thirds
        self.rules = rules

    def __call__(self, text):
        return discourse_tokens(text, self.global_max_ngram, self.mid_bucket,
                                self.edge_bigrams, self.thirds, self.rules)


class TextFeatures016(BaseEstimator, TransformerMixin):
    """Representación 005 parametrizada (canal discursivo único, sin caracteres)."""

    def __init__(self, global_max_ngram=2, mid_bucket=False, edge_bigrams=False, thirds=False,
                 min_df=3, max_features=160000, rules=False):
        self.global_max_ngram = global_max_ngram
        self.mid_bucket = mid_bucket
        self.edge_bigrams = edge_bigrams
        self.thirds = thirds
        self.min_df = min_df
        self.max_features = max_features
        self.rules = rules

    def fit(self, X, y=None):
        ordered = TfidfVectorizer(
            analyzer=DiscourseAnalyzer(self.global_max_ngram, self.mid_bucket,
                                       self.edge_bigrams, self.thirds, self.rules),
            min_df=self.min_df, sublinear_tf=True, max_features=self.max_features)
        ordered.fit(X)
        self.transformer_ = ordered
        return self

    def transform(self, X):
        return self.transformer_.transform(X)
