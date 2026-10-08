"""Características clásicas de orden y contexto; sin etiquetas ni recursos externos."""
import re, unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

def clean(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(text).lower()) if not unicodedata.combining(c))

def positioned(text, contextual=False):
    text=clean(text)
    sentences=[s.strip() for s in re.split(r'[.!?;]+',text) if s.strip()]
    words=re.findall(r'\b\w+\b',text)
    contexts=['plain']
    if contextual and sentences:
        # Interacción con la apertura del cierre, sin decidir su polaridad.
        tail=re.findall(r'\b\w+\b',sentences[-1])
        contexts+=['tail_'+'_'.join(tail[:2]),'tail_'+'_'.join(tail[:3])]
    for n in (1,2,3):
        for i in range(len(words)-n+1):
            gram='_'.join(words[i:i+n])
            yield 'global:'+gram
            for ctx in contexts:
                yield ctx+':third'+str(min(2,3*i//max(1,len(words))))+':'+gram
    for i,s in enumerate(sentences):
        tokens=re.findall(r'\b\w+\b',s)
        positions={'start'+str(min(i,2)), 'end'+str(min(len(sentences)-1-i,2))}
        for n in (1,2):
            for j in range(len(tokens)-n+1):
                gram='_'.join(tokens[j:j+n])
                for pos in positions:
                    for ctx in contexts:
                        yield ctx+':'+pos+':'+gram

class OrderedAnalyzer:
    def __init__(self, contextual=False): self.contextual=contextual
    def __call__(self,text): return positioned(text,self.contextual)

class TextFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,mode='word_char'):
        self.mode=mode
    def fit(self,X,y=None):
        word=TfidfVectorizer(strip_accents='unicode',ngram_range=(1,3),min_df=3,sublinear_tf=True,max_features=70000)
        char=TfidfVectorizer(strip_accents='unicode',analyzer='char',ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=70000)
        if self.mode=='word_char':
            self.transformer_=FeatureUnion([('words',word),('chars',char)],transformer_weights={'words':1.,'chars':0.7})
        else:
            ordered=TfidfVectorizer(analyzer=DiscourseAnalyzer() if self.mode=='discourse' else OrderedAnalyzer(self.mode=='context'),min_df=3,sublinear_tf=True,max_features=160000)
            self.transformer_=FeatureUnion([('ordered',ordered),('chars',char)],transformer_weights={'ordered':1.,'chars':0.5})
        self.transformer_.fit(X)
        return self
    def transform(self,X): return self.transformer_.transform(X)

# Marcadores lingüísticos, no diccionarios de polaridad ni reglas de clasificación.
EPILOGUE = re.compile(r'^(?:en fin|cada quien|para gustos|queda a criterio|no me queda claro|ni tan bueno|sopesalo|para quien|eso es todo)\b')
CONNECTOR = re.compile(r'\b(?:eso si|aun asi|sin embargo|no obstante|con todo|por otro lado|por cierto|de paso|aunque|pero|al final(?: de cuentas)?|y (?=el |la |los |las ))\b')

def discourse(text):
    text=clean(text)
    sentences=[s.strip(' ,:') for s in re.split(r'[.!?;]+',text) if s.strip(' ,:')]
    epilogue=bool(sentences and EPILOGUE.search(sentences[-1]))
    # El cierre se conserva en el canal global; los demás canales modelan cláusulas.
    raw=re.findall(r'\b\w+\b',text)
    for n in (1,2):
        for j in range(len(raw)-n+1): yield 'global:'+'_'.join(raw[j:j+n])
    body=sentences[:-1] if epilogue else sentences
    chunks=[]
    for sentence in body:
        chunks.extend(x.strip(' ,:') for x in CONNECTOR.split(sentence) if x.strip(' ,:'))
    for i,chunk in enumerate(chunks):
        tokens=re.findall(r'\b\w+\b',chunk)
        pos=['start'+str(min(i,3)),'end'+str(min(len(chunks)-i-1,3))]
        for n in (1,2,3):
            for j in range(len(tokens)-n+1):
                term='_'.join(tokens[j:j+n])
                for p in pos:
                    yield p+':'+term
                    yield 'epilogue'+str(epilogue)+':'+p+':'+term

class DiscourseAnalyzer:
    def __call__(self,text): return discourse(text)
