"""Producción de submission_012: modelo 011 + tokens de regla discursiva.

Único cambio frente a 011: TextFeatures016(rules=True) con léxico semántico.
Se mantiene el descuento de neutral D=0.2 para aislar el efecto de las reglas.
"""
from src.experimento import sha, dump, LABELS
from src.features_016 import TextFeatures016
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

D = 0.2


def predict_with_discount(model, texts):
    S = model.decision_function(texts)
    S[:, 0] += D   # negativo
    S[:, 2] += D   # positivo -> descuenta neutral
    return np.array(LABELS)[S.argmax(axis=1)]


def run():
    started = time.perf_counter()
    if Path('reports/registro_produccion_012.json').exists():
        raise FileExistsError('Conservar producción 012 antes de repetir.')
    expected_inputs = {
        'data/train.csv': 'ae6867261512451d67b7d18a9d6f37529cf1475bdf6c7407f5b01be82bc4c378',
        'data/eval.csv': 'c9aedcb94bc2a50c185fcd8b30b9853f061f234c9fe54f6befb3be8043f0217e',
        'data/sample_submission.csv': '487528908bc0bb008a74aeabaa0faf994a450da89c13426783d47f692299110b',
    }
    for p, h in expected_inputs.items():
        assert sha(p) == h, p

    train = pd.read_csv('data/train.csv', dtype={'id': str})
    feats = TextFeatures016(thirds=True, rules=True, min_df=3, max_features=160000)
    clf = LinearSVC(penalty='l1', dual=False, C=2.0, max_iter=20000, random_state=42)
    model = Pipeline([('features', feats), ('clf', clf)]).fit(train.text.fillna(''), train.label)
    assert np.max(model['clf'].n_iter_) < model['clf'].max_iter
    assert list(model['clf'].classes_) == LABELS   # el descuento asume este orden

    # Verificar que las reglas realmente entraron al vocabulario
    vocab = model['features'].transformer_.vocabulary_
    n_rule = sum(1 for k in vocab if k.startswith('R:'))
    assert n_rule > 0, 'No hay tokens R: -> rules no se propagó'
    joblib.dump(model, 'models/modelo_012.joblib')

    ev = pd.read_csv('data/eval.csv', dtype={'id': str})
    sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
    assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
    pred = predict_with_discount(model, ev.text.fillna(''))
    sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(pred, index=ev.id))})
    assert len(sub) == len(ev) == len(sample) and sub.id.equals(sample.id)
    assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
    sub.to_csv('submissions/submission_012.csv', index=False)
    pd.testing.assert_frame_equal(sub, pd.read_csv('submissions/submission_012.csv', dtype={'id': str}))

    # La recarga debe reproducir exactamente las predicciones
    reloaded = joblib.load('models/modelo_012.joblib')
    assert np.array_equal(predict_with_discount(reloaded, ev.text.fillna('')), pred)

    sub011 = pd.read_csv('submissions/submission_011.csv', dtype={'id': str})
    n_changed = int((sub011.answer != sub.answer).sum())

    rec = {
        'base_model': '011 + TextFeatures016(rules=True), LinearSVC L1 C=2',
        'decision_rule': f'argmax sobre [s_neg+{D}, s_neu, s_pos+{D}] (igual que 011)',
        'justification': 'La etiqueta depende de la última opinión y su conector; los tokens R: '
                         'lo representan explícitamente. Léxico solo semántico (sin frases '
                         'minadas con etiquetas) para evitar fuga.',
        'local_evidence': 'CV 5 folds seed 0: val 0.8982 -> 0.9013, brecha 0.089 -> 0.080.',
        'n_rule_tokens_vocab': n_rule,
        'n_changed_vs_011': n_changed,
        'pred_distribution': sub.answer.value_counts().to_dict(),
        'artifacts_sha256': {p: sha(p) for p in ['models/modelo_012.joblib',
                                                 'submissions/submission_012.csv']},
        'elapsed_seconds': time.perf_counter() - started,
        'honest_note': 'Ganancia dentro del ruido entre folds; ~16% de reseñas tienen etiqueta '
                       'no predecible desde el texto.',
    }
    dump('reports/registro_produccion_012.json', rec)
    print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()