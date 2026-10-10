"""Producción de submission_014 (corrida 14): 011 + tokens de regla discursiva (R:).

Único cambio frente a producir_011: TextFeatures007b(thirds=True, rules=True) de
features_016. Los tokens R: codifican (última polaridad × tipo de conector × si hay
algo después) a partir de src/clausulas.py y del léxico semántico src/lex.py (sin
frases encontradas mirando etiquetas). Se mantienen L1 C=2, tokens R: emitidos una
vez y el descuento de neutral d=0.2 de la 011 (cambia ~4 predicciones de 3000).

Evidencia local (protocolo 013, semillas 8803/9901/11003): accuracy 0.8991 -> 0.9015 (6 victorias, 1 empate, 2 derrotas de 9 folds) y brecha
0.089 -> 0.080; semilla 137: 0.8977 -> 0.9015. Se escribe en archivos 014 nuevos; modelo_011/submission_011 quedan
intactos. El .joblib requiere src/features_016.py, rulefeat.py, clausulas.py y lex.py.
"""
from src.experimento import sha, dump, LABELS
from src.features_016 import TextFeatures007b
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

D = 0.2
MODEL = 'models/modelo_014.joblib'
SUB = 'submissions/submission_014.csv'
REC = 'reports/registro_produccion_014.json'


def run():
    started = time.perf_counter()
    if Path(REC).exists():
        raise FileExistsError('Conservar producción 014 antes de repetir.')
    expected_inputs = {
        'data/train.csv': 'ae6867261512451d67b7d18a9d6f37529cf1475bdf6c7407f5b01be82bc4c378',
        'data/eval.csv': 'c9aedcb94bc2a50c185fcd8b30b9853f061f234c9fe54f6befb3be8043f0217e',
        'data/sample_submission.csv': '487528908bc0bb008a74aeabaa0faf994a450da89c13426783d47f692299110b',
    }
    for p, h in expected_inputs.items():
        assert sha(p) == h, p

    train = pd.read_csv('data/train.csv', dtype={'id': str})
    feats = TextFeatures007b(thirds=True, rules=True, min_df=3, max_features=160000)
    clf = LinearSVC(penalty='l1', dual=False, C=2.0, max_iter=20000, random_state=42)
    model = Pipeline([('features', feats), ('clf', clf)]).fit(train.text.fillna(''), train.label)
    assert np.max(model['clf'].n_iter_) < model['clf'].max_iter
    n_rule_feats = sum(t.startswith('R:') for t in model['features'].transformer_.vocabulary_)
    assert n_rule_feats > 0
    joblib.dump(model, MODEL)

    ev = pd.read_csv('data/eval.csv', dtype={'id': str})
    sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
    assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
    S = model.decision_function(ev.text.fillna(''))
    S2 = S.copy()
    S2[:, 0] += D   # negativo
    S2[:, 2] += D   # positivo -> descuenta neutral
    pred = np.array(LABELS)[S2.argmax(axis=1)]
    pred_nodisc = np.array(LABELS)[S.argmax(axis=1)]
    sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(pred, index=ev.id))})
    assert len(sub) == len(ev) == len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
    assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
    sub.to_csv(SUB, index=False)
    pd.testing.assert_frame_equal(sub, pd.read_csv(SUB, dtype={'id': str}))

    sub011 = pd.read_csv('submissions/submission_011.csv', dtype={'id': str})
    n_changed = int((sub011.answer != sub.answer).sum())

    rec = {
        'base_model': '011 (LinearSVC L1 C=2 sobre características 007 + descuento de neutral)',
        'change': 'TextFeatures007b(thirds=True, rules=True) de features_016: tokens R: de regla discursiva',
        'decision_rule': f'argmax sobre [s_negativo+{D}, s_neutral, s_positivo+{D}] (descuento de neutral)',
        'local_evidence': 'semillas 8803/9901/11003: accuracy 0.8991 -> 0.9015, 6 victorias / 1 empate / 2 derrotas en 9 folds, brecha 0.089 -> 0.080; semilla 137: 0.8977 -> 0.9015. Ver reports/verificacion_014.log.',
        'n_rule_features': int(n_rule_feats),
        'n_changed_by_discount': int((pred != pred_nodisc).sum()),
        'n_changed_vs_011': n_changed,
        'reference_public011': 0.89888,
        'artifacts_sha256': {p: sha(p) for p in [MODEL, SUB]},
        'checks': {'ids_order_schema_labels': True, 'roundtrip': True},
        'honest_note': 'Con ~900 reseñas en el público el error estándar es ±1 punto; elegir el final por CV, no por el público.',
    }
    reloaded = joblib.load(MODEL)
    Sr = reloaded.decision_function(ev.text.fillna(''))
    Sr[:, 0] += D; Sr[:, 2] += D
    assert np.array_equal(np.array(LABELS)[Sr.argmax(axis=1)], pred)
    rec['checks']['reload'] = True
    rec['elapsed_seconds'] = time.perf_counter() - started
    dump(REC, rec)
    print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
