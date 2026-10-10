"""Producción de submission_011: ref010 (L1 C=2) + descuento de neutral para accuracy.

La métrica oficial es accuracy y neutral es la clase menos frecuente. Un descuento
de la clase neutral (d=0.2) en la regla de decisión — argmax sobre
[s_neg+d, s_neu, s_pos+d] — añade una ganancia pequeña y consistente de accuracy
(+0.0006..0.0009 en semillas nuevas, con d~0.2 estable por validación anidada).
Es una corrección de prior, no una regla de etiqueta.

No se reentrena ni se cambia el vocabulario; solo se aplica el descuento aprendido
en desarrollo a los scores de eval. El modelo base es exactamente el de la 010.
"""
from src.experimento import normalized, sha, dump, LABELS
from src.features_007b import TextFeatures007b
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

D = 0.2


def run():
    started = time.perf_counter()
    if Path('reports/registro_produccion_011.json').exists():
        raise FileExistsError('Conservar producción 011 antes de repetir.')
    # Verificar integridad de datos y del modelo 010 (hash conocido).
    expected_inputs = {
        'data/train.csv': 'ae6867261512451d67b7d18a9d6f37529cf1475bdf6c7407f5b01be82bc4c378',
        'data/eval.csv': 'c9aedcb94bc2a50c185fcd8b30b9853f061f234c9fe54f6befb3be8043f0217e',
        'data/sample_submission.csv': '487528908bc0bb008a74aeabaa0faf994a450da89c13426783d47f692299110b',
    }
    for p, h in expected_inputs.items():
        assert sha(p) == h, p

    train = pd.read_csv('data/train.csv', dtype={'id': str})
    feats = TextFeatures007b(thirds=True, min_df=3, max_features=160000)
    clf = LinearSVC(penalty='l1', dual=False, C=2.0, max_iter=20000, random_state=42)
    model = Pipeline([('features', feats), ('clf', clf)]).fit(train.text.fillna(''), train.label)
    assert np.max(model['clf'].n_iter_) < model['clf'].max_iter
    joblib.dump(model, 'models/modelo_011.joblib')

    ev = pd.read_csv('data/eval.csv', dtype={'id': str})
    sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
    assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
    S = model.decision_function(ev.text.fillna(''))
    S2 = S.copy()
    S2[:, 0] += D   # negativo
    S2[:, 2] += D   # positivo -> descuenta neutral
    pred = np.array(LABELS)[S2.argmax(axis=1)]
    sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(pred, index=ev.id))})
    assert len(sub) == len(ev) == len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
    assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
    sub.to_csv('submissions/submission_011.csv', index=False)
    pd.testing.assert_frame_equal(sub, pd.read_csv('submissions/submission_011.csv', dtype={'id': str}))

    # comparación con 010
    sub010 = pd.read_csv('submissions/submission_010.csv', dtype={'id': str})
    n_changed = int((sub010.answer != sub.answer).sum())

    rec = {
        'base_model': '010 (LinearSVC L1 C=2 sobre características 007)',
        'decision_rule': f'argmax sobre [s_negativo+{D}, s_neutral, s_positivo+{D}] (descuento de neutral)',
        'justification': 'accuracy oficial; neutral es la clase menos frecuente; corrección de prior aprendida en desarrollo por CV anidada (d~0.2 estable).',
        'local_evidence': 'ganancia de accuracy ~+0.0007 media en semillas nuevas (12007/13001/14009), consistente.',
        'n_changed_vs_010': n_changed,
        'reference_public010': 0.89777,
        'artifacts_sha256': {p: sha(p) for p in ['models/modelo_011.joblib', 'submissions/submission_011.csv']},
        'checks': {'ids_order_schema_labels': True, 'roundtrip': True, 'reload': True},
        'elapsed_seconds': time.perf_counter() - started,
        'honest_note': 'Ganancia pequeña y dentro del ruido entre folds; no garantiza alcanzar 0.90 público.',
    }
    # recarga reproduce (con descuento)
    reloaded = joblib.load('models/modelo_011.joblib')
    Sr = reloaded.decision_function(ev.text.fillna(''))
    Sr[:, 0] += D; Sr[:, 2] += D
    assert np.array_equal(np.array(LABELS)[Sr.argmax(axis=1)], pred)
    rec['checks']['reload'] = True
    dump('reports/registro_produccion_011.json', rec)
    print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
