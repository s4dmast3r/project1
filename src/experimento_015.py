"""Ronda 015 (diagnóstico): aislar selección vs interacciones, con C fijo en 2.

Candidatas:
- ref011        : representación completa + LinearSVC L1 C=2.
- sel500        : top-500 rasgos (L1) sin productos + L1 C=2.
- sel500_poly   : top-500 + productos grado 2 (bloque L2-normalizado) + L1 C=2.
- full_polyblock: representación completa + bloque de interacciones (top-500,
                  L2-normalizado) + L1 C=2.

Se reportan accuracy (y F1) con argmax directo y con el descuento neutral 0.2, para
separar el efecto de la regla de decisión del efecto de las características. No es
una ronda de selección/submission: es un diagnóstico para interpretar la 014.
"""
from src.experimento import normalized, sha, dump, LABELS
from src.features_007b import TextFeatures007b
from src.features_015 import L1Selector, PolyBlock, FullPlusPoly
from pathlib import Path
import time, warnings, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.exceptions import ConvergenceWarning

FRESH_SEEDS = [19001, 20011, 21013]
D = 0.2


def make_model(name):
    feats = TextFeatures007b(thirds=True, min_df=3, max_features=160000)
    clf = LinearSVC(penalty='l1', dual=False, C=2.0, max_iter=20000, random_state=42)
    if name == 'ref011':
        return Pipeline([('features', feats), ('clf', clf)])
    if name == 'sel500':
        return Pipeline([('features', feats), ('select', L1Selector(k=500, C=2.0)), ('clf', clf)])
    if name == 'sel500_poly':
        return Pipeline([('features', feats), ('poly', PolyBlock(k=500, C=2.0)), ('clf', clf)])
    if name == 'full_polyblock':
        return Pipeline([('features', feats), ('plus', FullPlusPoly(k=500, C=2.0)), ('clf', clf)])
    raise ValueError(name)


def fit(pipe, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipe.fit(X, y)
    assert np.max(pipe['clf'].n_iter_) < pipe['clf'].max_iter
    return pipe


def acc(y, p):
    return float(accuracy_score(y, p))


def f1m(y, p):
    return float(f1_score(y, p, average='macro', labels=LABELS, zero_division=0))


def predict_with(S, d):
    S2 = S.copy()
    if d:
        S2[:, 0] += d; S2[:, 2] += d
    return np.array(LABELS)[S2.argmax(axis=1)]


def evaluate(X, y, g, name, seed):
    pred_raw = np.empty(len(y), dtype=object)
    pred_d = np.empty(len(y), dtype=object)
    val_acc_raw = []
    for k, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g)):
        assert not set(g.iloc[a]) & set(g.iloc[b])
        model = fit(make_model(name), X.iloc[a], y.iloc[a])
        S = model.decision_function(X.iloc[b])
        pred_raw[b] = predict_with(S, 0)
        pred_d[b] = predict_with(S, D)
        val_acc_raw.append(acc(y.iloc[b], pred_raw[b]))
    return {'name': name, 'seed': seed,
            'acc_raw': acc(y, pred_raw), 'acc_d02': acc(y, pred_d),
            'f1_raw': f1m(y, pred_raw), 'f1_d02': f1m(y, pred_d),
            'fold_acc_raw': val_acc_raw}


def run():
    started = time.perf_counter()
    if Path('reports/registro_015.json').exists():
        raise FileExistsError('Conservar resultados 015 antes de repetir.')
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    parts = pd.read_csv('reports/particiones.csv', dtype={'id': str})
    d = parts.partition.eq('development')
    gall = normalized(train.text)
    assert not set(gall[d]) & set(gall[~d])
    X = train.text.fillna('')[d].reset_index(drop=True)
    y = train.label[d].reset_index(drop=True)
    g = gall[d].reset_index(drop=True)

    NAMES = ['ref011', 'sel500', 'sel500_poly', 'full_polyblock']
    rows = []
    for seed in FRESH_SEEDS:
        for name in NAMES:
            r = evaluate(X, y, g, name, seed)
            rows.append(r)
            print(r, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('reports/resultados_015.csv', index=False)
    summary = df.groupby('name').agg(
        acc_raw=('acc_raw', 'mean'), acc_d02=('acc_d02', 'mean'),
        f1_raw=('f1_raw', 'mean'), f1_d02=('f1_d02', 'mean')).reset_index()
    print('\n=== Resumen (media de 3 semillas) ===', flush=True)
    print(summary.to_string(index=False), flush=True)

    rec = {
        'diagnostic': True,
        'purpose': 'Aislar el efecto de recortar a 500 rasgos, de añadir interacciones y de conservar la representación completa, con C fijo en 2 y bloque polinómico L2-normalizado.',
        'candidates': NAMES,
        'seeds': FRESH_SEEDS,
        'neutral_discount': D,
        'summary': summary.to_dict('records'),
        'seed_rows': rows,
        'elapsed_seconds': time.perf_counter() - started,
        'source_sha256': {p: sha(p) for p in ['src/experimento_015.py', 'src/features_015.py', 'src/features_007b.py', 'reports/particiones.csv']},
    }
    dump('reports/registro_015.json', rec)
    print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
