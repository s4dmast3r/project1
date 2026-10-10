"""Ronda 014: interacciones no lineales (polinómico grado 2) sobre L1 seleccionada.

Primera prueba de interacciones entre características: hasta ahora solo hubo modelos
lineales. Hipótesis (idea 1 del usuario): una palabra puede valer distinto según qué
otras señales aparezcan. Se estudia con un SVM lineal sobre características
polinómicas de grado 2 (interaction_only) construidas SOLO sobre las k características
más discriminativas seleccionadas por L1 dentro de cada entrenamiento.

Candidatas:
- ref011 : referencia 011 (LinearSVC L1 C=2 + descuento neutral 0.2).
- poly_300 : L1 selecciona top-300 rasgos -> polinomio grado 2 -> LinearSVC L1 C=1.
- poly_500 : igual con top-500 rasgos.

La regla de decisión final aplica el mismo descuento neutral 0.2 a todas.

Protocolo pre-registrado: semillas NUEVAS (16007, 17011, 18013) + puerta histórica 137.
Métrica primaria accuracy; F1 macro control. Reemplaza 011 solo si cumple las puertas.
"""
from src.experimento import normalized, sha, dump, LABELS
from src.features_007b import TextFeatures007b
from pathlib import Path
import time, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

FRESH_SEEDS = [16007, 17011, 18013]
HISTORICAL_SEED = 137
D = 0.2  # descuento neutral de la 011
REFERENCE_137_ACC = 0.89888  # (público 011; la comparación 137 usa accuracy OOF real)


class L1Selector(BaseEstimator, TransformerMixin):
    """Selecciona las k características con mayor |coeficiente| de un SVM L1."""

    def __init__(self, k=300, C=2.0):
        self.k = k
        self.C = C

    def fit(self, X, y):
        m = LinearSVC(penalty='l1', dual=False, C=self.C, max_iter=20000, random_state=42).fit(X, y)
        assert np.max(m.n_iter_) < m.max_iter
        imp = np.abs(m.coef_).max(axis=0)
        self.idx_ = np.argsort(-imp)[:self.k]
        return self

    def transform(self, X):
        return X[:, self.idx_]


def make_model(name):
    feats = TextFeatures007b(thirds=True, min_df=3, max_features=160000)
    if name == 'ref011':
        return Pipeline([('features', feats),
                         ('clf', LinearSVC(penalty='l1', dual=False, C=2.0, max_iter=20000, random_state=42))])
    if name in ('poly_300', 'poly_500'):
        k = 300 if name == 'poly_300' else 500
        return Pipeline([('features', feats),
                         ('select', L1Selector(k=k, C=2.0)),
                         ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                         ('clf', LinearSVC(penalty='l1', dual=False, C=1.0, max_iter=20000, random_state=42))])
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


def evaluate(X, y, g, name, seed):
    pred = np.empty(len(y), dtype=object)
    val_acc = []
    train_acc = []
    for k, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g)):
        assert not set(g.iloc[a]) & set(g.iloc[b])
        model = fit(make_model(name), X.iloc[a], y.iloc[a])
        S = model.decision_function(X.iloc[b])
        S2 = S.copy(); S2[:, 0] += D; S2[:, 2] += D
        pred[b] = np.array(LABELS)[S2.argmax(axis=1)]
        S_tr = model.decision_function(X.iloc[a])
        S_tr2 = S_tr.copy(); S_tr2[:, 0] += D; S_tr2[:, 2] += D
        train_acc.append(acc(y.iloc[a], np.array(LABELS)[S_tr2.argmax(axis=1)]))
        val_acc.append(acc(y.iloc[b], pred[b]))
    return {'name': name, 'seed': seed, 'oof_acc': acc(y, pred), 'oof_f1': f1m(y, pred),
            'fold_acc': val_acc, 'fold_train': train_acc,
            'gap': float(np.mean([t - v for t, v in zip(train_acc, val_acc)])),
            'pred': pred}


def run():
    started = time.perf_counter()
    if Path('reports/registro_014.json').exists():
        raise FileExistsError('Conservar resultados 014 antes de repetir.')
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    parts = pd.read_csv('reports/particiones.csv', dtype={'id': str})
    d = parts.partition.eq('development')
    gall = normalized(train.text)
    assert not set(gall[d]) & set(gall[~d])
    X = train.text.fillna('')[d].reset_index(drop=True)
    y = train.label[d].reset_index(drop=True)
    g = gall[d].reset_index(drop=True)

    NAMES = ['ref011', 'poly_300', 'poly_500']
    plan = {
        'candidates': NAMES,
        'fresh_seeds': FRESH_SEEDS,
        'historical_seed': HISTORICAL_SEED,
        'metric': 'accuracy primaria; F1 macro control; brecha diagnóstico',
        'gates': ['ganar >=7/9 folds accuracy vs ref011',
                  'delta medio accuracy >= 0.001',
                  'delta F1 medio >= 0',
                  'aumento brecha accuracy media <= 0.005',
                  'no perder accuracy ni F1 en 137'],
        'idea': 'interacciones polinómicas grado 2 (interaction_only) sobre las k rasgos L1 más discriminativos; selección dentro de cada fold.',
        'limitations': 'Las semillas nuevas no crean datos vírgenes; selección acumulada sobre desarrollo. Interacciones pueden sobreajustar.',
        'reference_public011': 0.89888,
    }
    dump('reports/plan_014.json', plan)

    seed_rows = []
    results = {}
    for seed in FRESH_SEEDS:
        ref = evaluate(X, y, g, 'ref011', seed)
        results['ref011', seed] = ref
        for name in NAMES[1:]:
            cand = evaluate(X, y, g, name, seed)
            results[name, seed] = cand
            deltas = np.array(cand['fold_acc']) - np.array(ref['fold_acc'])
            row = {'name': name, 'seed': seed, 'ref011_acc': ref['oof_acc'],
                   'cand_acc': cand['oof_acc'], 'acc_delta': cand['oof_acc'] - ref['oof_acc'],
                   'fold_wins': int((deltas > 0).sum()), 'gap_ref': ref['gap'],
                   'gap_cand': cand['gap'], 'gap_delta': cand['gap'] - ref['gap'],
                   'f1_ref': ref['oof_f1'], 'f1_cand': cand['oof_f1'],
                   'fold_deltas': np.round(deltas, 5).tolist()}
            seed_rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(seed_rows)
    agg = df.groupby('name').agg(acc_delta=('acc_delta', 'mean'),
                                 fold_wins=('fold_wins', 'sum'),
                                 gap_delta=('gap_delta', 'mean'),
                                 f1_cand_mean=('f1_cand', 'mean'),
                                 f1_ref_mean=('f1_ref', 'mean')).reset_index()
    agg['f1_delta'] = agg.f1_cand_mean - agg.f1_ref_mean
    checks = []
    eligible = []
    for name in NAMES[1:]:
        r = agg[agg.name == name].iloc[0]
        g1 = int(r.fold_wins) >= 7
        g2 = float(r.acc_delta) >= 0.001
        g3 = float(r.f1_delta) >= 0
        g4 = float(r.gap_delta) <= 0.005
        checks.append({'name': name, 'fold_wins': int(r.fold_wins),
                       'mean_acc_delta': float(r.acc_delta), 'mean_f1_delta': float(r.f1_delta),
                       'mean_gap_delta': float(r.gap_delta), 'passes_fresh': bool(g1 and g2 and g3 and g4)})
        if g1 and g2 and g3 and g4:
            eligible.append(name)
    pd.DataFrame(seed_rows).to_csv('reports/resultados_014.csv', index=False)
    pd.DataFrame(checks).to_csv('reports/criterios_014.csv', index=False)

    hist = {}
    accepted = []
    if eligible:
        ref137 = evaluate(X, y, g, 'ref011', HISTORICAL_SEED)
        for name in eligible:
            cand = evaluate(X, y, g, name, HISTORICAL_SEED)
            hist[name] = {'ref011_137_acc': ref137['oof_acc'], 'cand_137_acc': cand['oof_acc'],
                          'ref011_137_f1': ref137['oof_f1'], 'cand_137_f1': cand['oof_f1'],
                          'passes': cand['oof_acc'] >= ref137['oof_acc'] and cand['oof_f1'] >= ref137['oof_f1']}
            if hist[name]['passes']:
                accepted.append(name)
            print('puerta 137', name, hist[name], flush=True)

    winner = None
    if accepted:
        winner = str(agg[agg.name.isin(accepted)].sort_values('acc_delta', ascending=False).iloc[0]['name'])

    rec = {'plan': plan, 'seed_rows': seed_rows, 'checks': checks, 'historical_137': hist,
           'winner': winner, 'submission_created': False, 'reference_public011': 0.89888}

    if winner:
        model = fit(make_model(winner), train.text.fillna(''), train.label)
        joblib.dump(model, 'models/modelo_012.joblib')
        ev = pd.read_csv('data/eval.csv', dtype={'id': str})
        sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
        S = model.decision_function(ev.text.fillna(''))
        S2 = S.copy(); S2[:, 0] += D; S2[:, 2] += D
        p = np.array(LABELS)[S2.argmax(axis=1)]
        sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(p, index=ev.id))})
        assert len(sub) == len(ev) == len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
        sub.to_csv('submissions/submission_012.csv', index=False)
        pd.testing.assert_frame_equal(sub, pd.read_csv('submissions/submission_012.csv', dtype={'id': str}))
        pred_diag = evaluate(X, y, g, winner, FRESH_SEEDS[-1])['pred']
        per_class = classification_report(y, pred_diag, labels=LABELS, output_dict=True, zero_division=0)
        per_class.pop('accuracy', None)
        rec.update(submission_created=True,
                   artifacts_sha256={pth: sha(pth) for pth in ['models/modelo_012.joblib', 'submissions/submission_012.csv']},
                   per_class=per_class, confusion=confusion_matrix(y, pred_diag, labels=LABELS).tolist(),
                   csv_checks_passed=True,
                   decision=f'{winner} aprobada; submission_012.csv generado.')
    else:
        rec['decision'] = 'Conservar011: ninguna candidata cumple las puertas pre-registradas. No generar submission_012.'
    rec['elapsed_seconds'] = time.perf_counter() - started
    rec['source_sha256'] = {p: sha(p) for p in ['src/experimento_014.py', 'src/features_007b.py', 'reports/particiones.csv']}
    dump('reports/registro_014.json', rec)
    print(json.dumps({'winner': winner, 'submission_created': rec['submission_created'],
                      'decision': rec['decision'], 'seconds': rec['elapsed_seconds']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
