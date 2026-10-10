"""Ronda 013: optimizar para ACCURACY (métrica oficial confirmada).

El usuario confirmó que las respuestas se evalúan con accuracy. Re-evaluamos el C de
la 009 (LinearSVC L1) bajo accuracy como métrica primaria.

Candidatas (todas sobre TextFeatures007b(thirds=True), L1, dual=False):
- ref009 : C=1 (referencia exacta 009).
- l1_c15 : C=1.5.
- l1_c2  : C=2.0.
- l1_c3  : C=3.0.

Protocolo pre-registrado: semillas NUEVAS (8803, 9901, 11003) + puerta histórica 137.
Métrica primaria: accuracy; secundarias: F1 macro y brecha. Reemplaza 009 solo si
cumple las puertas; si no, se conserva 009 (público 0.89555). No se bajan umbrales.
"""
from src.experimento import normalized, sha, dump, LABELS
from src.features_007b import TextFeatures007b
from pathlib import Path
import time, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

FRESH_SEEDS = [8803, 9901, 11003]
HISTORICAL_SEED = 137
NAMES = ['ref009', 'l1_c15', 'l1_c2', 'l1_c3']
C_VALUES = {'ref009': 1.0, 'l1_c15': 1.5, 'l1_c2': 2.0, 'l1_c3': 3.0}
REFERENCE_137_ACC = 0.89500  # 009 en 137 bajo accuracy (medido)


def make_model(name):
    feats = TextFeatures007b(thirds=True, min_df=3, max_features=160000)
    clf = LinearSVC(penalty='l1', dual=False, C=C_VALUES[name], max_iter=20000, random_state=42)
    return Pipeline([('features', feats), ('clf', clf)])


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
        pred[b] = model.predict(X.iloc[b])
        train_acc.append(acc(y.iloc[a], model.predict(X.iloc[a])))
        val_acc.append(acc(y.iloc[b], pred[b]))
    return {'name': name, 'seed': seed, 'oof_acc': acc(y, pred),
            'oof_f1': f1m(y, pred), 'fold_acc': val_acc, 'fold_train': train_acc,
            'gap': float(np.mean([t - v for t, v in zip(train_acc, val_acc)])),
            'pred': pred}


def run():
    started = time.perf_counter()
    if Path('reports/registro_013.json').exists():
        raise FileExistsError('Conservar resultados 013 antes de repetir.')
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    parts = pd.read_csv('reports/particiones.csv', dtype={'id': str})
    d = parts.partition.eq('development')
    gall = normalized(train.text)
    assert not set(gall[d]) & set(gall[~d])
    X = train.text.fillna('')[d].reset_index(drop=True)
    y = train.label[d].reset_index(drop=True)
    g = gall[d].reset_index(drop=True)

    plan = {
        'candidates': NAMES,
        'C_values': C_VALUES,
        'fresh_seeds': FRESH_SEEDS,
        'historical_seed': HISTORICAL_SEED,
        'metric': 'accuracy (oficial confirmada); F1 macro y brecha como secundarias',
        'gates': [
            'ganar >=7 de 9 folds emparejados frente a ref009 (3 semillas nuevas)',
            'delta medio de accuracy >= 0.003',
            'no perder accuracy frente a ref009 en semilla histórica 137',
        ],
        'note_gap': 'La brecha se reporta como diagnóstico de sobreajuste; un aumento se evalúa explícitamente en la decisión final.',
        'limitations': 'Las semillas nuevas no crean datos vírgenes; todos los scores se usan para esta decisión pre-registrada.',
        'reference_public009': 0.89555,
    }
    dump('reports/plan_013.json', plan)

    seed_rows = []
    results = {}
    for seed in FRESH_SEEDS:
        ref = evaluate(X, y, g, 'ref009', seed)
        results['ref009', seed] = ref
        for name in NAMES[1:]:
            cand = evaluate(X, y, g, name, seed)
            results[name, seed] = cand
            deltas = np.array(cand['fold_acc']) - np.array(ref['fold_acc'])
            row = {'name': name, 'seed': seed, 'ref009_acc': ref['oof_acc'],
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
                                 gap_delta=('gap_delta', 'mean')).reset_index()
    checks = []
    eligible = []
    for name in NAMES[1:]:
        r = agg[agg.name == name].iloc[0]
        g1 = int(r.fold_wins) >= 7
        g2 = float(r.acc_delta) >= 0.003
        checks.append({'name': name, 'fold_wins': int(r.fold_wins),
                       'mean_acc_delta': float(r.acc_delta),
                       'mean_gap_delta': float(r.gap_delta),
                       'passes_fresh': bool(g1 and g2)})
        if g1 and g2:
            eligible.append(name)
    pd.DataFrame(seed_rows).to_csv('reports/resultados_013.csv', index=False)
    pd.DataFrame(checks).to_csv('reports/criterios_013.csv', index=False)

    hist = {}
    accepted = []
    if eligible:
        ref137 = evaluate(X, y, g, 'ref009', HISTORICAL_SEED)
        assert abs(ref137['oof_acc'] - REFERENCE_137_ACC) < 1e-6, (ref137['oof_acc'], REFERENCE_137_ACC)
        for name in eligible:
            cand = evaluate(X, y, g, name, HISTORICAL_SEED)
            hist[name] = {'ref009_137_acc': ref137['oof_acc'], 'cand_137_acc': cand['oof_acc'],
                          'passes': cand['oof_acc'] >= ref137['oof_acc']}
            if hist[name]['passes']:
                accepted.append(name)
            print('puerta 137', name, hist[name], flush=True)

    winner = None
    if accepted:
        winner = str(agg[agg.name.isin(accepted)].sort_values('acc_delta', ascending=False).iloc[0]['name'])

    rec = {'plan': plan, 'seed_rows': seed_rows, 'checks': checks, 'historical_137': hist,
           'winner': winner, 'submission_created': False, 'reference_public009': 0.89555}

    if winner:
        model = fit(make_model(winner), train.text.fillna(''), train.label)
        joblib.dump(model, 'models/modelo_010.joblib')
        ev = pd.read_csv('data/eval.csv', dtype={'id': str})
        sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
        p = model.predict(ev.text.fillna(''))
        sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(p, index=ev.id))})
        assert len(sub) == len(ev) == len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
        sub.to_csv('submissions/submission_010.csv', index=False)
        pd.testing.assert_frame_equal(sub, pd.read_csv('submissions/submission_010.csv', dtype={'id': str}))
        assert np.array_equal(joblib.load('models/modelo_010.joblib').predict(ev.text.fillna('')), p)
        pred_diag = evaluate(X, y, g, winner, FRESH_SEEDS[-1])['pred']
        per_class = classification_report(y, pred_diag, labels=LABELS, output_dict=True, zero_division=0)
        per_class.pop('accuracy', None)
        rec.update(submission_created=True,
                   artifacts_sha256={pth: sha(pth) for pth in ['models/modelo_010.joblib', 'submissions/submission_010.csv']},
                   per_class=per_class, confusion=confusion_matrix(y, pred_diag, labels=LABELS).tolist(),
                   csv_checks_passed=True,
                   decision=f'{winner} aprobada bajo accuracy; submission_010.csv generado.')
    else:
        rec['decision'] = 'Conservar009: ninguna candidata cumple las puertas pre-registradas bajo accuracy. No generar submission_010.'
    rec['elapsed_seconds'] = time.perf_counter() - started
    rec['source_sha256'] = {p: sha(p) for p in ['src/experimento_013.py', 'src/features_007b.py', 'reports/particiones.csv']}
    dump('reports/registro_013.json', rec)
    print(json.dumps({'winner': winner, 'submission_created': rec['submission_created'],
                      'decision': rec['decision'], 'seconds': rec['elapsed_seconds']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
