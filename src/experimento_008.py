"""Ronda 008: peso del bloque de tercios. Criterio fijado ANTES de ejecutar.

Candidatas:
- ref007   : referencia exacta 007 (TextFeatures007b con thirds=True, un solo TF-IDF).
- split_w1 : bloques base/thirds separados, peso de thirds 1.0 (control del split).
- split_w05: peso de thirds 0.5.
- split_w15: peso de thirds 1.5.

Protocolo (pre-registrado): comparación emparejada contra ref007 en tres semillas
NUEVAS (711, 1237, 2003) nunca usadas en rondas previas, más puerta histórica 137.
Solo una candidata que cumpla las cuatro puertas reemplaza a 007; si ninguna, se
conserva 007 (público 0.88888) y se documenta. No se bajan umbrales después de ver
resultados ni se cambian semillas hasta aprobar.
"""
from src.experimento import normalized, sha, dump, LABELS
from src.features_007b import TextFeatures007b
from src.features_008 import SplitThirdsFeatures
from pathlib import Path
import time, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

FRESH_SEEDS = [711, 1237, 2003]
HISTORICAL_SEED = 137
NAMES = ['ref007', 'split_w1', 'split_w05', 'split_w15']
WEIGHTS = {'split_w1': 1.0, 'split_w05': 0.5, 'split_w15': 1.5}
REFERENCE_137_OOF = 0.8778223683005089  # valor registrado de la 007 en confirmación


def make_model(name):
    if name == 'ref007':
        feats = TextFeatures007b(thirds=True, min_df=3, max_features=160000)
    else:
        feats = SplitThirdsFeatures(third_weight=WEIGHTS[name], min_df=3, max_features=160000)
    return Pipeline([('features', feats),
                     ('clf', LinearSVC(C=2, dual='auto', max_iter=10000, random_state=42))])


def fit(pipe, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipe.fit(X, y)
    assert np.max(pipe['clf'].n_iter_) < pipe['clf'].max_iter
    return pipe


def score(y, p):
    return float(f1_score(y, p, average='macro', labels=LABELS, zero_division=0))


def evaluate(X, y, g, name, seed):
    pred = np.empty(len(y), dtype=object)
    val_scores = []
    train_scores = []
    for k, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g)):
        assert not set(g.iloc[a]) & set(g.iloc[b])
        model = fit(make_model(name), X.iloc[a], y.iloc[a])
        pred[b] = model.predict(X.iloc[b])
        train_scores.append(score(y.iloc[a], model.predict(X.iloc[a])))
        val_scores.append(score(y.iloc[b], pred[b]))
    return {'name': name, 'seed': seed, 'oof_f1': score(y, pred),
            'fold_val': val_scores, 'fold_train': train_scores,
            'gap': float(np.mean([t - v for t, v in zip(train_scores, val_scores)])),
            'pred': pred}


def run():
    started = time.perf_counter()
    if Path('reports/registro_008.json').exists():
        raise FileExistsError('Conservar resultados 008 antes de repetir.')
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    parts = pd.read_csv('reports/particiones.csv', dtype={'id': str})
    d = parts.partition.eq('development')
    gall = normalized(train.text)
    assert not set(gall[d]) & set(gall[~d])
    X = train.text.fillna('')[d].reset_index(drop=True)
    y = train.label[d].reset_index(drop=True)
    g = gall[d].reset_index(drop=True)
    ids = train.id[d].reset_index(drop=True)

    plan = {
        'candidates': NAMES,
        'weights': WEIGHTS,
        'fresh_seeds': FRESH_SEEDS,
        'historical_seed': HISTORICAL_SEED,
        'metric': 'f1_macro; labels explícitos; zero_division=0',
        'gates': [
            'ganar >=7 de 9 folds emparejados frente a ref007 (3 semillas nuevas)',
            'delta medio F1 macro OOF >= 0.003',
            'brecha media <= brecha media de ref007',
            'no perder F1 OOF frente a ref007 en semilla histórica 137',
        ],
        'excluded': 'Las variantes suave (triangular) y de identidad del conector requieren implementación y controles adicionales; se dejan para una ronda posterior fijada de antemano, no se introducen tras ver estos resultados.',
        'limitations': 'Las semillas nuevas no crean datos vírgenes; el desarrollo se ha explorado en muchas rondas. Todos los scores de las semillas nuevas se usan para esta decisión pre-registrada; no son una evaluación externa independiente.',
        'reference_public007': 0.88888,
    }
    dump('reports/plan_008.json', plan)

    fold_rows = []
    seed_rows = []
    results = {}
    for seed in FRESH_SEEDS:
        ref = evaluate(X, y, g, 'ref007', seed)
        results['ref007', seed] = ref
        for name in NAMES[1:]:
            cand = evaluate(X, y, g, name, seed)
            results[name, seed] = cand
            deltas = np.array(cand['fold_val']) - np.array(ref['fold_val'])
            row = {'name': name, 'seed': seed, 'ref007_oof': ref['oof_f1'],
                   'cand_oof': cand['oof_f1'], 'oof_delta': cand['oof_f1'] - ref['oof_f1'],
                   'fold_wins': int((deltas > 0).sum()), 'gap_ref': ref['gap'],
                   'gap_cand': cand['gap'], 'gap_delta': cand['gap'] - ref['gap'],
                   'fold_deltas': np.round(deltas, 5).tolist()}
            seed_rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(seed_rows)
    agg = df.groupby('name').agg(oof_delta=('oof_delta', 'mean'),
                                 fold_wins=('fold_wins', 'sum'),
                                 gap_delta=('gap_delta', 'mean')).reset_index()
    checks = []
    eligible = []
    for name in NAMES[1:]:
        r = agg[agg.name == name].iloc[0]
        g1 = int(r.fold_wins) >= 7
        g2 = float(r.oof_delta) >= 0.003
        g3 = float(r.gap_delta) <= 0
        checks.append({'name': name, 'fold_wins': int(r.fold_wins), 'mean_oof_delta': float(r.oof_delta),
                       'mean_gap_delta': float(r.gap_delta), 'passes_fresh': bool(g1 and g2 and g3)})
        if g1 and g2 and g3:
            eligible.append(name)
    pd.DataFrame(seed_rows).to_csv('reports/resultados_008.csv', index=False)
    pd.DataFrame(checks).to_csv('reports/criterios_008.csv', index=False)

    hist = {}
    accepted = []
    if eligible:
        ref137 = evaluate(X, y, g, 'ref007', HISTORICAL_SEED)
        assert abs(ref137['oof_f1'] - REFERENCE_137_OOF) < 1e-9, (ref137['oof_f1'], REFERENCE_137_OOF)
        for name in eligible:
            cand = evaluate(X, y, g, name, HISTORICAL_SEED)
            hist[name] = {'ref007_137': ref137['oof_f1'], 'cand_137': cand['oof_f1'],
                          'passes': cand['oof_f1'] >= ref137['oof_f1']}
            if hist[name]['passes']:
                accepted.append(name)
            print('puerta 137', name, hist[name], flush=True)

    winner = None
    if accepted:
        winner = str(agg[agg.name.isin(accepted)].sort_values(
            ['oof_delta', 'gap_delta'], ascending=[False, True]).iloc[0]['name'])

    rec = {'plan': plan, 'seed_rows': seed_rows, 'checks': checks, 'historical_137': hist,
           'winner': winner, 'submission_created': False,
           'reference_public007': 0.88888}

    if winner:
        model = fit(make_model(winner), train.text.fillna(''), train.label)
        joblib.dump(model, 'models/modelo_008.joblib')
        ev = pd.read_csv('data/eval.csv', dtype={'id': str})
        sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
        p = model.predict(ev.text.fillna(''))
        sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(p, index=ev.id))})
        assert len(sub) == len(ev) == len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
        sub.to_csv('submissions/submission_008.csv', index=False)
        pd.testing.assert_frame_equal(sub, pd.read_csv('submissions/submission_008.csv', dtype={'id': str}))
        assert np.array_equal(joblib.load('models/modelo_008.joblib').predict(ev.text.fillna('')), p)
        pred_diag = evaluate(X, y, g, winner, FRESH_SEEDS[-1])['pred']
        per_class = classification_report(y, pred_diag, labels=LABELS, output_dict=True, zero_division=0)
        per_class.pop('accuracy', None)
        rec.update(submission_created=True,
                   artifacts_sha256={pth: sha(pth) for pth in ['models/modelo_008.joblib', 'submissions/submission_008.csv']},
                   per_class=per_class, confusion=confusion_matrix(y, pred_diag, labels=LABELS).tolist(),
                   csv_checks_passed=True,
                   decision=f'{winner} aprobada por las puertas pre-registradas; submission_008.csv generado.')
    else:
        rec['decision'] = 'Conservar007: ninguna candidata cumple las cuatro puertas pre-registradas. No generar CSV.'
    rec['elapsed_seconds'] = time.perf_counter() - started
    rec['source_sha256'] = {p: sha(p) for p in ['src/experimento_008.py', 'src/features_008.py', 'src/features_007b.py', 'reports/particiones.csv']}
    dump('reports/registro_008.json', rec)
    print(json.dumps({'winner': winner, 'submission_created': rec['submission_created'],
                      'decision': rec['decision'], 'seconds': rec['elapsed_seconds']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
