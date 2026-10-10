"""Confirmación pre-registrada de `thirds` sobre semillas NUEVAS (1142, 2087, 3313).

Motivo: en la ronda 007b, `thirds` ganó 9/9 folds emparejados frente a ref005
(media +0.0028, p(signo)≈0.002) y redujo la brecha en las tres semillas, pero no
alcanzó el umbral de +0.005 por semilla fijado para esa ronda. Para no relajar un
criterio después de ver resultados, esta confirmación fija ANTES de ejecutar un
criterio propio sobre semillas que nunca se han usado en ninguna ronda:

  1. thirds gana >=7 de los 9 folds emparejados (3 semillas x 3 folds);
  2. delta medio de F1 macro OOF >= 0.002;
  3. brecha media de thirds <= brecha media de ref005;
  4. puerta histórica: thirds no pierde F1 OOF frente a ref005 en semilla 137.

Solo si se cumplen las cuatro, se produce submission_007.csv; si no, se conserva
005 (0.88111 público) y se documenta. Todos los scores de las semillas nuevas se
usan para la decisión pre-registrada; no son una evaluación externa independiente.
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

NEW_SEEDS = [1142, 2087, 3313]
HISTORICAL_SEED = 137


def make_model(thirds=False):
    feats = TextFeatures007b(min_df=3, max_features=160000, thirds=thirds)
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


def evaluate(X, y, g, thirds, seed):
    pred = np.empty(len(y), dtype=object)
    folds = []
    for k, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g)):
        assert not set(g.iloc[a]) & set(g.iloc[b])
        model = fit(make_model(thirds), X.iloc[a], y.iloc[a])
        pred[b] = model.predict(X.iloc[b])
        folds.append((score(y.iloc[a], model.predict(X.iloc[a])), score(y.iloc[b], pred[b])))
    return {'oof_f1': score(y, pred), 'fold_val': [v for _, v in folds],
            'gap': float(np.mean([tr - va for tr, va in folds])), 'pred': pred}


def run():
    started = time.perf_counter()
    if Path('reports/registro_confirmacion_007b.json').exists():
        raise FileExistsError('Conservar confirmación existente antes de repetir.')
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    parts = pd.read_csv('reports/particiones.csv', dtype={'id': str})
    d = parts.partition.eq('development')
    gall = normalized(train.text)
    assert not set(gall[d]) & set(gall[~d])
    X = train.text.fillna('')[d].reset_index(drop=True)
    y = train.label[d].reset_index(drop=True)
    g = gall[d].reset_index(drop=True)
    ids = train.id[d].reset_index(drop=True)

    criterio = {
        'motivo': 'thirds ganó 9/9 folds en 007b pero bajo el umbral +0.005 por semilla; se confirma en semillas nuevas con criterio fijado antes de ver resultados.',
        'nuevas_semillas': NEW_SEEDS,
        'puertas': [
            'ganar >=7 de 9 folds emparejados (thirds vs ref005)',
            'delta medio F1 macro OOF >= 0.002',
            'brecha media thirds <= brecha media ref005',
            'no perder F1 OOF en semilla histórica 137',
        ],
        'nota': 'Los scores de las semillas nuevas se usan para esta decisión pre-registrada; no son evaluación externa independiente.',
    }
    dump('reports/plan_confirmacion_007b.json', criterio)

    fold_rows = []
    new_seed_rows = []
    for seed in NEW_SEEDS:
        ref = evaluate(X, y, g, False, seed)
        cand = evaluate(X, y, g, True, seed)
        deltas = np.array(cand['fold_val']) - np.array(ref['fold_val'])
        row = {'seed': seed, 'ref005_oof': ref['oof_f1'], 'thirds_oof': cand['oof_f1'],
               'oof_delta': cand['oof_f1'] - ref['oof_f1'], 'fold_wins': int((deltas > 0).sum()),
               'gap_ref': ref['gap'], 'gap_thirds': cand['gap'],
               'gap_delta': cand['gap'] - ref['gap'], 'fold_deltas': np.round(deltas, 5).tolist()}
        new_seed_rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(new_seed_rows)
    total_wins = int(df.fold_wins.sum())
    mean_delta = float(df.oof_delta.mean())
    mean_gap_delta = float(df.gap_delta.mean())
    gate1 = total_wins >= 7
    gate2 = mean_delta >= 0.002
    gate3 = mean_gap_delta <= 0

    hist = None
    gate4 = False
    if gate1 and gate2 and gate3:
        ref = evaluate(X, y, g, False, HISTORICAL_SEED)
        cand = evaluate(X, y, g, True, HISTORICAL_SEED)
        hist = {'ref005_oof_137': ref['oof_f1'], 'thirds_oof_137': cand['oof_f1'],
                'passes': cand['oof_f1'] >= ref['oof_f1']}
        gate4 = hist['passes']
        print('puerta histórica 137:', hist, flush=True)

    approved = bool(gate1 and gate2 and gate3 and gate4)
    rec = {'criterio': criterio, 'seed_rows': new_seed_rows,
           'totals': {'fold_wins': total_wins, 'mean_oof_delta': mean_delta,
                      'mean_gap_delta': mean_gap_delta},
           'gates': {'wins_7_9': gate1, 'mean_delta_002': gate2, 'gap_not_worse': gate3,
                     'historical_137': gate4},
           'approved': approved, 'submission_created': False}

    if approved:
        model = fit(make_model(True), train.text.fillna(''), train.label)
        joblib.dump(model, 'models/modelo_007.joblib')
        ev = pd.read_csv('data/eval.csv', dtype={'id': str})
        sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
        p = model.predict(ev.text.fillna(''))
        sub = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(p, index=ev.id))})
        assert len(sub) == len(ev) == len(sample) and sub.id.is_unique and sub.id.equals(sample.id)
        assert not sub.isna().any().any() and set(sub.answer) <= set(LABELS)
        sub.to_csv('submissions/submission_007.csv', index=False)
        pd.testing.assert_frame_equal(sub, pd.read_csv('submissions/submission_007.csv', dtype={'id': str}))
        assert np.array_equal(joblib.load('models/modelo_007.joblib').predict(ev.text.fillna('')), p)
        # Per-class OOF de thirds (solo referencia diagnóstica, última semilla nueva)
        all_pred = evaluate(X, y, g, True, NEW_SEEDS[-1])['pred']
        per_class = classification_report(y, all_pred, labels=LABELS, output_dict=True, zero_division=0)
        per_class.pop('accuracy', None)
        rec.update(submission_created=True,
                   artifacts_sha256={pth: sha(pth) for pth in ['models/modelo_007.joblib', 'submissions/submission_007.csv']},
                   per_class_last_seed_oof=per_class,
                   confusion_last_seed=confusion_matrix(y, all_pred, labels=LABELS).tolist(),
                   csv_checks_passed=True,
                   decision='thirds aprobada por confirmación pre-registrada en semillas nuevas; submission_007.csv generado.')
    else:
        rec['decision'] = 'Conservar005: la confirmación pre-registrada no se cumplió. No generar CSV.'
    rec['elapsed_seconds'] = time.perf_counter() - started
    rec['source_sha256'] = {p: sha(p) for p in ['src/confirmar_007b.py', 'src/features_007b.py', 'reports/particiones.csv']}
    dump('reports/registro_confirmacion_007b.json', rec)
    print(json.dumps({'approved': approved, 'submission_created': rec['submission_created'],
                      'gates': rec['gates'], 'decision': rec['decision']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
