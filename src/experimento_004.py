"""Versión 004: ronda acotada de ablaciones sobre el canal discursivo 003.

Protocolo fijado antes de ejecutar (ver reports/plan_004.json):
- Solo desarrollo (9600 filas) para selección y OOF; la antigua prueba local no
  se reutiliza y eval.csv solo sirve para predecir al final.
- CV de selección: StratifiedGroupKFold(3, shuffle=True, random_state=42) sobre
  desarrollo, idéntica a la usada por 003 para su selección final.
- OOF: los mismos 3 folds externos de 003 (seed 137), reconstruidos desde
  reports/predicciones_oof_003.csv para una comparación emparejada honesta.
- Métrica principal: F1 macro con labels explícitos y zero_division=0.
- Criterio de elección (fijado antes de ver resultados):
    1) elegibles = configuraciones con OOF >= max(OOF) - 0.003;
    2) entre elegibles, menor brecha F1 entrenamiento-validación (CV);
    3) desempate por mayor OOF.
- Criterio de parada: la ganadora reemplaza 003 solo si no es la réplica y
  supera la OOF de la réplica en más de 0.002; en caso contrario se conserva 003.
"""
import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

from src.experimento import normalized, sha, dump, SEED, LABELS
from src.features_004 import TextFeatures

ROOT = Path(__file__).resolve().parent.parent

CANDIDATES = [
    dict(name='ref003',         keep_epilogue=False, use_connector=True,  max_ngram=3, min_df=3, char_weight=0.5, C=2.0),
    dict(name='keep_epi',       keep_epilogue=True,  use_connector=True,  max_ngram=3, min_df=3, char_weight=0.5, C=2.0),
    dict(name='no_conn',        keep_epilogue=False, use_connector=False, max_ngram=3, min_df=3, char_weight=0.5, C=2.0),
    dict(name='ng2',            keep_epilogue=False, use_connector=True,  max_ngram=2, min_df=3, char_weight=0.5, C=2.0),
    dict(name='mindf7',         keep_epilogue=False, use_connector=True,  max_ngram=3, min_df=7, char_weight=0.5, C=2.0),
    dict(name='no_char',        keep_epilogue=False, use_connector=True,  max_ngram=3, min_df=3, char_weight=0.0, C=2.0),
    dict(name='c05',            keep_epilogue=False, use_connector=True,  max_ngram=3, min_df=3, char_weight=0.5, C=0.5),
    dict(name='mindf7_no_char', keep_epilogue=False, use_connector=True,  max_ngram=3, min_df=7, char_weight=0.0, C=2.0),
]


def make_pipe(cfg):
    feats = TextFeatures(keep_epilogue=cfg['keep_epilogue'], use_connector=cfg['use_connector'],
                         max_ngram=cfg['max_ngram'], min_df=cfg['min_df'], char_weight=cfg['char_weight'])
    clf = LinearSVC(C=cfg['C'], dual='auto', max_iter=10000, random_state=SEED)
    return Pipeline([('features', feats), ('clf', clf)])


def fit_and_measure(pipe, Xtr, ytr, Xva, yva):
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        pipe.fit(Xtr, ytr)
    conv = any(issubclass(x.category, ConvergenceWarning) for x in w)
    pred_tr = pipe.predict(Xtr)
    pred_va = pipe.predict(Xva)
    n_iter = getattr(pipe['clf'], 'n_iter_', None)
    if isinstance(n_iter, (list, tuple, np.ndarray)):
        n_iter = int(n_iter[0]) if len(n_iter) else None
    else:
        n_iter = int(n_iter) if n_iter is not None else None
    return {
        'fit_s': time.time() - t0,
        'convergence_warning': conv,
        'n_iter': n_iter,
        'f1_train': f1_score(ytr, pred_tr, average='macro', labels=LABELS, zero_division=0),
        'f1_val': f1_score(yva, pred_va, average='macro', labels=LABELS, zero_division=0),
        'pred_va': pred_va,
    }


def run():
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[key] = '1'

    before = {p: sha(p) for p in ['data/train.csv', 'data/eval.csv', 'data/sample_submission.csv']}
    train = pd.read_csv('data/train.csv', dtype={'id': str})
    parts = pd.read_csv('reports/particiones.csv', dtype={'id': str})
    oof3 = pd.read_csv('reports/predicciones_oof_003.csv', dtype={'id': str})
    assert train.id.equals(parts.id)
    assert set(parts.loc[parts.partition.eq('development'), 'id']) == set(oof3.id)
    dev_mask = parts.partition.eq('development').to_numpy()
    X = train.text.fillna('').loc[dev_mask].reset_index(drop=True)
    y = train.label.loc[dev_mask].reset_index(drop=True)
    g = normalized(train.text).loc[dev_mask].reset_index(drop=True)
    # Folds externos de 003 (seed 137), por id, reconstruidos desde su OOF.
    fold_of_id = dict(zip(oof3.id, oof3.outer_fold))
    outer_fold = train.id.loc[dev_mask].map(fold_of_id).to_numpy()
    assert set(np.unique(outer_fold)) == {0, 1, 2}
    for k in (0, 1, 2):
        a = np.flatnonzero(outer_fold != k)
        b = np.flatnonzero(outer_fold == k)
        assert not set(g.iloc[a]) & set(g.iloc[b]), f'grupos no disjuntos en fold externo {k}'

    cv_folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(X, y, g))
    for a, b in cv_folds:
        assert not set(g.iloc[a]) & set(g.iloc[b])

    plan = {
        'metric': 'f1_macro exclusivamente (labels explícitos, zero_division=0)',
        'seed_cv': SEED,
        'seed_outer': 137,
        'outer_folds_source': 'reports/predicciones_oof_003.csv (mismos folds externos de 003)',
        'candidates': CANDIDATES,
        'selection_rule': 'OOF máxima; entre <=0.003 del máximo, menor brecha F1 train-val (CV); desempate mayor OOF',
        'stop_rule': 'la ganadora reemplaza 003 solo si no es ref003 y supera su OOF en >0.002; si no, se conserva 003',
        'reference_003_oof_recorded': 0.8354002604334546,
        'limitations': 'Los folds externos y el desarrollo ya fueron observados en 003; la OOF es comparación emparejada, no evaluación virgen.',
    }
    dump('reports/plan_004.json', plan)

    cv_rows = []
    oof_arrays = {}
    oof_meta = {}
    for cfg in CANDIDATES:
        name = cfg['name']
        print(f'== {name} {cfg}', flush=True)
        # CV de selección (3 folds, seed 42) -> train/test F1 y brecha.
        fold_rows = []
        for a, b in cv_folds:
            r = fit_and_measure(make_pipe(cfg), X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b])
            fold_rows.append(r)
        mean_train = float(np.mean([r['f1_train'] for r in fold_rows]))
        mean_test = float(np.mean([r['f1_val'] for r in fold_rows]))
        std_test = float(np.std([r['f1_val'] for r in fold_rows]))
        gap = max(0.0, mean_train - mean_test)
        cv_rows.append(dict(name=name, C=cfg['C'], keep_epilogue=cfg['keep_epilogue'],
                            use_connector=cfg['use_connector'], max_ngram=cfg['max_ngram'],
                            min_df=cfg['min_df'], char_weight=cfg['char_weight'],
                            mean_train_f1=mean_train, mean_test_f1=mean_test, std_test_f1=std_test,
                            f1_gap=gap,
                            convergence_warnings=sum(1 for r in fold_rows if r['convergence_warning']),
                            fit_s_mean=float(np.mean([r['fit_s'] for r in fold_rows]))))
        # OOF en los folds externos de 003.
        oof = np.empty(len(y), dtype=object)
        conv_outer = 0
        for k in (0, 1, 2):
            a = np.flatnonzero(outer_fold != k)
            b = np.flatnonzero(outer_fold == k)
            r = fit_and_measure(make_pipe(cfg), X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b])
            oof[b] = r['pred_va']
            conv_outer += int(r['convergence_warning'])
        oof_arrays[name] = oof
        oof_meta[name] = dict(
            f1_macro=float(f1_score(y, oof, average='macro', labels=LABELS, zero_division=0)),
            convergence_warnings=conv_outer,
        )
        print(f'   CV: train={mean_train:.6f} test={mean_test:.6f} gap={gap:.6f} | '
              f'OOF={oof_meta[name]["f1_macro"]:.6f}', flush=True)

    # Selección con la regla fijada.
    df = pd.DataFrame(cv_rows).set_index('name')
    oof_f1 = pd.Series({n: oof_meta[n]['f1_macro'] for n in df.index})
    df['oof_f1'] = oof_f1
    df = df.reset_index()
    best_oof = df.oof_f1.max()
    eligible = df[df.oof_f1 >= best_oof - 0.003].copy()
    winner = eligible.sort_values(['f1_gap', 'oof_f1'], ascending=[True, False]).iloc[0]
    winner_name = winner['name']
    df['eligible'] = df.oof_f1 >= best_oof - 0.003
    df['selected'] = df.name == winner_name
    df.to_csv('reports/resultados_cv_004.csv', index=False)

    # OOF amplia: id, label, fold + una columna por config.
    oof_out = pd.DataFrame({'id': train.id.loc[dev_mask].reset_index(drop=True),
                            'label': y, 'outer_fold': outer_fold})
    for n in df.name:
        oof_out['pred_' + n] = oof_arrays[n]
    oof_out.to_csv('reports/predicciones_oof_004.csv', index=False)

    ref_oof = oof_meta['ref003']['f1_macro']
    replace = (winner_name != 'ref003') and (winner.oof_f1 > ref_oof + 0.002)

    per_class = classification_report(y, oof_arrays[winner_name], labels=LABELS,
                                      output_dict=True, zero_division=0)
    per_class.pop('accuracy', None)
    record = {
        'plan': plan,
        'reference_003_oof_recorded': 0.8354002604334546,
        'reference_003_oof_reproduced': ref_oof,
        'winner': winner.to_dict(),
        'replaces_003': bool(replace),
        'n_candidates': len(CANDIDATES),
        'oof_f1_by_candidate': {n: oof_meta[n]['f1_macro'] for n in df.name},
        'per_class_winner_oof': per_class,
        'confusion_winner_oof': confusion_matrix(y, oof_arrays[winner_name], labels=LABELS).tolist(),
        'labels': LABELS,
        'local_test_reused': False,
        'public_score_004': None,
        'versions': json.loads(Path('reports/registro_experimento_001.json').read_text())['versions'],
        'input_sha256': {p: before[p] for p in before},
        'original_files_unchanged': all(before[p] == sha(p) for p in before),
    }
    dump('reports/registro_004.json', record)

    print('\n== Resultados finales ==', flush=True)
    print(df[['name', 'mean_train_f1', 'mean_test_f1', 'f1_gap', 'oof_f1', 'eligible', 'selected']].to_string(index=False), flush=True)
    print(f'\nReferencia 003 OOF registrada={record["reference_003_oof_recorded"]:.6f} reproducida={ref_oof:.6f}', flush=True)
    print(f'Ganadora={winner_name} OOF={winner.oof_f1:.6f} gap={winner.f1_gap:.6f} reemplaza_003={replace}', flush=True)

    if replace:
        winner_cfg = next(c for c in CANDIDATES if c['name'] == winner_name)
        model = make_pipe(winner_cfg)
        model.fit(train.text.fillna(''), train.label)
        joblib.dump(model, 'models/modelo_004.joblib')
        ev = pd.read_csv('data/eval.csv', dtype={'id': str})
        sample = pd.read_csv('data/sample_submission.csv', dtype={'id': str})
        assert ev.id.is_unique and sample.id.is_unique and set(ev.id) == set(sample.id)
        pred = model.predict(ev.text.fillna(''))
        submission = pd.DataFrame({'id': sample.id, 'answer': sample.id.map(pd.Series(pred, index=ev.id))})
        assert len(submission) == len(ev) == len(sample) and submission.id.is_unique
        assert submission.id.equals(sample.id) and not submission.isna().any().any()
        assert set(submission.answer) <= set(LABELS)
        submission.to_csv('submissions/submission_004.csv', index=False)
        pd.testing.assert_frame_equal(submission, pd.read_csv('submissions/submission_004.csv', dtype={'id': str}))
        assert np.array_equal(joblib.load('models/modelo_004.joblib').predict(ev.text.fillna('')), pred)
        record['artifacts_sha256'] = {p: sha(p) for p in ['models/modelo_004.joblib', 'submissions/submission_004.csv']}
        record['checks'] = {
            'csv_ids_order_schema_labels': True,
            'reload_predictions': True,
            'roundtrip_csv': True,
            'disjoint_groups': True,
            'originals_unchanged': True,
        }
        dump('reports/registro_004.json', record)
        print('Artefactos 004 generados: models/modelo_004.joblib, submissions/submission_004.csv', flush=True)

    print('Fin experimento 004', flush=True)
    return record


if __name__ == '__main__':
    run()
