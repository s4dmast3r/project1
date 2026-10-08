"""Audita el experimento cerrado sin seleccionar parámetros ni tocar el envío."""
from src.experimento import normalized, sha, dump, SEED
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score


def run():
    record=json.loads(Path('reports/registro_experimento_001.json').read_text())
    protected={p:sha(p) for p in [*record['input_sha256'],*record['artifacts_sha256']]}
    assert all(sha(p)==h for p,h in {**record['input_sha256'],**record['artifacts_sha256']}.items())
    train=pd.read_csv('data/train.csv',dtype={'id':str})
    ev=pd.read_csv('data/eval.csv',dtype={'id':str})
    sample=pd.read_csv('data/sample_submission.csv',dtype={'id':str})
    parts=pd.read_csv('reports/particiones.csv',dtype={'id':str})
    assert train.id.equals(parts.id) and train.label.equals(parts.label)
    X=train.text.fillna(''); y=train.label; groups=normalized(X)
    dev,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,groups))
    assert set(dev)==set(np.flatnonzero(parts.partition.eq('development')))
    assert set(test)==set(np.flatnonzero(parts.partition.eq('local_test')))
    assert not set(dev)&set(test) and len(dev)+len(test)==len(train)
    assert not set(groups.iloc[dev])&set(groups.iloc[test])
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X.iloc[dev],y.iloc[dev],groups.iloc[dev]))
    final=joblib.load('models/modelo_001.joblib')
    for k,v in record['selected_params'].items():
        assert final.get_params()[k]==(tuple(v) if isinstance(v,list) else v)
    rows=[]
    # Solo desarrollo. La prueba local NO se vuelve a predecir ni se usa para ajustes.
    for i,(a,b) in enumerate(folds):
        fit_idx,val_idx=dev[a],dev[b]
        assert not set(fit_idx)&set(val_idx) and not set(fit_idx)&set(test)
        assert not set(groups.iloc[fit_idx])&set(groups.iloc[val_idx])
        assert set(val_idx)==set(np.flatnonzero(parts.cv_fold.eq(i)))
        estimator=clone(final).fit(X.iloc[fit_idx],y.iloc[fit_idx])
        assert max(estimator['clf'].n_iter_)<estimator['clf'].max_iter
        # Vectorizador independiente ajustado SOLO al entrenamiento de este fold.
        vectorizer=clone(final['tfidf']).fit(X.iloc[fit_idx])
        assert vectorizer.vocabulary_==estimator['tfidf'].vocabulary_
        np.testing.assert_allclose(vectorizer.idf_,estimator['tfidf'].idf_)
        tr=estimator.predict(X.iloc[fit_idx]); va=estimator.predict(X.iloc[val_idx])
        row={'fold':i,'train_rows':len(a),'validation_rows':len(b),'train_accuracy':accuracy_score(y.iloc[fit_idx],tr),'validation_accuracy':accuracy_score(y.iloc[val_idx],va),'train_f1_macro':f1_score(y.iloc[fit_idx],tr,average='macro'),'validation_f1_macro':f1_score(y.iloc[val_idx],va,average='macro'),'iterations':int(max(estimator['clf'].n_iter_))}
        row['accuracy_gap']=row['train_accuracy']-row['validation_accuracy']
        rows.append(row)
        print(row,flush=True)
    scores=pd.DataFrame(rows)
    np.testing.assert_allclose(scores.validation_accuracy.mean(),record['metrics']['cv']['mean_test_accuracy'])
    # El modelo de producción se ajustó sobre todo train una vez cerrado el diagnóstico.
    full_vectorizer=clone(final['tfidf']).fit(X)
    assert full_vectorizer.vocabulary_==final['tfidf'].vocabulary_
    np.testing.assert_allclose(full_vectorizer.idf_,final['tfidf'].idf_)
    submission=pd.read_csv('submissions/submission_001.csv',dtype={'id':str})
    assert list(submission)==['id','answer'] and len(submission)==len(ev)==len(sample)
    assert ev.id.is_unique and sample.id.is_unique and submission.id.is_unique
    assert submission.id.equals(sample.id) and set(ev.id)==set(sample.id)
    assert not submission.isna().any().any() and set(submission.answer)<=set(train.label)
    predictions=pd.Series(final.predict(ev.text.fillna('')),index=ev.id)
    assert np.array_equal(sample.id.map(predictions),submission.answer)
    assert protected=={p:sha(p) for p in protected}
    scores.to_csv('reports/auditoria_sobreajuste_folds.csv',index=False)
    summary={'checks_passed':True,'development_rows':len(dev),'untouched_local_test_rows':len(test),'train_accuracy_mean':scores.train_accuracy.mean(),'validation_accuracy_mean':scores.validation_accuracy.mean(),'train_validation_gap':scores.accuracy_gap.mean(),'original_local_accuracy':record['metrics']['accuracy'],'cv_local_difference':scores.validation_accuracy.mean()-record['metrics']['accuracy'],'vocabulary_and_idf_match_training_only_each_fold':True,'final_vocabulary_and_idf_match_train_csv_only':True,'model_and_submission_unchanged':True,'protected_sha256':protected,'scope':'Configuración congelada, cinco reajustes diagnósticos solo sobre desarrollo. No se reevalúa la prueba local ni se seleccionan nuevos parámetros. No garantiza ausencia de sobreajuste ni descarta similitud entre paráfrasis.'}
    dump('reports/auditoria_sobreajuste.json',summary)
    report=f'''# Revisión de separación y sobreajuste

Se verificó la integridad de los archivos contra las huellas del primer experimento. Las {len(dev)} filas de desarrollo y las {len(test)} de prueba local son disjuntas; tampoco comparten grupos de texto normalizado. Los cinco folds coinciden con las asignaciones guardadas. La revisión del código confirma que la búsqueda recibe exclusivamente desarrollo y que cada Pipeline aprende vocabulario e IDF dentro de su fold. Se verificó numéricamente la igualdad con vectorizadores ajustados únicamente sobre cada entrenamiento. Los IDs no son características. Las respuestas de sample_submission no se usan.

## Diagnóstico con configuración congelada

- Accuracy media sobre entrenamiento de cada fold: {scores.train_accuracy.mean():.6f}.
- Accuracy media sobre validación de cada fold: {scores.validation_accuracy.mean():.6f}.
- Brecha entrenamiento-validación: {100*scores.accuracy_gap.mean():.2f} puntos porcentuales.
- Accuracy de la prueba local original, sin reevaluarla: {record['metrics']['accuracy']:.6f}.
- Diferencia CV-prueba local original: {100*(scores.validation_accuracy.mean()-record['metrics']['accuracy']):.2f} puntos porcentuales.

La brecha entrenamiento-validación mide adaptación al entrenamiento y debe reconocerse como señal de sobreajuste. No se debe publicar la accuracy de entrenamiento como capacidad de generalización. La cercanía entre CV y prueba local no demuestra ausencia de sobreajuste, pero no muestra una caída grande adicional en la prueba reservada. La CV participó en la selección de 32 configuraciones y puede tener optimismo de selección; la prueba reservada aporta una estimación independiente del procedimiento original. No se ajustan parámetros después de observarla.

La regularización y min_df ya forman parte del modelo seleccionado. No se reduce C automáticamente después de conocer la prueba: eso convertiría esa prueba en otra validación. Cualquier nuevo experimento debe planificarse en desarrollo y requerirá una evaluación independiente para afirmar una mejora.

## Envío

El ajuste final sobre todo train.csv incluye las antiguas filas de prueba local solo DESPUÉS de cerrar selección y evaluación. Esto es correcto para el modelo de producción, pero ese modelo final no puede evaluarse honestamente sobre esas filas. Las métricas publicadas corresponden al modelo previo ajustado solo sobre desarrollo. eval.csv no participa en ningún fit: se verificó además que el vocabulario e IDF finales coinciden con un ajuste exclusivo sobre train.csv.

El modelo guardado reproduce las respuestas de submission_001.csv; esquema, etiquetas, IDs y orden son correctos. Ambos archivos permanecen idénticos, verificados por SHA-256. El CSV está listo para el primer envío manual; no se consultó Kaggle.

## Límites

No hay duplicados exactos o normalizados según la auditoría original. Esto no descarta paráfrasis, plantillas compartidas o dependencias de origen no documentadas. Sin metadatos de autor, producto o procedencia no se puede garantizar independencia por esas entidades. Tampoco se puede garantizar una puntuación privada ni ausencia absoluta de sobreajuste. Esta revisión no usa el texto de eval para decidir cambios en el modelo.
'''
    cv=pd.read_csv('reports/resultados_cv.csv')
    sensitivity=cv[(cv.param_tfidf__ngram_range=='(1, 2)') & (cv.param_tfidf__min_df==2) & (cv.param_tfidf__sublinear_tf==False)]
    report+='\n## Regularización: evidencia de la búsqueda original\n\nCon el resto de parámetros seleccionados fijos, los resultados ya ejecutados fueron:\n\n'
    for _,r in sensitivity.sort_values('param_clf__C').iterrows():
        report+=f"- C={r.param_clf__C:g}: accuracy CV {r.mean_test_accuracy:.6f}.\n"
    report+='\nReducir C aumenta la regularización, pero en esta comparación no mejora la accuracy de validación. Se conserva la configuración original para el primer envío; esto no elimina la brecha observada ni garantiza que sea la mejor solución posible.\n'
    Path('reports/revision_sobreajuste.md').write_text(report)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': run()
