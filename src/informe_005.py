"""Resumen y notebook de análisis de los resultados ejecutados de 005."""
from pathlib import Path
import json,sys
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
import pandas as pd
r=json.loads(Path('reports/registro_005.json').read_text())
cv=pd.read_csv('reports/resultados_cv_005.csv');folds=pd.read_csv('reports/cv_005_anidada.csv')
assert len(cv)==6 and len(folds)==3
for k in range(3):
 t=pd.read_csv(f'reports/cv_005_interna_{k}.csv');assert len(t)==6 and t.filter(regex='split.*test_score').notna().all().all()
report=f'''# Versión 005: simplificación de 004 con F1 macro

Configuración seleccionada: `{r['selected_config']}`. Clasificador LinearSVC C=2.

| F1 macro | Referencia 004 | Procedimiento 005 |
|---|---:|---:|
| CV usada para selección | {r['baseline_selection_f1']:.6f} | {r['selection_f1']:.6f} |
| Brecha train-CV | {r['baseline_gap']:.6f} | {r['gap']:.6f} |
| OOF externa agrupada | {r['baseline_nested_oof_f1']:.6f} | {r['nested_oof_f1']:.6f} |

F1 de entrenamiento de la selección 005: {r['train_f1']:.6f}. F1 medio de folds externos: {r['external_fold_mean']:.6f} ± {r['external_fold_std']:.6f}. La media de folds y F1 de predicciones agrupadas no son la misma operación.

## Interpretación

El auditor mejoró 004 quitando trigramas de cláusulas. Esta ronda contrasta más simplificación, frecuencia mínima y ausencia de caracteres con un presupuesto previo de seis configuraciones. Selección exclusivamente por F1 macro y control de brecha: {r['rule']}

La evaluación externa repite la selección dentro del entrenamiento de cada fold: no selecciona directamente por los scores externos como hizo 004. Desarrollo ya fue examinado en rondas anteriores, por lo que esta separación no equivale a una prueba nueva. La referencia 004 también fue elegida históricamente con estos folds; no se oculta esa adaptación.

La meta 0,90 {'se alcanza' if r['nested_oof_f1']>=0.90 else 'no se alcanza'} en la estimación externa local. Un F1 mayor no elimina el sobreajuste. La brecha persiste y las familias de textos o plantillas podrían inducir dependencia no capturada por los grupos de minúsculas/espacios.

La antigua prueba local se excluyó. Tras cerrar selección, todo train se usa para producción; eval solo para predecir. No se consultó Kaggle. El público conocido de 004 es 0,86111; no es el resultado de esta versión ni se usó para ajustar parámetros. El público 005 está pendiente.

## Artefactos y verificaciones

CSV nuevo creado: {r['submission_created']}. {'Archivo: submissions/submission_005.csv. Modelo: models/modelo_005.joblib. El Pipeline requiere src/features_004.py.' if r['submission_created'] else 'La regla previa no permitió reemplazar la referencia: conservar submission_004.csv.'}

Se verificaron matrices dispersas, grupos disjuntos, convergencia de los nuevos SVM, conservación de datos y artefactos anteriores. Si se generó CSV, se verificaron esquema id,answer, cantidad, IDs únicos, orden de sample_submission, etiquetas, ausencia de vacíos, lectura de ida y vuelta y predicciones tras recarga. Duración total medida del experimento: {r['elapsed_seconds']:.1f} segundos.

La numeración 005 es versión de archivo. Los envíos reales reportados por el usuario son 002, 003 y 004; esta sería la siguiente carga manual, salvo otros envíos no comunicados.
'''
Path('reports/resumen_005.md').write_text(report)
cells=[n.v4.new_markdown_cell('''# Experimento 005 — simplificación y validación de F1 macro
Reproducir entrenamiento desde la raíz con `.venv/bin/python -m src.experimento_005`; este comando regenera artefactos de 005, por lo que se debe conservar una copia si se necesita su historial. Generar este notebook con `.venv/bin/python src/informe_005.py`. No se vuelven a ejecutar los experimentos anteriores.

La auditoría previa está en `reports/revision_auditor_004_para_005.md`. La regla y las seis alternativas se registraron antes de ejecutar en `reports/plan_005.json`.'''),n.v4.new_code_cell('''from pathlib import Path
import os,json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
if not Path('reports/registro_005.json').exists():os.chdir('..')
r=json.loads(Path('reports/registro_005.json').read_text())
display(Markdown(Path('reports/resumen_005.md').read_text()))
cv=pd.read_csv('reports/resultados_cv_005.csv')
display(cv[['name','mean_train_score','mean_test_score','std_test_score','f1_gap','selected']])
display(pd.read_csv('reports/cv_005_anidada.csv'))'''),n.v4.new_markdown_cell('''## Diagnóstico externo
Estas métricas usan predicciones fuera del fold de entrenamiento. Los parámetros se escogen dentro de cada entrenamiento externo. No se reajusta la configuración tras observar estas matrices. Precisión y recall son diagnósticos; el criterio de selección sigue siendo F1 macro.'''),n.v4.new_code_cell('''display(pd.DataFrame(r['per_class']).T)
a=np.asarray(r['confusion'])
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,values,title in [(axs[0],a,'Conteos OOF'),(axs[1],a/a.sum(axis=1,keepdims=True),'Proporción por clase real')]:
 ax.imshow(values,cmap='Blues')
 ax.set(xticks=range(3),yticks=range(3),xticklabels=r['labels'],yticklabels=r['labels'],xlabel='Predicción',ylabel='Clase real',title=title)
 for i in range(3):
  for j in range(3):ax.text(j,i,str(values[i,j]) if title=='Conteos OOF' else f'{values[i,j]:.3f}',ha='center')
plt.tight_layout();plt.savefig('reports/confusion_005.png',dpi=160);plt.show()
oof=pd.read_csv('reports/predicciones_oof_005.csv',dtype={'id':str})
train=pd.read_csv('data/train.csv',dtype={'id':str})
errors=oof[oof.label.ne(oof.prediction)].merge(train[['id','text']],on='id',validate='one_to_one')
for _,x in errors.sample(min(5,len(errors)),random_state=42).iterrows():
 display(Markdown(f'**ID {x.id}; real {x.label}; predicción {x.prediction}.** '+x.text))'''),n.v4.new_markdown_cell('''## Interpretación de errores y límites
La selección de ejemplos es reproducible y posterior al cierre de parámetros. Leer una reseña con valoraciones contrapuestas no demuestra cuál expresión causó un fallo; evitar explicaciones causales no verificadas. La comparación emparejada sirve para describir diferencias, pero las decisiones históricas sobre este mismo desarrollo limitan la independencia de la evaluación. Un score público puede fluctuar por la composición de su subconjunto.

La menor complejidad se contrasta junto al F1 de validación; no se presupone que menor vocabulario o mayor regularización siempre sean mejores. Nunca se usan los IDs como características ni respuestas de sample_submission como etiquetas. Los originales 001–004 se conservan.'''),n.v4.new_code_cell('''display(r.get('checks',{}))
display(r.get('artifacts_sha256',{}))
print('Artefactos anteriores intactos:',r['old_artifacts_unchanged'])
if r['submission_created']:
 s=pd.read_csv('submissions/submission_005.csv',dtype={'id':str});display(s.head());display(s.answer.value_counts())
else: print(r['reason'])''')]
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python del proyecto','language':'python'}})
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(Path.cwd())}}).execute()
n.write(nb,'notebooks/05_modelo_f1_macro.ipynb')
print(report)
