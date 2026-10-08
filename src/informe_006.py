from pathlib import Path
import json,sys
import pandas as pd
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
r=json.loads(Path('reports/registro_006.json').read_text())
cv=pd.read_csv('reports/resultados_cv_006.csv');assert len(cv)==4
for k in range(3):
 t=pd.read_csv(f'reports/cv_006_interna_{k}.csv');assert len(t)==4 and t.filter(regex='split.*test_score').notna().all().all()
report=f'''# Versión 006: recomendaciones contrastadas

Candidato seleccionado: **{r['selected_config']['name']}**.

| F1 macro | Referencia 005 | Procedimiento 006 |
|---|---:|---:|
| Selección CV | {r['baseline_selection_f1']:.6f} | {r['selection_f1']:.6f} |
| Brecha entrenamiento-CV | {r['baseline_gap']:.6f} | {r['gap']:.6f} |
| Predicciones OOF externas agrupadas | {r['baseline_nested_oof_f1']:.6f} | {r['nested_oof_f1']:.6f} |

F1 entrenamiento de selección: {r['train_f1']:.6f}. Media de folds externos: {r['external_fold_mean']:.6f} ± {r['external_fold_std']:.6f}. El F1 sobre todas las predicciones externas no es la misma operación que promediar F1 por fold.

## Cambios y evaluación

Se contrastaron referencia 005, C=1/min_df=5, eliminación de interacciones de epílogo y stacking SVC/LR/NB. Solo F1 macro dirige la selección, con regla previa de control de brecha y mejora mínima. Los detalles se fijaron en reports/plan_006.json y src/modelos_006.py.

La CV externa repite selección interna. Cada stacking genera sus propios scores OOF de bases dentro del entrenamiento recibido; no reutiliza scores globales para entrenar el agregador. QuantileTransformer aprende únicamente sobre OOF de entrenamiento, nunca calcula rangos a partir del lote eval. Las bases no necesitan ser estadísticamente independientes: su utilidad debe comprobarse fuera de entrenamiento.

La cifra exploratoria 0,8805 del documento del auditor no se usa como resultado verificado: no se encontró su implementación y artefactos intermedios en las carpetas revisadas. Esta implementación tampoco pretende reproducir exactamente un ranking por fold no especificado.

El público 005 aportado por el usuario es 0,88111; no se consultó la plataforma. El público 006 es desconocido. Meta local 0,90: {'alcanzada en esta OOF' if r['nested_oof_f1']>=0.90 else 'no alcanzada en esta OOF'}. No se extrapola del score público anterior.

## Sobreajuste y límites

La brecha reportada es diagnóstica; su reducción no demuestra ausencia de sobreajuste. El mismo desarrollo se ha explorado en varias versiones. La CV anidada protege los ajustes de esta ronda, pero no borra adaptación histórica ni dependencias entre plantillas. La prueba local antigua no se reutilizó. Tras selección, el modelo final se ajusta con todo train; si gana stacking, también se regeneran scores OOF de todo train para su meta-modelo. eval solo se utiliza para predecir.

Los scores externos no se utilizaron para cambiar parámetros de esta ronda. No se ejecutaron stemming, stopwords, árboles ni KNN, para limitar coste y decisiones adaptativas. No se añadieron datos externos o sintéticos.

## Entregables

CSV nuevo creado: {r['submission_created']}. {'Usar submissions/submission_006.csv; modelo models/modelo_006.joblib. Conservar src/modelos_006.py, src/features_004.py y src/experimento.py para las dependencias del Pipeline.' if r['submission_created'] else 'Ningún candidato cumplió la regla previa; conservar la versión 005. No se fabrica un envío equivalente.'}

Duración medida: {r['elapsed_seconds']:.1f} segundos. Los archivos anteriores permanecen intactos. Registro: reports/registro_006.json. Resultados completos: reports/resultados_cv_006.csv, cv_006_interna_*.csv y cv_006_anidada.csv.
'''
Path('reports/resumen_006.md').write_text(report)
cells=[n.v4.new_markdown_cell('''# Experimento 006 — stacking sin fuga indirecta y simplificación

Ejecutar desde raíz: `.venv/bin/python -m src.experimento_006` para reproducir entrenamiento, y `.venv/bin/python src/informe_006.py` para regenerar este notebook. Guardar copias si se necesita preservar otra ejecución de 006. No se modifican versiones anteriores.

La justificación detallada está en `reports/revision_recomendaciones_006.md`. El documento del auditor se trata como recomendaciones, no como prueba de resultados cuya ejecución no está disponible.'''),n.v4.new_code_cell('''from pathlib import Path
import os,json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
if not Path('reports/registro_006.json').exists():os.chdir('..')
r=json.loads(Path('reports/registro_006.json').read_text())
display(Markdown(Path('reports/resumen_006.md').read_text()))
cv=pd.read_csv('reports/resultados_cv_006.csv')
display(cv[['name','mean_train_score','mean_test_score','std_test_score','f1_gap','selected']])
display(pd.read_csv('reports/cv_006_anidada.csv'))
display(pd.DataFrame(r['per_class']).T)'''),n.v4.new_code_cell('''a=np.array(r['confusion'])
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,v,title in [(axs[0],a,'Conteos OOF'),(axs[1],a/a.sum(axis=1,keepdims=True),'Proporción por clase real')]:
 ax.imshow(v,cmap='Blues');ax.set(xticks=range(3),yticks=range(3),xticklabels=r['labels'],yticklabels=r['labels'],xlabel='Predicción',ylabel='Clase real',title=title)
 for i in range(3):
  for j in range(3):ax.text(j,i,str(v[i,j]) if title=='Conteos OOF' else f'{v[i,j]:.3f}',ha='center')
plt.tight_layout();plt.savefig('reports/confusion_006.png',dpi=160);plt.show()
oof=pd.read_csv('reports/predicciones_oof_006.csv',dtype={'id':str})
train=pd.read_csv('data/train.csv',dtype={'id':str})
e=oof[oof.label.ne(oof.prediction)].merge(train[['id','text']],on='id',validate='one_to_one')
for _,x in e.sample(min(5,len(e)),random_state=42).iterrows():display(Markdown(f'**ID {x.id}, real {x.label}, predicción {x.prediction}.** '+x.text))'''),n.v4.new_markdown_cell('''Los ejemplos se seleccionan con semilla fija después de cerrar parámetros. No se reentrena para corregirlos individualmente. Las valoraciones contrapuestas no prueban por sí solas la causa de un fallo.

Se conserva el criterio F1 macro y las tres etiquetas explícitas. Precisión, recall y matriz son diagnósticos. Una comparación entre modelos ajustados con matrices OOF previamente generadas sobre todo desarrollo podría filtrar etiquetas al meta-modelo; esta implementación regenera las OOF dentro de cada fit del stacking.

Las capturas no muestran el nombre de la métrica del concurso: el criterio local proviene de la instrucción del usuario. El próximo envío será manual y su resultado sigue pendiente.'''),n.v4.new_code_cell('''display(r.get('checks',{}));display(r.get('artifacts_sha256',{}))
print('Versiones anteriores intactas:',r['old_artifacts_unchanged'])
if r['submission_created']:
 s=pd.read_csv('submissions/submission_006.csv',dtype={'id':str});display(s.head());display(s.answer.value_counts())
else: print(r['reason'])''')]
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python del proyecto','language':'python'}})
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(Path.cwd())}}).execute()
n.write(nb,'notebooks/06_modelo_f1_macro.ipynb')
print(report)
