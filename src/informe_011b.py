"""Informe reproducible y notebook ejecutado de la ronda011b."""
from pathlib import Path
import json,sys
import pandas as pd
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
r=json.loads(Path('reports/registro_011b.json').read_text())
cv=pd.read_csv('reports/resultados_011b.csv');main=cv[cv.seed.isin(r['plan']['seeds'])]
assert len(main)==18
means=main.groupby('name')[['accuracy','f1','gap']].mean()
checks=pd.read_csv('reports/criterios_011b.csv').set_index('name')
lines=['# Ronda011b: revisión de010 y alternativas clásicas','',
'Accuracy es el criterio principal y F1 macro es un control, por confirmación explícita del usuario el9 de octubre de2026. Público010: **0,89777**. Público011: desconocido.','',
'## Qué mejoró en010','',
'La entrega010 corresponde al experimento013: conserva TF-IDF de discurso y posición, y LinearSVC L1, pero cambia C=1 por C=2. La menor regularización elevó la accuracy OOF media de0,895660 a0,899062 y la brecha de0,060799 a0,088976. El modelo pasó de3252 a5491 coeficientes no nulos. Mejoró el público de0,89555 a0,89777, a costa de más ajuste al entrenamiento. Los hashes y la recarga del modelo010 se verificaron antes de comparar.','',
'## Investigación e implementación','',
'Ver [investigacion_011b.md](investigacion_011b.md) para fuentes primarias y decisiones. Se implementaron presencia binaria con/sin IDF, TF sublineal sin IDF, identidad de conectores con L1 y regresión logística L1. No se incorporaron datos externos. La ronda011 previa se preserva.','',
'## Resultados de selección','',
'| Candidata | Accuracy media OOF | F1 macro medio OOF | Brecha accuracy | Folds ganados vs010 |',
'|---|---:|---:|---:|---:|']
for name,row in means.sort_values('accuracy',ascending=False).iterrows():
    wins='—' if name=='ref010' else str(int(checks.loc[name,'folds_won']))+'/9'
    lines.append(f'| {name} | {row.accuracy:.6f} | {row.f1:.6f} | {row.gap:.6f} | {wins} |')
lines+=['','Semillas8803/9901/11003, tres folds agrupados por semilla. Los puntajes OOF se calculan sobre las9600 predicciones de desarrollo de cada semilla y luego se promedian. La brecha promedia accuracy de entrenamiento menos validación por fold.','',
'## Criterio fijado antes de entrenar','',r['plan']['gates'],'',
'La comparación137 es otra puerta de selección, no una prueba independiente.']
for name,h in r['historical'].items():
    lines.append(f"- {name}: accuracy {h['candidate']['accuracy']:.6f} vs {h['reference']['accuracy']:.6f}; F1 {h['candidate']['f1']:.6f} vs {h['reference']['f1']:.6f}. Aprobada: {h['passes']}.")
lines+=['','## Decisión','',f"Ganadora: **{r['winner'] or 'ninguna'}**. CSV creado: **{r['submission_created']}**."]
if r['submission_created']:
    lines+=['','Se ajustó la ganadora con las12000 filas de train después de cerrar selección. Exportación: `submissions/submission_011.csv` y `models/modelo_011.joblib`. Se verificaron3000 filas, IDs únicos en el orden del ejemplo, etiquetas válidas, lectura de ida/vuelta y reproducción tras recargar el modelo. No se ha enviado a la plataforma.']
else:lines+=['',r['decision']]
lines+=['','## Límites y controles','',
'No se usó el test local histórico para elegir ni eval para ajustar. Todos los vocabularios, IDF y coeficientes se aprendieron dentro del entrenamiento de cada fold. Grupos de texto normalizado disjuntos; esa agrupación no garantiza detectar duplicados semánticos. Convergencia exigida. Artefactos previos preservados por hash.','',
'Los datos de desarrollo se han explorado repetidamente. Estos scores sirven para selección y pueden ser optimistas; las nuevas semillas no crearían datos vírgenes. Los9folds no son independientes ni se presenta un test de significación a partir de contar victorias. Menor brecha es un diagnóstico, no una garantía de ausencia de sobreajuste.','',
'Que una métrica local coincida numéricamente con un score público no demuestra cuál es la métrica oficial. Aquí accuracy se usa por instrucción directa del usuario. Tampoco los experimentos negativos previos prueban un techo absoluto para ML clásico. No se promete0,90 público.','',
'## Reproducibilidad','',
f"Tiempo de experimento: {r['elapsed_seconds']:.1f} segundos. Artefactos anteriores intactos: {r['old_artifacts_unchanged']}.",'',
'Entrenamiento: `.venv/bin/python -m src.experimento_011b`. La ejecución se detiene si ya existe el registro o el destino del modelo/envío. Informe y notebook: `.venv/bin/python -m src.informe_011b`. El notebook analiza resultados guardados, no vuelve a entrenar.','',
'Archivos de auditoría: `plan_011b.json`, `registro_011b.json`, `folds_011b.csv`, `resultados_011b.csv`, `criterios_011b.csv` y `predicciones_oof_011b.csv`.']
report='\n'.join(lines)+'\n';Path('reports/resumen_011b.md').write_text(report)
cells=[n.v4.new_markdown_cell('# Ronda011b — accuracy principal, F1 macro control\nAnálisis de una ejecución cerrada; resultados de selección, no prueba externa.'),
n.v4.new_code_cell("""from pathlib import Path
import os,json
import pandas as pd
from IPython.display import display,Markdown
if not Path('reports/registro_011b.json').exists():os.chdir('..')
r=json.loads(Path('reports/registro_011b.json').read_text())
display(Markdown(Path('reports/resumen_011b.md').read_text()))
cv=pd.read_csv('reports/resultados_011b.csv')
display(cv)
display(pd.read_csv('reports/criterios_011b.csv'))
display(r['historical'])"""),
n.v4.new_code_cell("""import matplotlib.pyplot as plt
main=cv[cv.seed.isin(r['plan']['seeds'])]
fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,metric in zip(axes,['accuracy','f1']):
 main.pivot(index='seed',columns='name',values=metric).plot(marker='o',ax=ax)
 ax.set(xlabel='Semilla',ylabel=metric,title='Comparación local de selección')
plt.tight_layout();plt.savefig('reports/comparacion_011b.png',dpi=160);plt.show()
display(pd.read_csv('reports/folds_011b.csv'))"""),
n.v4.new_code_cell("""display(r['source_sha256'])
print('Artefactos anteriores intactos:',r['old_artifacts_unchanged'])
if r['submission_created']:
 display(r['artifacts_sha256'])
 display(pd.read_csv('submissions/submission_011.csv').head())
else:print(r['decision'])""")]
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python del proyecto','language':'python'}})
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(Path.cwd())}}).execute()
n.write(nb,'notebooks/11b_revision_y_modelo.ipynb');print(report)
