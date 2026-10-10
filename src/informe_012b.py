"""Informe y notebook de la comparación012b frente a011."""
from pathlib import Path
import json,sys
import pandas as pd
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
r=json.loads(Path('reports/registro_012b.json').read_text())
cv=pd.read_csv('reports/resultados_012b.csv');main=cv[cv.seed.isin(r['plan']['seeds'])];assert len(main)==15
means=main.groupby('name')[['accuracy','f1','gap']].mean();checks=pd.read_csv('reports/criterios_012b.csv').set_index('name')
lines=['# Ronda012b — auditoría011 y decisiones multiclase','',
'Accuracy principal y F1 macro control por autorización del usuario. Score visible011: **0,89888**, frente a010:0,89777. Score012 desconocido.','',
'## Qué cambió en011','',
'Los coeficientes de011 son idénticos a010: LinearSVC L1 C=2 y las mismas características. El cambio es sumar0,2 a los márgenes positivo y negativo antes del argmax. Solo4 de3000 predicciones cambian. La subida pública es0,00111; no permite identificar los aciertos públicos ni demostrar un techo de rendimiento.','',
'El artefacto modelo_011.joblib guarda el clasificador base sin esa regla: su predict() difiere del CSV en4filas. La regla se aplica en producir_011.py. La nueva referencia encapsula ambos componentes en predict() y usa nombres de clase. Los artefactos previos no se modifican. El script anterior sí reentrena la base, aunque el informe decía que no; se verificó igualdad de coeficientes.','',
'## Métodos investigados e implementados','',
'- `ref011`: reproducción de011 con descuento neutral0,2.',
'- `polarity_c1` y `polarity_c2`: conservan exactamente la puerta neutral011; un SVM binario decide positivo/negativo, entrenado solo con ejemplos polares del entrenamiento.',
'- `ovo_l1`: tres clasificadores entre pares de clases y combinación de decisiones, todos L1 C=2.',
'- `sigmoid`: calibración con predicciones internas fuera de fold; tres particiones internas agrupadas, vocabulario/IDF dentro de cada ajuste y refit final de la base. Sin descuento manual adicional.', '',
'Fuentes y protocolo: [investigacion_012b.md](investigacion_012b.md), [OneVsOneClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html) y [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html). No se añadieron datos externos ni sintéticos.','',
'## Resultados locales de selección','',
'| Candidata | Accuracy media OOF | F1 macro medio OOF | Brecha accuracy | Folds ganados |','|---|---:|---:|---:|---:|']
for name,row in means.sort_values('accuracy',ascending=False).iterrows():
    wins='—' if name=='ref011' else str(int(checks.loc[name,'folds_won']))+'/9'
    lines.append(f'| {name} | {row.accuracy:.6f} | {row.f1:.6f} | {row.gap:.6f} | {wins} |')
lines+=['','Las medias OOF promedian los resultados de9600 predicciones de desarrollo por semilla. La brecha es accuracy de entrenamiento menos validación, promediada entre folds. Son las mismas particiones para todas las candidatas.','',
'## Criterios y decisión','',r['plan']['gates'],'',r['plan']['gate_rationale'],'',
'Este es un protocolo nuevo, más sensible a mejoras pequeñas que011b; no se reclasifican retrospectivamente los resultados de esa ronda. Las condiciones se fijaron antes de medir012b y no se relajaron después.']
for name,h in r['historical'].items():
    lines.append(f"- Puerta137 para{name}: accuracy {h['candidate']['accuracy']:.6f} vs {h['reference']['accuracy']:.6f}; F1 {h['candidate']['f1']:.6f} vs {h['reference']['f1']:.6f}; aprobación: {h['passes']}.")
lines+=['',f"Ganadora: **{r['winner'] or 'ninguna'}**. CSV creado: **{r['submission_created']}**."]
if r['submission_created']:
    lines+=['','Tras cerrar selección, se ajustó la ganadora con las12000 filas etiquetadas. Se exportaron `models/modelo_012.joblib` y `submissions/submission_012.csv`; predict() contiene la regla de decisión completa. CSV validado:3000 filas, id/answer, orden del ejemplo, etiquetas válidas y recarga reproducible. No se envió a la plataforma.']
else:lines+=['',r['decision']]
lines+=['','## Controles y límites','',
'Las9600 filas de desarrollo son las únicas usadas para seleccionar. El test local histórico no se reutiliza para decidir; eval se usa únicamente para auditar artefactos anteriores y exportar después de decidir. Grupos de texto normalizado disjuntos. Vocabulario, IDF, clasificadores y calibración aprendidos dentro del entrenamiento correspondiente; convergencia exigida. La calibración interna se rehace dentro de cada fold externo.','',
'Hay selección adaptativa acumulada sobre este dataset: nuevas semillas no crean datos vírgenes. La CV interna evita fuga del fold externo al calibrador, pero no elimina el sesgo de elegir entre alternativas tras muchas rondas. Los nueve folds repetidos no son independientes. Una brecha menor no demuestra ausencia de sobreajuste y una tolerancia de0,005 no significa que ningún aumento sea aceptable universalmente.','',
'El registro previo011 declara una calibración anidada de su descuento, pero no se localizaron tablas/código de esas semillas en esta revisión. La exportación011 sí se verificó, y la nueva comparación es reproducible. No se asume que frecuencia de clases justifique automáticamente descontar un margen SVM. No se promete0,90 público ni se concluye un techo absoluto del ML clásico.','',
'## Reproducibilidad','',f"Tiempo: {r['elapsed_seconds']:.1f} segundos. Artefactos anteriores intactos: {r['old_artifacts_unchanged']}.",'',
'`.venv/bin/python -m src.experimento_012b` (protege destinos existentes), `.venv/bin/python -m src.verificar_012b` y `.venv/bin/python -m src.informe_012b`. El notebook analiza resultados guardados; no vuelve a entrenar. Ver plan, registro, criterios, folds y predicciones_oof con sufijo012b.']
report='\n'.join(lines)+'\n';Path('reports/resumen_012b.md').write_text(report)
cells=[n.v4.new_markdown_cell('# Ronda012b — decisiones multiclase\nAccuracy principal; F1 macro control. Resultados de selección sobre datos ya explorados.'),
n.v4.new_code_cell("""from pathlib import Path
import os,json
import pandas as pd
from IPython.display import display,Markdown
if not Path('reports/registro_012b.json').exists():os.chdir('..')
r=json.loads(Path('reports/registro_012b.json').read_text())
display(Markdown(Path('reports/resumen_012b.md').read_text()))
cv=pd.read_csv('reports/resultados_012b.csv');display(cv)
display(pd.read_csv('reports/criterios_012b.csv'));display(r['historical'])"""),
n.v4.new_code_cell("""import matplotlib.pyplot as plt
main=cv[cv.seed.isin(r['plan']['seeds'])]
fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,metric in zip(axes,['accuracy','f1']):
 main.pivot(index='seed',columns='name',values=metric).plot(marker='o',ax=ax)
 ax.set(xlabel='Semilla',ylabel=metric,title='Comparación local de selección')
plt.tight_layout();plt.savefig('reports/comparacion_012b.png',dpi=160);plt.show()
display(pd.read_csv('reports/folds_012b.csv'))"""),
n.v4.new_code_cell("""display(r['source_sha256'])
print('Artefactos anteriores intactos:',r['old_artifacts_unchanged'])
if r['submission_created']:
 display(r['artifacts_sha256']);display(pd.read_csv('submissions/submission_012.csv').head())
else:print(r['decision'])""")]
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python del proyecto','language':'python'}})
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(Path.cwd())}}).execute()
n.write(nb,'notebooks/12b_revision_y_modelo.ipynb');print(report)
