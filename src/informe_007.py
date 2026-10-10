from pathlib import Path
import json,sys
import pandas as pd
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
r=json.loads(Path('reports/registro_007.json').read_text())
cv=pd.read_csv('reports/resultados_cv_007.csv');gates=pd.read_csv('reports/criterios_007.csv')
assert len(cv[cv.seed.isin([271,599,913])])==12 and len(gates)==9
main=cv[cv.seed.isin([271,599,913])]
mean=main.groupby('name').agg(f1=('oof_f1','mean'),gap=('gap','mean'))
lines=['# Experimento 007 — negación, contraste y estabilidad','',
'La ronda se ejecutó con cuatro candidatos, tres semillas y tres folds por semilla. Todos los scores intervienen en la selección; no son una evaluación independiente. Se preservó la prueba local antigua. Cambiar semillas no elimina el historial de exploración.','',
'| Candidato | F1 macro OOF medio entre semillas | Brecha media |','|---|---:|---:|']
for name,row in mean.iterrows():lines.append(f'| {name} | {row.f1:.6f} | {row.gap:.6f} |')
lines+=['','## Criterios y decisión','',r['plan']['selection'],'',f"Ganadora: {r['winner'] or 'ninguna'}. CSV nuevo creado: {r['submission_created']}.",
('Archivo listo: submissions/submission_007.csv. Modelo: models/modelo_007.joblib.' if r['submission_created'] else r['decision']),
'', '## Interpretación y límites','',
'Los canales añadidos son características, no reglas de etiqueta: no se obliga a que una negación invierta la clase ni a que la cláusula posterior al contraste domine la reseña. La negación se aproxima con tres tokens hasta puntuación o corte de contraste; no constituye análisis sintáctico completo. El canal post_contraste abarca el segmento posterior a pero/aunque/sin embargo hasta la puntuación de oración. Ambos conservan toda la representación base de 005. Peso adicional 0,35, min_df=3 y hasta 30000 rasgos se fijaron sin optimizar contra resultados.',
'', 'C=1/min_df=5 combina dos cambios ya propuestos en el documento; no aísla causalmente cuál de ellos explica diferencias. El opcional char_wb no se ejecutó para mantener el presupuesto. No se borran ni reetiquetan reseñas; no se usan datos externos o sintéticos.',
'', 'Se exigió +0,005 en CADA semilla, no solo en el promedio. En cada una también se exigieron tres victorias de fold, o al menos dos y brecha menor. La comparación histórica137 solo se ejecuta si una candidata supera esos requisitos. Si se usa, es otra puerta de selección, no prueba confirmatoria. La semilla271 ya se usó dentro del stacking006, aunque no como partición externa de esa versión.',
'', 'El documento del auditor atribuye la caída pública de006 a adaptación a folds. Es una hipótesis plausible, no causalidad demostrada por un score público. El tamaño del delta respecto a la desviación entre folds tampoco basta para concluir señal o ruido: haría falta analizar diferencias emparejadas e incertidumbre, considerando adaptación. El documento cita ~0,87 sin captura exacta disponible en este turno; no se registra como valor exacto confirmado.',
'', 'El público confirmado de005 es0,88111. La plataforma no fue consultada. No se extrapolan puntajes ni se garantiza0,90. F1 macro sigue siendo el criterio local por instrucción del usuario; las capturas no verifican la fórmula oficial.',
'', '## Reproducibilidad','',
f"Tiempo medido del experimento: {r['elapsed_seconds']:.1f} segundos. Archivos anteriores intactos: {r['old_artifacts_unchanged']}.",
'Entrenamiento: `.venv/bin/python -m src.experimento_007`. Se detiene si ya existe registro_007.json para evitar sobrescribir resultados accidentalmente; archivar explícitamente la ejecución previa antes de repetir. Notebook: `.venv/bin/python src/informe_007.py`. Modelos anteriores y módulos no se modifican.']
report='\n'.join(lines)+'\n';Path('reports/resumen_007.md').write_text(report)
cells=[n.v4.new_markdown_cell('''# Ronda 007: características de negación y contraste
La instrucción del usuario autoriza implementar el plan `revision_006_y_plan_007.md`. Sus afirmaciones se contrastan, no se asumen como resultados ya comprobados. Las reglas de esta ejecución están en `plan_007.json`. La referencia es 005, conservando 006 para trazabilidad.'''),n.v4.new_code_cell('''from pathlib import Path
import os,json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
if not Path('reports/registro_007.json').exists():os.chdir('..')
r=json.loads(Path('reports/registro_007.json').read_text())
display(Markdown(Path('reports/resumen_007.md').read_text()))
cv=pd.read_csv('reports/resultados_cv_007.csv')
display(cv)
display(pd.read_csv('reports/criterios_007.csv'))
display(r['historical'])'''),n.v4.new_code_cell('''main=cv[cv.seed.isin([271,599,913])]
ax=main.pivot(index='seed',columns='name',values='oof_f1').plot(marker='o',figsize=(9,4))
ax.set(xlabel='Semilla',ylabel='F1 macro OOF',title='CV repetida de selección; comparación emparejada')
plt.tight_layout();plt.savefig('reports/comparacion_007.png',dpi=160);plt.show()
folds=pd.read_csv('reports/folds_007.csv')
display(folds[['name','seed','fold','train_f1','validation_f1','gap','iterations']])'''),n.v4.new_markdown_cell('''## Comprobaciones y límites
Todas las particiones separan grupos de texto normalizado; los vectorizadores aprenden vocabulario e IDF dentro de cada entrenamiento. Convergencia del SVM exigida como condición de ejecución. Las pruebas de la función de aceptación usan únicamente métricas de ejemplo, nunca reseñas sintéticas de entrenamiento.

Una candidata no aceptada no se exporta bajo otro número solo para fabricar un envío. Si se acepta, se reentrena con todo train tras cerrar selección y eval se usa únicamente para generar predicciones. El mismo modelo y CSV previos permanecen disponibles.

Los puntajes de selección no son pruebas externas nuevas. Ni el cambio de semilla ni las condiciones más estrictas garantizan ausencia de sobreajuste. No se utiliza la prueba local anterior ni el leaderboard para ajustar estas características.'''),n.v4.new_code_cell('''display(r['source_sha256'])
print('Archivos anteriores intactos:',r['old_artifacts_unchanged'])
if r['submission_created']:
 display(r['artifacts_sha256']);display(pd.read_csv('submissions/submission_007.csv').head())
else:print(r['decision'])''')]
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python del proyecto','language':'python'}})
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(Path.cwd())}}).execute()
n.write(nb,'notebooks/07_modelo_f1_macro.ipynb')
print(report)
