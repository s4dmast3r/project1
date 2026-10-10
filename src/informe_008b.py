"""Genera informe y notebook ejecutado a partir de la ronda cerrada 008b."""
from pathlib import Path
import json
import sys
import pandas as pd
import nbformat as nb
from nbclient import NotebookClient
from jupyter_client import KernelManager

r = json.loads(Path('reports/registro_008b.json').read_text())
cv = pd.read_csv('reports/resultados_cv_008b.csv')
main = cv[cv.seed.isin(r['plan']['seeds'])]
assert len(main) == 18
means = main.groupby('name').agg(f1=('oof_f1', 'mean'), gap=('gap', 'mean'))
criteria = pd.read_csv('reports/criterios_008b.csv').set_index('name')
lines = ['# Ronda 008b: estrategias investigadas y ejecutadas', '',
         '## Resultado', '',
         f"Candidata seleccionada: **{r['winner'] or 'ninguna'}**. CSV creado: **{r['submission_created']}**.", '',
         '| Modelo | F1 macro OOF medio | Diferencia frente a 007 | Brecha entrenamiento-validación | Folds ganados |',
         '|---|---:|---:|---:|---:|']
for name, row in means.sort_values('f1', ascending=False).iterrows():
    wins = '—' if name == 'ref007' else f"{int(criteria.loc[name, 'folds_won'])}/9"
    lines.append(f'| {name} | {row.f1:.6f} | {row.f1-means.loc["ref007", "f1"]:+.6f} | {row.gap:.6f} | {wins} |')
lines += ['', 'Los valores son medias de F1 macro calculado sobre predicciones fuera de fold (OOF) en tres particiones de desarrollo. La brecha es la media de F1 de entrenamiento menos F1 de validación por fold. La brecha menor no prueba ausencia de sobreajuste.', '',
          '## Qué se implementó', '',
          '- `ref007`: reproducción del modelo 007, TF-IDF con posición por tercios y LinearSVC C=2.',
          '- `joint_w05` y `joint_w15`: pesos 0,5 y 1,5 para tercios en el mismo TF-IDF y normalización final L2. El control de peso 1 reproduce numéricamente 007.',
          '- `connector_identity`: características que unen la identidad del conector con palabras de su cláusula, sin reglas manuales de etiqueta.',
          '- `chi2_50`: selección supervisada del 50 % de características dentro de cada fold, seguida de normalización y LinearSVC C=2.',
          '- `nbsvm`: presencia binaria de las características de discurso de 007, ponderación logarítmica Naive Bayes por clase frente al resto y tres SVM C=1; gana la clase de mayor margen. Es una adaptación multiclase del artículo, sin su interpolación opcional.', '',
          'Investigación y fuentes: [detalle](investigacion_008b.md), [Wang y Manning, 2012](https://aclanthology.org/P12-2018/), [selección de características](https://scikit-learn.org/stable/modules/feature_selection.html), [prevención de fuga](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Los recursos externos aportan métodos, nunca ejemplos de entrenamiento.', '',
          '## Protocolo y decisión', '', r['plan']['gates'], '',
          'Se usaron las mismas semillas 711, 1237 y 2003 de la ronda 008 para comparar con 007. La ronda anterior se conserva. Las candidatas elegibles se compararon también en la partición histórica 137:', '']
for name, h in r['historical'].items():
    lines.append(f'- {name}: {h["candidate"]:.6f}, referencia 007: {h["reference"]:.6f}; supera puerta: {h["passes"]}.')
if r['submission_created']:
    lines += ['', 'Tras cerrar la selección se ajustó la candidata elegida con los 12000 ejemplos etiquetados. Se exportaron `models/modelo_008.joblib` y `submissions/submission_008.csv`. El CSV conserva el orden del ejemplo de envío, con 3000 identificadores únicos, columnas `id,answer` y etiquetas válidas. No se ha enviado a la plataforma.']
else:
    lines += ['', r['decision']]
lines += ['', '## Sobreajuste y límites', '',
          'Solo 9600 ejemplos de desarrollo participan en selección. Los 2400 del antiguo test local no se reutilizan para seleccionar; ya habían sido observados en la historia del proyecto y no se presentan como prueba virgen. Tras decidir el modelo, incluirlos en el ajuste final para predecir eval es legítimo.', '',
          'Se mantienen grupos disjuntos por texto normalizado. Vocabulario, IDF, chi² y cocientes Naive Bayes se aprenden únicamente con la parte de entrenamiento de cada fold. Las etiquetas de validación solo sirven para calcular F1 macro. Eval no interviene en estos ajustes ni decisiones.', '',
          'Todas estas métricas son de selección: se han explorado repetidamente los mismos datos. Los folds y semillas no son observaciones independientes; no se atribuye significación estadística a contar victorias. El criterio fijo reduce decisiones oportunistas en esta ronda, pero no elimina el sobreajuste de selección acumulado. Tampoco se garantiza ausencia de duplicados semánticos con una normalización simple.', '',
          'La captura aportada registra 0,88888 público para 007. No verifica la fórmula oficial de la métrica. El score público de 008 es desconocido y el F1 local no promete 0,90 público. El entrenamiento casi perfecto, incluso con una brecha menor, exige conservar esta cautela.', '',
          'La conclusión causal del informe anterior 008 era demasiado fuerte: separar bloques altera su normalización, pero no el IDF de características comunes con igual frecuencia documental. El control real de esta ronda comprueba esta igualdad; no se concluye que toda ponderación de tercios sea inútil.', '',
          '## Reproducibilidad', '',
          f'Tiempo del experimento: {r["elapsed_seconds"]:.1f} segundos. Artefactos anteriores intactos: {r["old_artifacts_unchanged"]}.', '',
          'Desde la raíz: `.venv/bin/python -m src.experimento_008b`. El registro existente impide repetir accidentalmente la ronda. Generar este informe y notebook: `.venv/bin/python -m src.informe_008b`.', '',
          'Los hashes, versiones, resultados por fold, predicciones OOF y criterios están en los archivos `registro_008b.json`, `folds_008b.csv`, `predicciones_oof_008b.csv` y `criterios_008b.csv` de reports. El notebook reproduce el análisis guardado; no vuelve a entrenar.']
report = '\n'.join(lines) + '\n'
Path('reports/resumen_008b.md').write_text(report)
cells = [nb.v4.new_markdown_cell('# Ronda 008b — investigación y resultados\nAnálisis de una ejecución cerrada. Todos los scores locales intervienen en selección.'),
         nb.v4.new_code_cell("""from pathlib import Path
import os, json
import pandas as pd
from IPython.display import display, Markdown
if not Path('reports/registro_008b.json').exists(): os.chdir('..')
r = json.loads(Path('reports/registro_008b.json').read_text())
display(Markdown(Path('reports/resumen_008b.md').read_text()))
display(pd.read_csv('reports/resultados_cv_008b.csv'))
display(pd.read_csv('reports/criterios_008b.csv'))
display(r['historical'])"""),
         nb.v4.new_code_cell("""import matplotlib.pyplot as plt
cv = pd.read_csv('reports/resultados_cv_008b.csv')
main = cv[cv.seed.isin(r['plan']['seeds'])]
ax = main.pivot(index='seed', columns='name', values='oof_f1').plot(marker='o', figsize=(10,5))
ax.set(xlabel='Semilla de partición', ylabel='F1 macro OOF', title='Comparación de selección; no prueba externa')
plt.tight_layout()
plt.savefig('reports/comparacion_008b.png', dpi=160)
plt.show()
display(pd.read_csv('reports/folds_008b.csv'))"""),
         nb.v4.new_code_cell("""display(r['source_sha256'])
print('Artefactos anteriores intactos:', r['old_artifacts_unchanged'])
if r['submission_created']:
    display(r['artifacts_sha256'])
    display(pd.read_csv('submissions/submission_008.csv').head())
else:
    print(r['decision'])""")]
notebook = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name':'python3', 'display_name':'Python del proyecto', 'language':'python'}})
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
NotebookClient(notebook, km=km, timeout=180, resources={'metadata': {'path':str(Path.cwd())}}).execute()
nb.write(notebook, 'notebooks/08b_modelo_f1_macro.ipynb')
print(report)
