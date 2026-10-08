from pathlib import Path
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
import sys
nb=n.v4.new_notebook(cells=[
n.v4.new_markdown_cell('''# Corrección de sobreajuste: versión 002

El experimento reproducible está en `src/regularizar_modelo.py`. Desde la raíz, ejecutar `.venv/bin/python -m src.regularizar_modelo` para entrenar y luego `.venv/bin/python src/notebook_regularizacion.py` para regenerar este análisis con kernel limpio.

El usuario priorizó reducir sobreajuste. Antes de ejecutar se fijó una regla: entre modelos a un punto porcentual o menos de la mejor accuracy CV, elegir la menor brecha entrenamiento-validación. Se limita vocabulario y frecuencia mínima y se compara regularización. No se usa la prueba local ya examinada para ajustar ni evaluar de nuevo. Las cifras siguientes son CV de selección, no una nueva estimación independiente.'''),
n.v4.new_code_cell('''from pathlib import Path
import os, json
import pandas as pd
from IPython.display import display, Markdown
if not Path('reports/registro_experimento_002.json').exists():
    os.chdir('..')
r=json.loads(Path('reports/registro_experimento_002.json').read_text())
display(Markdown(Path('reports/correccion_sobreajuste_002.md').read_text()))'''),
n.v4.new_code_cell('''cv=pd.read_csv('reports/resultados_cv_regularizacion_002.csv')
cols=['params','mean_train_accuracy','mean_test_accuracy','std_test_accuracy','mean_test_f1_macro','accuracy_gap','selected']
with pd.option_context('display.max_rows',30,'display.max_colwidth',180):
    display(cv[cols].sort_values('mean_test_accuracy',ascending=False))
display(r['checks'])
display(r['artifact_sha256'])'''),
n.v4.new_markdown_cell('''## Interpretación y límites
Una brecha menor sugiere menos adaptación al entrenamiento; no basta por sí sola para demostrar mejor generalización. El límite de pérdida en validación evita elegir una solución muy débil solo por una brecha pequeña. No se garantiza ausencia de sobreajuste, ni independencia de reseñas que compartan plantillas o procedencia desconocida.

El modelo final se ajusta sobre todo train después de seleccionar parámetros. El antiguo resultado de prueba local pertenece al modelo anterior, nunca al modelo corregido. El CSV original se conserva para trazabilidad; la versión 002 puede ser el primer envío real si aún no se ha subido ninguno.''')])
nb.metadata={'kernelspec':{'name':'python3','display_name':'Python del proyecto','language':'python'}}
km=KernelManager(kernel_name='python3'); km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=120,resources={'metadata':{'path':str(Path.cwd())}}).execute()
n.write(nb,'notebooks/02_regularizacion.ipynb')
