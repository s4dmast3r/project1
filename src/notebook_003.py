from pathlib import Path
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
import sys
cells=[]
def md(x): cells.append(n.v4.new_markdown_cell(x))
def code(x): cells.append(n.v4.new_code_cell(x))
md('''# Versión 003: selección exclusivamente por F1 macro

El usuario reportó un puntaje público de 0,73888 para la versión 002 y pidió usar exclusivamente F1 macro. La captura muestra el puntaje, pero no el nombre de la métrica del concurso. Se registra el valor como evidencia aportada por el usuario, sin asumir que verifica por sí solo la fórmula del leaderboard. La meta solicitada se interpreta como F1 macro 0,90 (90%), no como una garantía.

## Qué cambia y por qué
Una bolsa global de palabras no distingue bien el papel de cada valoración en una reseña con sentimientos mixtos. Comparamos una representación de palabras y caracteres con representaciones que distinguen tercios del texto y posiciones de las oraciones desde ambos extremos. Una variante segmenta cláusulas por conectores y distingue cierres discursivos como «en fin» o «cada quien», sin asignar una polaridad a esos marcadores. Otra variante añade interacciones entre los n-gramas y las primeras dos o tres palabras de la última oración; son rasgos observables, sin polaridad impuesta ni etiquetas externas. El clasificador aprende sus pesos. No existe una regla que asigne la clase de la última oración.

Se normalizan acentos en esta versión para compartir evidencia entre variantes ortográficas; no se eliminan negaciones ni palabras vacías. Los caracteres permiten compartir fragmentos entre palabras y tolerar algunos errores tipográficos. Las nuevas representaciones aumentan capacidad, por lo que se usan frecuencia mínima, límites de vocabulario y regularización. Todos los vocabularios e IDF se aprenden dentro del entrenamiento de cada Pipeline.

## Protocolo fijado antes de ejecutar
Nueve configuraciones: cuatro representaciones con dos valores de C y la referencia 002. Se elige por F1 macro; entre resultados a 0,003 del máximo se prefiere menor brecha de F1 entrenamiento-validación. La brecha no se minimiza sola, pues eso puede favorecer subajuste.

Se usa solo el desarrollo anterior: 9600 filas. La antigua prueba local permanece excluida. La CV anidada usa tres folds externos y tres internos; cada fold externo repite la selección únicamente dentro de su entrenamiento. Se reporta además el 002 sobre exactamente los mismos folds externos.

**Límite:** desarrollo ya fue usado en experimentos anteriores y se leyeron ejemplos para diseñar las características. La CV anidada protege la separación de ajuste y selección dentro de cada fold, pero no convierte estos datos en una prueba externa virgen. No se deben reutilizar los resultados de la prueba anterior como métricas de este modelo.''')
code('''from pathlib import Path
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
if not Path('reports/registro_experimento_003.json').exists(): os.chdir('..')
r=json.loads(Path('reports/registro_experimento_003.json').read_text())
cv=pd.read_csv('reports/resultados_cv_003.csv')
display(cv[['params','mean_train_score','mean_test_score','std_test_score','f1_gap','selected']])
display(Markdown('**Configuración elegida:** '+r['selected_params']))
comparison=pd.DataFrame({'medida':['F1 macro CV selección','F1 macro OOF anidada 003','F1 macro OOF anidada referencia 002'], 'valor':[r['selected_cv_f1_macro'],r['nested_f1_macro_oof'],r['nested_baseline002_f1_macro_oof']]})
display(comparison)
display(pd.read_csv('reports/cv_003_anidada.csv'))
display(Markdown(f"F1 macro externo, media ± desviación entre folds: **{r['nested_f1_macro_fold_mean']:.5f} ± {r['nested_f1_macro_fold_std']:.5f}**. F1 macro OOF agrupa todas las predicciones externas y puede diferir de la media de folds."))
display(Markdown('La meta 0,90 se alcanza en esta estimación local.' if r['nested_f1_macro_oof']>=0.90 else 'La meta 0,90 NO se alcanza en esta estimación local. No se inventa ni promete ese resultado.'))''')
md('''## Diagnóstico por clase y errores
La selección se basa únicamente en F1 macro. Precisión, recall, soporte y matrices se muestran para interpretar las clases, no para seleccionar otra configuración después de la evaluación externa.''')
code('''display(pd.DataFrame(r['per_class']).T)
a=np.array(r['confusion'])
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,v,title in [(axs[0],a,'Conteos OOF'),(axs[1],a/a.sum(axis=1,keepdims=True),'Proporción por clase real')]:
    ax.imshow(v,cmap='Blues')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=r['labels'],yticklabels=r['labels'],xlabel='Predicción',ylabel='Clase real',title=title)
    for i in range(3):
        for j in range(3): ax.text(j,i,str(v[i,j]) if title=='Conteos OOF' else f'{v[i,j]:.3f}',ha='center')
plt.tight_layout();plt.savefig('reports/confusion_003.png',dpi=160);plt.show()
oof=pd.read_csv('reports/predicciones_oof_003.csv',dtype={'id':str})
train=pd.read_csv('data/train.csv',dtype={'id':str})
errors=oof[oof.label.ne(oof.prediction)].merge(train[['id','text']],on='id',validate='one_to_one')
display(errors.groupby(['label','prediction']).size().rename('errores').to_frame())
for _,row in errors.sample(min(6,len(errors)),random_state=42).iterrows():
    display(Markdown(f'**ID {row.id}: real {row.label}, predicción {row.prediction}.** '+row.text))''')
md('''## Producción y límites
Después de cerrar el procedimiento se reentrena la configuración elegida con todo train. El texto de eval se utiliza solo para producir las predicciones. El CSV mantiene el orden de sample_submission y no copia sus respuestas. Modelo y CSV anteriores se preservan mediante verificación de huellas.

F1 macro CV no es un puntaje de Kaggle. No se accedió a la plataforma ni se realizaron consultas o envíos. El puntaje público de la versión 003 permanece pendiente. Las plantillas o procedencias compartidas no detectadas pueden limitar la independencia de las reseñas; los grupos usados solo cubren equivalencia tras minúsculas y espacios normalizados. Una brecha pequeña tampoco descarta desplazamiento de distribución en prueba.

Para reproducir el entrenamiento desde la raíz: `.venv/bin/python -m src.experimento_003`. Para regenerar este notebook de análisis con kernel limpio: `.venv/bin/python src/notebook_003.py`. Al cargar el modelo conservar `src/features_003.py`, porque joblib utiliza esas clases de transformación.''')
code('''display(r['checks'])
display(r['artifacts_sha256'])
s=pd.read_csv('submissions/submission_003.csv',dtype={'id':str})
display(s.head());display(s.answer.value_counts())
print('Modelo: models/modelo_003.joblib')
print('Envío para cargar manualmente: submissions/submission_003.csv')''')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python del proyecto','language':'python'}})
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(Path.cwd())}}).execute()
n.write(nb,'notebooks/03_modelo_f1_macro.ipynb')
