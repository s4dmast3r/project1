from pathlib import Path
import nbformat as n
nb=n.v4.new_notebook()
c=[]
def md(s): c.append(n.v4.new_markdown_cell(s))
def code(s): c.append(n.v4.new_code_cell(s))
md('''# Primer modelo: sentimiento de reseñas en español
Este experimento compara TF-IDF y regresión logística con una referencia mayoritaria. La métrica de selección es accuracy; F1 macro permite detectar desigualdad entre clases. No se usan datos externos, embeddings, redes neuronales ni servicios de IA para entrenar o predecir.

Reservamos el primer fold de una partición estratificada con grupos como prueba local (~20%). Los grupos se definen únicamente con minúsculas y espacios normalizados, sin borrar filas. La prueba no interviene en la selección. El vocabulario y el IDF se aprenden dentro de cada Pipeline y fold.

Comparamos 32 configuraciones en cinco folds del desarrollo. Antes de ejecutar fijamos una tolerancia de 0,001 de accuracy respecto al máximo: preferimos unigramas, mayor min_df, menor C y TF lineal, en ese orden. Es una regla práctica de simplicidad, no una prueba de equivalencia estadística. Se conserva la negación, los acentos y las palabras vacías; no asumimos que la última oración define el sentimiento.''')
code('''from pathlib import Path
import os, sys
root = Path.cwd()
if not (root / 'data/train.csv').exists():
    root = root.parent
assert (root / 'data/train.csv').exists(), 'Ejecutar desde la raíz o notebooks/'
os.chdir(root)
sys.path.insert(0, str(root))
from src.experimento import run
registro = run()''')
md('''## Auditoría y particiones
Los identificadores sirven únicamente para conservar la correspondencia del envío. Los faltantes de texto se convierten explícitamente en cadenas vacías; un faltante de etiqueta detiene la ejecución. La auditoría de coincidencias con eval no se usa para aprender ni elegir parámetros. Las huellas permiten verificar que los CSV originales permanecieron intactos.''')
code('''import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
pd.set_option('display.max_colwidth', 160)
audit = json.loads(Path('reports/auditoria_datos.json').read_text())
display(audit)
parts = pd.read_csv('reports/particiones.csv')
display(pd.crosstab(parts.partition, parts.label))
display(pd.crosstab(parts.cv_fold, parts.label))
pd.Series(audit['class_counts']).plot.bar(title='Distribución de clases en train', ylabel='Reseñas', rot=0)
plt.tight_layout(); plt.show()''')
md('''## Comparación en validación cruzada
La desviación estándar describe variación entre folds, no un intervalo de confianza. El DummyClassifier aprende la clase mayoritaria en cada entrenamiento. No balanceamos clases automáticamente: el desbalance observado se documenta y las métricas por clase permiten examinar sus consecuencias.''')
code('''cv = pd.read_csv('reports/resultados_cv.csv')
m = json.loads(Path('reports/evaluacion_local.json').read_text())
cols = ['param_tfidf__ngram_range','param_tfidf__min_df','param_tfidf__sublinear_tf','param_clf__C','mean_test_accuracy','std_test_accuracy','mean_test_f1_macro','std_test_f1_macro']
with pd.option_context('display.max_rows', 40):
    display(cv[cols].sort_values('mean_test_accuracy', ascending=False))
display(pd.DataFrame({'seleccionado':m['cv']}))
display(m['dummy_cv'])
display(m['selected_params'])
gram_summary = cv.groupby('param_tfidf__ngram_range').mean_test_accuracy.agg(['min','mean','max'])
display(gram_summary)
# Comparación emparejada: se mantienen los otros hiperparámetros.
pairs = cv.pivot(index=['param_tfidf__min_df','param_tfidf__sublinear_tf','param_clf__C'], columns='param_tfidf__ngram_range', values='mean_test_accuracy')
pairs['delta_bigramas'] = pairs['(1, 2)'] - pairs['(1, 1)']
display(pairs)
display(Markdown(f"La diferencia media emparejada al incorporar bigramas es **{pairs.delta_bigramas.mean():.4f}**; su rango es {pairs.delta_bigramas.min():.4f} a {pairs.delta_bigramas.max():.4f}. Esta evidencia se limita a estas particiones y configuraciones."))''')
md('''## Prueba local final: una sola evaluación
Tras seleccionar por CV, ajustamos sobre desarrollo y evaluamos una vez la prueba reservada. A partir de aquí no cambiamos parámetros. La evaluación de Kaggle sigue pendiente y no puede deducirse de esta prueba.''')
code('''display(pd.DataFrame({'regresion_logistica':{'accuracy':m['accuracy'],'f1_macro':m['f1_macro']},'mayoritaria':m['dummy_local']}))
display(pd.DataFrame(m['per_class']).T)
fig, axes = plt.subplots(1,2,figsize=(11,4))
for ax, key, title in zip(axes,['confusion_counts','confusion_normalized'],['Conteos','Proporción por clase real']):
    a = np.asarray(m[key]); ax.imshow(a,cmap='Blues',vmin=0)
    ax.set(xticks=range(3),yticks=range(3),xticklabels=m['labels'],yticklabels=m['labels'],xlabel='Predicción',ylabel='Clase real',title=title)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(a[i,j]) if key.endswith('counts') else f'{a[i,j]:.3f}',ha='center',va='center')
plt.tight_layout(); plt.savefig('reports/matrices_confusion.png',dpi=160); plt.show()''')
md('''## Aciertos, errores, negación y contraste
La selección de ejemplos usa semilla 42, hasta cuatro por categoría. Los marcadores de negación y contraste son reglas descriptivas posteriores a la evaluación, nunca predictores adicionales. Un marcador de contraste no demuestra sentimiento mixto: debe leerse el texto. Las etiquetas son las del conjunto, no juicios nuevos del estudiante.''')
code('''cases = pd.read_csv('reports/predicciones_locales.csv')
examples = pd.read_csv('reports/ejemplos.csv')
errors = cases[~cases.correct]
display(errors.groupby(['label','prediction']).size().rename('errores').sort_values(ascending=False).to_frame())
for marker in ['negation_marker','contrast_marker']:
    display(cases.groupby(marker).correct.agg(['count','mean']).rename(columns={'mean':'accuracy_descriptiva'}))
for category, rows in examples.groupby('category',sort=False):
    display(Markdown('### '+category))
    for _, r in rows.iterrows():
        display(Markdown(f"**ID {r.id} — real: {r.label}; predicción: {r.prediction}.** {r.text}"))
        relation = 'acierto' if r.correct else f'confusión de {r.label} con {r.prediction}'
        display(Markdown(f"Se observa un {relation}. Marcador de negación: {r.negation_marker}; marcador de contraste: {r.contrast_marker}. Estos indicadores describen el texto, pero no explican causalmente la decisión."))''')
md('## Lectura de los ejemplos observados\n\nEn los errores seleccionados, el ID 11279 combina «valió la pena» para la cámara con arrepentimiento sobre la relación calidad-precio; el ID 4550 combina una terminación fiable con una pantalla de mala calidad. Ambos tienen etiqueta negativa y predicción positiva. En el ID 1062 ocurre la dirección contraria: el texto combina arrepentimiento por el servicio y satisfacción con la carga, con etiqueta positiva y predicción negativa. La coexistencia de valoraciones opuestas es observable, pero no permite demostrar qué expresión causó cada decisión. Tampoco permite deducir una regla sobre la última oración.\n\nEl ID 1121 muestra que no todos los errores requieren sentimiento mixto: contiene una valoración de comodidad «superior», pero la predicción fue negativa. Por su parte, los aciertos 7383 y 5726 también mezclan aspectos favorables y desfavorables. Por tanto, la mezcla no determina por sí sola el éxito o fracaso del clasificador.\n\nEntre las confusiones con neutral, el ID 4302 contiene «malísimo» para el cable y el 10816 dice que el teclado es lo que buscaba; ambos se predijeron neutrales pese a sus etiquetas polarizadas. La descripción factual que acompaña esas valoraciones es visible, pero atribuirle causalmente el fallo requeriría un análisis adicional que aquí no se realiza.\n\nLas negaciones cumplen funciones distintas: en el ID 9212 «no una sola» precisa la cantidad y la clasificación neutral fue correcta; en el ID 11406 «no me dio ninguna vuelta» acompaña una experiencia positiva correctamente clasificada. Esto justifica conservar las negaciones y también muestra por qué su sola presencia no define la polaridad.\n\nLa tabla de confusiones evidencia que el problema principal de esta prueba es separar positivo de negativo. El buen rendimiento en neutral no debe ocultar el F1 inferior de ambas clases polarizadas. Las tasas de los subgrupos con marcadores son descriptivas y no aíslan efectos de negación o contraste.\n')
md('''## Asociaciones aprendidas
Los coeficientes corresponden al modelo ajustado únicamente sobre desarrollo, antes del reentrenamiento final. Un coeficiente positivo favorece el logit de esa clase, condicionado por el resto de términos. No demuestra causalidad ni garantiza una predicción aislada. Si se seleccionan unigramas, el modelo no tiene coeficientes de bigramas; no se inventan.''')
code('''coef = pd.read_csv('reports/coeficientes.csv')
for (label, kind, direction), rows in coef.groupby(['clase','tipo','direccion']):
    display(Markdown(f'**{label} · {kind} · coeficientes {direction}s**'))
    display(rows[['termino','coeficiente']].reset_index(drop=True))
if 'bigrama' not in set(coef.tipo):
    print('La configuración seleccionada no incluye bigramas.')''')
md('''## Limitaciones y próximos experimentos
TF-IDF representa presencia y frecuencia local, pero no modela completamente alcance de negación, ironía ni cuál aspecto domina cuando una reseña mezcla valoraciones. Los grupos evitan fuga por duplicados normalizados, aunque pueden quedar paráfrasis similares en particiones distintas. Una sola partición local no mide toda la incertidumbre ni garantiza generalización al leaderboard. Los subgrupos descritos no son experimentos causales.

Como trabajos futuros podrían compararse n-gramas de caracteres, otros clasificadores lineales y análisis de estabilidad con nuevas particiones definidas de antemano. **No se ejecutan aquí**, ni se reajusta usando los errores locales. El valor histórico 0.91555 del leaderboard no es comparable directamente con estas métricas. Según el prompt, el leaderboard público usa aproximadamente 30% y el privado 70%; no se consultó Kaggle.

## Modelo final y envío
Se reentrena la configuración elegida con todo train; eval solo aporta texto para predecir. Las respuestas se asocian por los IDs de eval y se ordenan como sample_submission. Sus respuestas de ejemplo no se copian. Se comprueban esquema, cantidad, orden, unicidad, etiquetas, recarga del modelo, lectura del CSV y huellas originales.''')
code('''display(registro['checks'])
display(registro['versions'])
display(registro['artifacts_sha256'])
submission = pd.read_csv('submissions/submission_001.csv',dtype={'id':str})
display(submission.head())
display(submission.answer.value_counts().rename('predicciones'))
print('Modelo: models/modelo_001.joblib')
print('Primer envío: submissions/submission_001.csv')
print('Puntaje Kaggle: pendiente; no se realizó ningún envío.')''')
nb.cells=c
nb.metadata={'kernelspec':{'display_name':'Python (entorno del proyecto)','language':'python','name':'python3'},'language_info':{'name':'python'}}
n.write(nb,'notebooks/01_modelo_inicial.ipynb')
