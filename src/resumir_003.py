from pathlib import Path
import json
import pandas as pd
from src.experimento import sha, dump
r=json.loads(Path('reports/registro_experimento_003.json').read_text())
folds=pd.read_csv('reports/cv_003_anidada.csv')
cv=pd.read_csv('reports/resultados_cv_003.csv')
assert len(cv)==9 and len(folds)==3
assert cv.mean_test_score.notna().all()
for p,v in r['artifacts_sha256'].items():assert sha(p)==v
source={str(p):sha(p) for p in [Path('src/features_003.py'),Path('src/experimento_003.py'),Path('reports/particiones.csv')]}
r['source_and_partitions_sha256']=source
r['exploration_history']='Comparación inicial de 7 configuraciones archivada; ejecución interrumpida antes de completar el primer fold externo. Se añadió representación de cláusulas y se reinició el protocolo con 9 configuraciones. No se reajustó después de conocer los resultados externos finales.'
dump('reports/registro_experimento_003.json',r)
summary=f'''# Versión 003 — F1 macro

Configuración seleccionada: `{r['selected_params']}`.

| Medida de F1 macro | Valor |
|---|---:|
| CV usada para selección final | {r['selected_cv_f1_macro']:.6f} |
| Entrenamiento de esa CV | {r['selected_train_f1_macro']:.6f} |
| Brecha entrenamiento-validación | {r['selected_f1_gap']:.6f} |
| Predicciones externas agrupadas, CV anidada 003 | {r['nested_f1_macro_oof']:.6f} |
| Predicciones externas agrupadas, referencia 002 | {r['nested_baseline002_f1_macro_oof']:.6f} |
| Media de F1 en los folds externos | {r['nested_f1_macro_fold_mean']:.6f} |
| Desviación estándar entre folds externos | {r['nested_f1_macro_fold_std']:.6f} |

La comparación local emparejada mejora F1 macro en {r['nested_f1_macro_oof']-r['nested_baseline002_f1_macro_oof']:.6f}. El puntaje público de 002 aportado por el usuario fue 0,73888; no es la misma evaluación y no se usa como punto de referencia para calcular esa mejora.

Meta local de 0,90: {'alcanzada en OOF anidada' if r['nested_f1_macro_oof']>=0.90 else 'no alcanzada en OOF anidada'}. Esto no anticipa el resultado público ni privado. El puntaje de plataforma del 003 está pendiente.

## Separación y riesgo de sobreajuste

Solo desarrollo intervino en selección y CV anidada. Cada fold externo vuelve a seleccionar parámetros con CV interna, y sus textos no entran al ajuste de vocabulario, IDF o clasificador. Las particiones se agrupan por equivalencia de texto normalizado. La antigua prueba local no se volvió a evaluar. Después de cerrar selección se usa todo train para el modelo de producción. eval solo se usa para predecir.

La brecha de F1 sigue siendo un diagnóstico, no una garantía. La selección prioriza F1 y usa brecha para alternativas casi equivalentes. No hay garantía de mínimo sobreajuste global. Desarrollo ya fue usado para diseño y exploración; los resultados anidados controlan selección de hiperparámetros dentro de folds, pero no eliminan toda adaptación humana al conjunto. No se dispone de una nueva prueba externa etiquetada. Tampoco se puede descartar dependencia por plantillas o por origen no documentado.

## Ejecución y trazabilidad

{r['exploration_history']}

Se completaron 9 configuraciones con 3 folds para selección final, y 3 búsquedas internas de 9 configuraciones x 3 folds para evaluación externa. Los reportes contienen todas las puntuaciones. El log registra una advertencia del gestor de procesos si un trabajador se reinició; las verificaciones comprueban resultados completos, sin puntuaciones faltantes. No se usaron redes neuronales, APIs de IA, datos externos ni ejemplos sintéticos.

## Envío

Archivo: `submissions/submission_003.csv`. Modelo: `models/modelo_003.joblib`. Conservar `src/features_003.py` para cargar el Pipeline. Se verificaron 3000 IDs, orden, esquema, etiquetas válidas, lectura del CSV, recarga del modelo y conservación de los originales. La versión 003 está preparada para el segundo envío manual; no se subió automáticamente.
'''
Path('reports/resumen_003.md').write_text(summary)
print(summary)
