# Clasificación de sentimiento: primer experimento

Implementación tradicional en scikit-learn para `positivo`, `negativo` y `neutral`. Los CSV originales se mantienen intactos. No se usaron datos externos ni se accedió a Kaggle.

## Reproducir

Desde la raíz `Proyect1`, con Python 3.14 (versión exacta registrada en el JSON):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/ejecutar_notebook.py
```

El ejecutor abre un kernel limpio con el Python del entorno, ejecuta el notebook completo y guarda sus salidas. También puede abrirse el notebook con un kernel que use `.venv/bin/python`. Las rutas son relativas a la raíz, que el notebook detecta también desde `notebooks/`. La ejecución reproduce y reemplaza **solo los artefactos generados**; guardar una copia si se desea conservar otro experimento. Como alternativa sin notebook: `.venv/bin/python src/experimento.py`. No es necesario ejecutar ambos.

## Diseño

Semilla 42. Primera partición de StratifiedGroupKFold de cinco folds como prueba local (~20%), y cinco folds sobre desarrollo para comparar 32 configuraciones. Los textos equivalentes tras convertir a minúsculas y normalizar espacios pertenecen al mismo grupo. No se eliminan filas. Los IDs nunca se usan como predictores ni para unir train con eval.

TF-IDF y regresión logística forman un Pipeline: cada fold aprende su propio vocabulario e IDF. Limpieza mínima, faltantes como texto vacío, sin eliminar negaciones, palabras vacías ni acentos. Matrices dispersas, optimizador lbfgs multiclase, máximo 2500 iteraciones, dos procesos y un hilo numérico por proceso.

Selección por accuracy media, con tolerancia previamente fijada de 0.001 respecto al máximo; entre candidatos equivalentes se prefieren unigramas, min_df mayor, C menor y TF lineal, en ese orden. Es un criterio de simplicidad, no significancia estadística. Se evalúa una sola vez la configuración seleccionada en la prueba local y luego se reentrena con todo train sin modificar sus parámetros.

## Archivos

- `notebooks/01_modelo_inicial.ipynb`: explicación en español, ejecución completa, tablas, ejemplos y matrices de confusión.
- `models/modelo_001.joblib`: Pipeline final completo. Cargar únicamente artefactos joblib de confianza.
- `submissions/submission_001.csv`: único envío, columnas `id,answer`, orden de sample_submission; listo para carga manual.
- `reports/resultados_cv.csv`: resultados de las 32 configuraciones y los cinco folds.
- `reports/evaluacion_local.json`: métricas de CV, referencia mayoritaria y prueba local.
- `reports/registro_experimento_001.json`: parámetros, versiones, semilla, huellas SHA-256 y verificaciones.
- `reports/auditoria_datos.json`, `particiones.csv`: auditoría y asignaciones reproducibles.
- `reports/predicciones_locales.csv`, `ejemplos.csv`, `coeficientes.csv`: evidencia para el análisis. Los coeficientes son del ajuste en desarrollo.
- `reports/matrices_confusion.png`: conteos y proporciones por clase real.
- `requirements.txt`: versiones del entorno aislado; `uso_ia.md`: declaración y prompt.

Las cifras exactas se generan en los reportes y notebook, sin duplicarlas manualmente aquí. CV, prueba local y Kaggle son evaluaciones diferentes. El puntaje de Kaggle está pendiente. No se prometen resultados a partir del referente histórico del leaderboard.

## Revisión adicional de sobreajuste

Ver `reports/revision_sobreajuste.md` y `reports/auditoria_sobreajuste_folds.csv`. Para reproducir exclusivamente la auditoría: `.venv/bin/python -m src.auditar_validacion`. Mantiene parámetros, modelo y envío originales; diagnostica la brecha entrenamiento-validación con cinco ajustes sobre desarrollo y no reevalúa la prueba local. La brecha observada es sustancial y no debe confundirse con fuga de información.

## Versión corregida para reducir sobreajuste

El archivo recomendado ahora es `submissions/submission_002_regularizado.csv`, generado por `models/modelo_002_regularizado.joblib`. El número 002 representa una versión local, no un envío realizado. Se conservan los originales.

Ver `reports/correccion_sobreajuste_002.md`, `reports/registro_experimento_002.json` y el notebook ejecutado `notebooks/02_regularizacion.ipynb`. La nueva selección usa exclusivamente CV de desarrollo y no reevalúa la prueba local ya observada. No atribuir al modelo nuevo la accuracy de prueba del anterior.

Reproducir desde la raíz:

```bash
.venv/bin/python -m src.regularizar_modelo
.venv/bin/python src/notebook_regularizacion.py
```

La corrección reduce la complejidad del vocabulario; la brecha observada disminuye, pero no desaparece. Sus métricas de CV son de selección, no una evaluación independiente.

## Cambio de criterio para la versión 003

Por instrucción del usuario, desde la versión 003 la selección y evaluación principal usan exclusivamente **F1 macro**. Los informes anteriores conservan sus métricas históricas. El puntaje público del 002 aportado por el usuario es 0,73888; la captura no muestra el nombre de la métrica. No se accedió a la plataforma.

La versión 003 compara n-gramas de palabras, caracteres y posición/contexto con LinearSVC y la referencia 002. La selección y la comparación externa usan CV anidada dentro del desarrollo original, excluyendo la antigua prueba local. Las características usan solo texto, nunca IDs ni etiquetas como predictores. No se garantiza F1 0,90.

Reproducir desde la raíz:

```bash
.venv/bin/python -m src.experimento_003
.venv/bin/python src/notebook_003.py
```

Los resultados ejecutados quedan en `reports/registro_experimento_003.json`, `reports/resultados_cv_003.csv`, `reports/cv_003_anidada.csv` y `notebooks/03_modelo_f1_macro.ipynb`. Modelo: `models/modelo_003.joblib`; envío: `submissions/submission_003.csv`. Al cargar joblib, conservar el módulo `src/features_003.py` y ejecutar desde la raíz o incluirla en PYTHONPATH.

La versión 003 y su CSV ya están ejecutados y verificados. Consultar `reports/resumen_003.md` para la comparación emparejada y las limitaciones; el objetivo 0,90 no se alcanzó. El F1 en entrenamiento es casi perfecto y la brecha persiste: la mejora de F1 externo no equivale a eliminar el sobreajuste. Para regenerar el resumen: `.venv/bin/python -m src.resumir_003`.

## Versión 004 — ablaciones del canal discursivo

Tras una auditoría técnica (ver `reports/auditoria_y_resumen_004.md`), se ejecutó una
ronda acotada de 8 configuraciones sobre el canal discursivo del 003, con el mismo
protocolo de CV (3-fold semilla 42) y OOF (mismos 3 folds externos del 003, semilla
137). Métrica exclusiva: F1 macro con `zero_division=0`.

La ganadora es `ng2`: igual que 003 (discourse, C=2, min_df=3, canal de caracteres
con peso 0.5) pero con n-gramas posicionales limitados a (1–2) en lugar de (1–3).
Es una simplificación (menos características) que mejora el F1 macro OOF de 0.835400
a 0.850578 y reduce la brecha entrenamiento-validación de 0.166 a 0.149. El 0,90 no
se alcanzó y el sobreajuste persiste; la mejora es de comparación emparejada, no una
evaluación virgen.

Reproducir desde la raíz:

```bash
.venv/bin/python -m src.experimento_004
```

Modelo: `models/modelo_004.joblib`; envío recomendado: `submissions/submission_004.csv`
(conserva los previos). Al cargar joblib, conservar `src/features_004.py` y ejecutar
desde la raíz. El puntaje público del 004 está pendiente; el usuario realiza el envío
manual.

## Versión 005: revisión de 004 y simplificación

La captura aportada por el usuario reporta 0,86111 para 004; se conserva en `reports/resultado_publico_004.json` sin alterar el registro histórico de entrenamiento. La revisión del auditor está en `reports/revision_auditor_004_para_005.md`.

La ronda 005 usa F1 macro exclusivamente, seis configuraciones, restricción de brecha y CV anidada 3x3. La selección se repite dentro del entrenamiento de cada fold externo; no se elige directamente por el resultado externo como ocurrió en 004. El desarrollo ya explorado sigue sin ser una prueba nueva.

Reproducir desde la raíz (estos comandos regeneran los artefactos 005; conservar copias si se necesita historial):

```bash
.venv/bin/python -m src.experimento_005
.venv/bin/python src/informe_005.py
```

Consultar `reports/registro_005.json`, `reports/resumen_005.md` y `notebooks/05_modelo_f1_macro.ipynb`. El código solo crea `models/modelo_005.joblib` y `submissions/submission_005.csv` si la selección interna cumple el criterio previo de mejora y brecha. El modelo reutiliza `src/features_004.py`, que debe conservarse para cargar tanto 004 como 005. No se modifica el código de los modelos anteriores.

La ejecución 005 finalizó y creó el CSV nuevo; las verificaciones adicionales en un proceso limpio constan en `reports/verificacion_final_005.json`. El score público de 005 permanece pendiente.

## Versión 006: contraste de recomendaciones

El público 005 confirmado por el usuario fue 0,88111 (`reports/resultado_publico_005.json`). La revisión de recomendaciones y del riesgo de fuga indirecta en stacking está en `reports/revision_recomendaciones_006.md`.

Se comparan referencia 005, C=1/min_df=5, ausencia de interacciones de epílogo y stacking SVC/LR/NB con meta LogisticRegression. CV externa e interna de tres folds; cada stacking genera a su vez scores OOF solo en las filas de su propio fit. No se reutilizan matrices OOF globales para ajustar meta-modelos dentro de validación.

Reproducir desde raíz (regenera artefactos 006; conservar copias si se requiere historial):

```bash
.venv/bin/python -m src.experimento_006
.venv/bin/python src/informe_006.py
```

Ver `reports/registro_006.json`, `reports/resumen_006.md` y `notebooks/06_modelo_f1_macro.ipynb`. Si se crea el modelo nuevo, requiere `src/modelos_006.py` y sus dependencias locales para recarga. Los modelos anteriores y sus módulos se conservan. El resultado público de 006 está pendiente.
