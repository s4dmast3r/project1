# Ronda011b: revisión de010 y alternativas clásicas

Accuracy es el criterio principal y F1 macro es un control, por confirmación explícita del usuario el9 de octubre de2026. Público010: **0,89777**. Público011: desconocido.

## Qué mejoró en010

La entrega010 corresponde al experimento013: conserva TF-IDF de discurso y posición, y LinearSVC L1, pero cambia C=1 por C=2. La menor regularización elevó la accuracy OOF media de0,895660 a0,899062 y la brecha de0,060799 a0,088976. El modelo pasó de3252 a5491 coeficientes no nulos. Mejoró el público de0,89555 a0,89777, a costa de más ajuste al entrenamiento. Los hashes y la recarga del modelo010 se verificaron antes de comparar.

## Investigación e implementación

Ver [investigacion_011b.md](investigacion_011b.md) para fuentes primarias y decisiones. Se implementaron presencia binaria con/sin IDF, TF sublineal sin IDF, identidad de conectores con L1 y regresión logística L1. No se incorporaron datos externos. La ronda011 previa se preserva.

## Resultados de selección

| Candidata | Accuracy media OOF | F1 macro medio OOF | Brecha accuracy | Folds ganados vs010 |
|---|---:|---:|---:|---:|
| ref010 | 0.899062 | 0.903358 | 0.088976 | — |
| binary_idf | 0.898611 | 0.902981 | 0.090000 | 4/9 |
| connector_l1 | 0.898333 | 0.902623 | 0.090295 | 3/9 |
| binary_noidf | 0.893403 | 0.897798 | 0.083611 | 0/9 |
| tf_noidf | 0.893056 | 0.897346 | 0.080347 | 0/9 |
| logistic_l1 | 0.892743 | 0.896763 | 0.065799 | 0/9 |

Semillas8803/9901/11003, tres folds agrupados por semilla. Los puntajes OOF se calculan sobre las9600 predicciones de desarrollo de cada semilla y luego se promedian. La brecha promedia accuracy de entrenamiento menos validación por fold.

## Criterio fijado antes de entrenar

>=7/9 folds accuracy; delta medio accuracy >=0.003; delta F1 medio >=0; brecha accuracy media no mayor; no perder accuracy ni F1 en137.

La comparación137 es otra puerta de selección, no una prueba independiente.

## Decisión

Ganadora: **ninguna**. CSV creado: **False**.

Conservar010: ninguna candidata supera los criterios. No crear un envío equivalente ni bajar umbrales a posteriori.

## Límites y controles

No se usó el test local histórico para elegir ni eval para ajustar. Todos los vocabularios, IDF y coeficientes se aprendieron dentro del entrenamiento de cada fold. Grupos de texto normalizado disjuntos; esa agrupación no garantiza detectar duplicados semánticos. Convergencia exigida. Artefactos previos preservados por hash.

Los datos de desarrollo se han explorado repetidamente. Estos scores sirven para selección y pueden ser optimistas; las nuevas semillas no crearían datos vírgenes. Los9folds no son independientes ni se presenta un test de significación a partir de contar victorias. Menor brecha es un diagnóstico, no una garantía de ausencia de sobreajuste.

Que una métrica local coincida numéricamente con un score público no demuestra cuál es la métrica oficial. Aquí accuracy se usa por instrucción directa del usuario. Tampoco los experimentos negativos previos prueban un techo absoluto para ML clásico. No se promete0,90 público.

## Reproducibilidad

Tiempo de experimento: 549.8 segundos. Artefactos anteriores intactos: True.

Entrenamiento: `.venv/bin/python -m src.experimento_011b`. La ejecución se detiene si ya existe el registro o el destino del modelo/envío. Informe y notebook: `.venv/bin/python -m src.informe_011b`. El notebook analiza resultados guardados, no vuelve a entrenar.

Archivos de auditoría: `plan_011b.json`, `registro_011b.json`, `folds_011b.csv`, `resultados_011b.csv`, `criterios_011b.csv` y `predicciones_oof_011b.csv`.
