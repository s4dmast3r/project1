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

## Ronda 007: plan de estabilidad y características lingüísticas

Implementa `reports/revision_006_y_plan_007.md` con referencia005, neg_scope, post_contraste y C=1/min_df=5. Se ejecutan tres folds para cada semilla271/599/913. Los scores son de selección por CV repetida, no una prueba independiente. La271 ya se usó dentro del stack006; cambiar su papel no vuelve vírgenes los datos.

Se exige mejora mínima0,005 en cada semilla y estabilidad entre folds; solo las candidatas aprobadas se comparan con la referencia en137. No se produce un CSV si ninguna cumple. Se conservan todos los artefactos previos, incluida006 para trazabilidad;005 es el mejor público confirmado.

```bash
.venv/bin/python -m src.experimento_007
.venv/bin/python src/informe_007.py
```

El entrenamiento se detiene si ya existe `reports/registro_007.json`: archivar explícitamente la ejecución anterior antes de repetir. Ver `reports/resumen_007.md`, `reports/criterios_007.csv` y `notebooks/07_modelo_f1_macro.ipynb`. No se interpreta el aproximadamente0,87 de006 referido en el documento como un score exacto comprobado.

Resultado007: ninguna alternativa aprobó los criterios; no se generó modelo ni submission007. Conservar005 como recomendación. Los36 ajustes y el notebook quedaron ejecutados y verificados.

## Versión 007: tercios del texto completo sobre 005

Tras la ronda 007 (sin ganadora) se ejecutó la ronda 007b con cambios estructurales
del espacio de características: trigramas globales, bucket medio, bigramas de borde
y tercios del texto completo. Las tres primeras empeoran; **thirds** (etiqueta
`third0/1/2` según el tercio de la reseña donde aparece cada palabra) ganó 9/9 folds
emparejados frente a 005 en tres semillas (media +0,0028, brecha menor), pero por
debajo del umbral +0,005 de esa ronda, que concluyó «conservar 005».

Para no relajar el criterio después de ver resultados, se fijó una confirmación
pre-registrada sobre semillas nunca usadas (1142, 2087, 3313) y la puerta histórica
137: ganar ≥7/9 folds, delta medio ≥0,002, brecha no peor y no perder en 137.
Resultado: **9/9 folds, +0,00328 de media, brecha −0,00324, y 0,877822 vs 0,873382 en
137**. Aprobada: se generó `models/modelo_007.joblib` y
`submissions/submission_007.csv` (cambia 68 de 3000 predicciones respecto a 005).

Reproducir desde la raíz:

```bash
.venv/bin/python -m src.experimento_007b   # exploración estructural
.venv/bin/python -m src.confirmar_007b     # confirmación pre-registrada + CSV
```

Ver `reports/resumen_007b.md`, `reports/registro_007b.json` y
`reports/registro_confirmacion_007b.json`. El modelo requiere `src/features_007b.py`
para recarga. La mejora local es pequeña y consistente, pero no garantiza el público;
el envío manual lo decide el usuario. El público 005 confirmado sigue siendo 0,88111
y el de 006 quedó reportado como aproximadamente 0,87 sin captura exacta.

## Ronda 008: ponderación del bloque de tercios

Implementa el subconjunto rápido de `reports/revision_007_ideas_008.md`: referencia
007, control del split (base/thirds en vectorizadores separados a peso 1.0) y pesos
de tercios 0.5 y 1.5. Criterio pre-registrado sobre semillas nuevas 711/1237/2003
más puerta 137.

Resultado: ninguna candidata aprobó. El control `split_w1` es peor que la 007 en las
tres semillas (−0.0047 de media) y `split_w15` empeora (−0.0133); `split_w05` es
neutro. Esto indica que la ganancia de la 007 proviene de la normalización/IDF
conjunta del TF-IDF único, no del valor aislado de los tercios; re-ponderar el bloque
no es una palanca útil. Se conserva 007 (público 0.88888).

```bash
.venv/bin/python -m src.experimento_008
```

Ver `reports/resumen_008.md` y `reports/registro_008.json`. No se generó
`submission_008.csv`; los artefactos 001–007 permanecen intactos.

## Ronda 008b: investigación web y NB-SVM

Se preservó la exploración 008 anterior y se ejecutaron cinco alternativas nuevas
frente a la referencia 007: pesos de tercios en el mismo TF-IDF (0,5 y 1,5),
identidad de conectores, selección chi² del 50 % y NB-SVM multiclase. Se consultaron
el artículo de Wang y Manning (ACL 2012) y documentación oficial de scikit-learn;
las fuentes y las adaptaciones están en `reports/investigacion_008b.md`.

Ganó **nbsvm**. F1 macro OOF medio en las mismas tres particiones:
007 **0.877286**, NB-SVM **0.892841**;
9/9 folds favorables. Brecha media: 0.122364 → 0.107165.
También superó la puerta histórica 137. Se creó `submissions/submission_008.csv`
y `models/modelo_008.joblib`. Son resultados de selección sobre datos ya explorados,
no estimaciones independientes ni garantía de 0,90 público. El entrenamiento de
NB-SVM alcanza F1=1 y sigue existiendo riesgo de sobreajuste.

Corrección de la interpretación de 008: separar vectorizadores no cambia el IDF de
un rasgo común con igual frecuencia documental; sí altera la normalización por
bloques. Aquella comparación no demostraba que cualquier ponderación fuera inútil.
Las semillas 711/1237/2003 se reutilizan deliberadamente y no se consideran prueba
nueva. El test local antiguo se excluyó de selección. Los transformadores, chi² y
razones NB se aprenden dentro de cada entrenamiento. Eval solo se usa al exportar.

Reproducción: `.venv/bin/python -m src.experimento_008b` (protege el registro
existente). Informe: `.venv/bin/python -m src.informe_008b`.
Consultar `reports/resumen_008b.md`, `reports/verificacion_008b.json` y el notebook
 ejecutado `notebooks/08b_modelo_f1_macro.ipynb`. No se realizó envío a la plataforma.

## Versión 009: regularización L1 sobre 007

Tras la ronda 008 (ponderación de tercios, sin ganadora) se probaron estrategias
distintas: LinearSVC con regularización L1 (C=1, dual=False), canal de caracteres
`char_wb` 2-6 y NB-SVM sobre palabras. La ganadora es **L1 sobre las mismas
características de la 007**: reduce los pesos no nulos a ~2,2 % del total, baja la
brecha de 0.123 a 0.058 y gana 9/9 folds emparejados frente a 007 en tres semillas
nuevas (1009/2027/3053) con delta medio +0.0239, además de superar la puerta
histórica 137 (0.89909 vs 0.87782). Se generó `models/modelo_009.joblib` y
`submissions/submission_009.csv` (cambia 214 de 3000 predicciones).

```bash
.venv/bin/python -m src.experimento_009
```

Ver `reports/resumen_009.md` y `reports/registro_009.json`. El modelo requiere
`src/features_007b.py` para recarga. Es la primera evaluación local honesta en
~0.90, sin garantizar el público; el envío manual lo decide el usuario.

## Rondas 010 y 011: la 009 es un plateau

Público 009 confirmado: **0,89555**. Se probaron: barrido de C alrededor de la L1
(010) — C=1 óptimo y el control L2 C=1 retrocede, aislando la penalización L1 como
causa —, complementariedad con el NB-SVM 008b (no ayuda: voto por rangos 0.8897) y
re-ensayo bajo L1 de rasgos que fallaron bajo L2 (011: trigramas, bigramas de borde,
char_wb, min_df=2; todos neutros). Ninguna candidata superó las puertas
pre-registradas; se conserva 009.

```bash
.venv/bin/python -m src.experimento_010
.venv/bin/python -m src.experimento_011
```

Ver `reports/resumen_010_011.md`, `reports/registro_010.json`,
`reports/registro_011.json`. No se generaron envíos 010/011; `submission_009.csv`
sigue siendo el recomendado.

## Diagnóstico de polaridad y ronda 012

El diagnóstico confirmó que los errores positivo↔negativo de la 009 son reseñas de
aspecto mixto (más negación, contraste y marcas de opinión que los aciertos). Se
probaron rasgos densos de sentimiento (conteos de negación/contraste/cláusulas y un
léxico interno NB aprendido dentro del fold): neutros o peores. Ninguna candidata
superó las puertas; se conserva 009 y no se genera submission_010.

```bash
.venv/bin/python -m src.experimento_012
```

Ver `reports/resumen_012.md`, `reports/registro_012.json`. `submission_009.csv`
sigue siendo el recomendado. Pendiente del usuario: confirmar la métrica oficial
compartiendo el texto de las reglas.

## Versión 010: métrica oficial = accuracy

El usuario confirmó que la evaluación es **accuracy** (neutral algo menos frecuente).
Re-evaluando la 009 bajo accuracy, L1 C=2 gana a C=1: +0.0034 de accuracy media en
semillas nuevas (8803/9901/11003), 8/9 folds y puerta 137 (0.89771 vs 0.89500);
`class_weight='balanced'` empeora. Se generó `models/modelo_010.joblib` y
`submissions/submission_010.csv` (cambia 112 de 3000 predicciones). La brecha sube de
0.06 a 0.089 (memoriza más): ganancia modesta, pública no garantizada.

```bash
.venv/bin/python -m src.experimento_013
```

Ver `reports/resumen_013.md` y `reports/registro_013.json`.

## Continuación011b, 9 de octubre de2026

La captura del usuario confirma público010=0,89777. La entrega010 proviene del experimento013 (L1 C=2); los antiguos registros010/011 son rondas distintas sin aprobación. Se verificaron fuentes/modelo/CSV de010.

El usuario confirmó en este chat: «Accuracy como principal y F1 macro como control». Se consultaron fuentes primarias de scikit-learn y COLING, y se implementaron cinco alternativas: binary_idf, binary_noidf, tf_noidf, connector_l1 y logistic_l1. Se reutilizan particiones8803/9901/11003 para comparación con010; no son datos nuevos. Todos los aprendizajes quedan dentro del fold. No se accedió a Kaggle ni se añadieron datos externos.

| Modelo | Accuracy media | F1 macro medio | Brecha accuracy |
|---|---:|---:|---:|
| ref010 | 0.899062 | 0.903358 | 0.088976 |
| binary_idf | 0.898611 | 0.902981 | 0.090000 |
| connector_l1 | 0.898333 | 0.902623 | 0.090295 |
| binary_noidf | 0.893403 | 0.897798 | 0.083611 |
| tf_noidf | 0.893056 | 0.897346 | 0.080347 |
| logistic_l1 | 0.892743 | 0.896763 | 0.065799 |

Conservar010: ninguna candidata supera los criterios. No crear un envío equivalente ni bajar umbrales a posteriori.

Los resultados son de selección adaptativa acumulada, no evaluación externa independiente. No se concluye un techo absoluto de rendimiento a partir de esta ronda ni se promete0,90 público. Se conservan los experimentos previos.

Reproducción: `.venv/bin/python -m src.experimento_011b` (bloquea sobrescritura del registro existente), `.venv/bin/python -m src.verificar_011b` y `.venv/bin/python -m src.informe_011b`. Ver reports/resumen_011b.md, reports/investigacion_011b.md, reports/verificacion_011b.json y notebooks/11b_revision_y_modelo.ipynb.

## Versión 011: corrección de prior (descuento de neutral)

Sobre la 010 (L1 C=2) se aplica un descuento de la clase neutral (d=0.2) en la regla
de decisión, porque neutral es la clase menos frecuente y la métrica es accuracy.
Ganancia local +0.0006..0.0009 (consistente), 4 de 3000 predicciones de eval cambian.
Ver `reports/resumen_011.md` y `reports/registro_produccion_011.json`. Reproducir:
`.venv/bin/python -m src.producir_011`.

## Continuación012b, 9 de octubre de2026

La captura011 registra score0,89888. La entrega011 proviene de producir_011.py: mismos coeficientes que010 y ajuste de0,2 a los márgenes positivo/negativo; cambia4 predicciones. Se verificaron hashes y reproducción. La nueva referencia incorpora la regla completa en predict(). No se modifica el .joblib anterior, que guarda la base sin corrección.

Se investigaron y ejecutaron especialistas binarios de polaridad C=1/C=2, clasificación por pares y calibración sigmoide con OOF interna agrupada. Accuracy principal, F1 macro control.45 ajustes externos de pipeline en8803/9901/11003, más comparación137 solo para elegibles. Los ajustes internos del calibrador incluyen vectorizador y clasificador. Datos de desarrollo ya explorados; no prueba nueva.

El criterio se fijó antes de entrenar: delta accuracy>=0,001; al menos7/9 victorias; F1 medio no inferior; aumento de brecha<=0,005; no perder accuracy ni F1 en137. Es un criterio nuevo para esta ronda, distinto de011b, no una reclasificación retrospectiva de sus resultados.

| Modelo | Accuracy media | F1 macro medio | Brecha accuracy |
|---|---:|---:|---:|
| ref011 | 0.900243 | 0.904792 | 0.088056 |
| sigmoid | 0.900104 | 0.904595 | 0.088194 |
| polarity_c2 | 0.897014 | 0.901720 | 0.088316 |
| ovo_l1 | 0.895590 | 0.900167 | 0.089115 |
| polarity_c1 | 0.895208 | 0.899998 | 0.057222 |

Conservar011: ninguna candidata supera los criterios. No crear un envío equivalente ni bajar umbrales a posteriori.

No se infiere un techo absoluto de ML clásico ni se promete0,90 público. Tampoco la frecuencia de neutral demuestra que el descuento anterior sea una corrección probabilística de prior. La documentación previa afirma selección anidada de ese descuento, pero sus tablas/código de semillas no se localizaron en esta revisión. Los resultados012b sí quedan guardados por fold y ejemplo.

Reproducción: `.venv/bin/python -m src.experimento_012b` (protege registro y destinos existentes), `.venv/bin/python -m src.verificar_012b` y `.venv/bin/python -m src.informe_012b`. Ver reports/resumen_012b.md, reports/investigacion_012b.md, reports/verificacion_012b.json y notebooks/12b_revision_y_modelo.ipynb.

## Corrida 14: tokens de regla discursiva (R:)

Público confirmado: **0,90444** (011: 0,89888). Es la versión recomendada como envío final.

La etiqueta depende de qué opinión va al final y con qué conector: tras un conector contrastivo la última opinión decide la etiqueta el 99,9 % de las veces; tras uno aditivo, solo el 52 %. La 011 no representaba esto explícitamente. Se añaden tres módulos:

- `src/clausulas.py`: parte cada reseña en cláusulas con el conector que las introduce. No incluye «ahora», porque partiría mal frases como «por ahora».
- `src/lex.py`: léxico semántico de polaridad. Se eliminaron las frases encontradas mirando etiquetas («una estrella», «queda muy corto», «nada mal», «mal», «corto», etc.) para evitar fuga de información. Orden de evaluación: frases negativas, frases positivas, palabras sueltas, de modo que «no volvería» no cuenta como «volvería».
- `src/rulefeat.py`: emite una vez por reseña los tokens `R:` (última polaridad × tipo de conector × si hay algo después, más la secuencia de polaridades), para que el SVM aprenda una tasa por situación.

`src/features_016.py` es una copia de `features_007b.py` con el parámetro `rules` (guardado en `self` para que `clone()` lo conserve). Es copia y no edición para no romper la carga de los `.joblib` anteriores. El modelo es la 011 con un único cambio: `TextFeatures007b(thirds=True, rules=True)`, con L1, C=2 y el descuento de neutral d=0,2 (5 predicciones). Con C=1, C=3 o tokens repetidos ×3 se obtuvo menos.

**Corrección.** El commit 2d50ba1 tenía en `clausulas.py` el bucle de cláusulas dentro de `if first:`, de modo que se perdían las oraciones que empiezan con conector («Pero…», «Sin embargo…»), el 46 % de las reseñas. Con ese error las reglas no aportan nada; corregido, sí:

| Protocolo 013, 3 folds | Accuracy media (8803/9901/11003) | Folds | Brecha | Semilla 137 |
|---|---:|---:|---:|---:|
| 011 sin reglas | 0.8991 | — | 0.089 | 0.8977 |
| reglas, `clausulas.py` con error | 0.8989 | 5/9 | 0.089 | 0.9010 |
| reglas, `clausulas.py` corregido | **0.9015** | 6 victorias, 1 empate, 2 derrotas | **0.080** | **0.9015** |

La selección se hizo por CV antes de ver el público; el público lo confirma (unas 5 reseñas de 900, dentro del ruido de ±1 punto). Alrededor del 16 % del train parece tener etiqueta aleatoria, lo que pone el techo cerca de 0,91–0,92: no seguir ajustando hiperparámetros, descuentos ni umbrales contra el público, porque el privado (70 %) decide la nota.

Reproducir desde la raíz (protege el registro existente):

```bash
.venv/bin/python -m src.producir_014
```

Artefactos: `models/modelo_014.joblib`, `submissions/submission_014.csv` (cambia 170 de 3000 frente a 011), `reports/registro_produccion_014.json`, `reports/ejecucion_produccion_014.log` y `reports/verificacion_014.log` (ambas CV). Para cargar el modelo se requieren `src/features_016.py`, `rulefeat.py`, `clausulas.py` y `lex.py`. Los artefactos «014» de `experimento_014` son una ronda anterior distinta. `producir_011.py` se restauró a su versión original para que la 011 siga siendo reproducible.
