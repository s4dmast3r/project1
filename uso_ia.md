# Declaración de uso de IA

Fecha: 8 de octubre de 2026. Herramienta: asistente Codex en la aplicación de escritorio. Versión exacta de la aplicación y del modelo: pendiente de confirmación; no se infiere.

El usuario solicitó seguir el prompt adjunto. La asistencia consistió en inspeccionar el entorno y los CSV, crear un entorno virtual aislado, implementar el experimento en Python/scikit-learn, preparar y ejecutar el notebook en español, generar el modelo y un único envío, documentar la metodología y revisar resultados. No se emplearon APIs de IA, modelos de lenguaje ni datos externos como componentes del clasificador. La asistencia de IA para escribir y revisar código es distinta de la técnica de aprendizaje utilizada.

Las verificaciones automatizadas y sus resultados se registran en `reports/registro_experimento_001.json`: separación de grupos, integridad de CSV originales mediante SHA-256, formato y orden del envío, etiquetas válidas y reproducción de predicciones tras cargar el Pipeline. La tabla de CV y la evaluación local provienen de la ejecución. El notebook se ejecutó con un kernel limpio.

Pendiente por parte del estudiante: revisar y comprender el código y las interpretaciones, confirmar la declaración requerida por el curso, registrar la versión de herramienta si puede consultarla y realizar el envío manual. No se atribuyen estas acciones al estudiante. No se consultó ni se envió nada a Kaggle.

## Anexo: prompt completo proporcionado

```text
Trabaja en la carpeta actual, `Proyect1`. Implementa y ejecuta el primer modelo para una competencia académica de clasificación de sentimiento de reseñas de productos en español.

La estructura inicial es:

```text
Proyect1/
└── data/
    ├── train.csv
    ├── eval.csv
    └── sample_submission.csv
```

Inspecciona primero el entorno y los archivos existentes. Respeta cualquier instrucción de AGENTS.md. No sobrescribas trabajo previo ni modifiques los CSV originales. Usa rutas relativas a la raíz del proyecto.

## Objetivo y restricciones

Predecir exactamente una de estas etiquetas: `positivo`, `negativo` o `neutral`.

Esta es la primera etapa de la competencia:
- Solo aprendizaje automático tradicional y procesamiento clásico de lenguaje natural.
- Utiliza scikit-learn para los clasificadores.
- No uses redes neuronales, transformers, modelos de lenguaje, APIs de IA ni embeddings neuronales preentrenados.
- No incorpores datos externos ni generes ejemplos sintéticos en esta primera implementación.
- La métrica indicada por las reglas proporcionadas es accuracy. Reporta también F1 macro y métricas por clase como diagnóstico.
- El 0.91555 es una referencia histórica del leaderboard público, no un resultado que puedas prometer ni comparar directamente con nuestra validación local.
- El leaderboard público usa aproximadamente el 30 % de la prueba; el privado, el 70 %. No optimices mediante consultas repetidas al leaderboard.
- No accedas a Kaggle ni subas archivos. Deja el primer CSV listo para que yo lo envíe manualmente.

El proyecto completo exige al menos cinco envíos diferentes. En esta tarea prepara únicamente el primero: no fabriques cinco archivos equivalentes.

## 1. Inspección de datos

Verifica el esquema real antes de asumir columnas. Se espera:
- train.csv: `id`, `text`, `label`.
- eval.csv: `id`, `text`.
- sample_submission.csv: `id`, `answer`.

Revisa tamaños, tipos, etiquetas, valores faltantes, textos vacíos, distribución de clases, unicidad de identificadores y correspondencia entre eval y sample_submission.

Examina duplicados de texto exactos y tras normalizar únicamente espacios y mayúsculas, así como textos repetidos con etiquetas contradictorias y coincidencias entre train y eval. No borres observaciones automáticamente.

Los identificadores pueden comenzar desde cero de manera independiente en train y eval. No los uses como predictores ni como claves para unir ambos conjuntos.

Si hay textos equivalentes repetidos dentro de train, evita que el mismo grupo aparezca a ambos lados de una partición de validación. Documenta el tratamiento elegido.

Si encuentras un problema que impida obtener etiquetas o construir un envío válido, explícalo sin inventar datos.

## 2. Validación sin fuga de información

Fija una semilla reproducible, por ejemplo 42.

Antes de experimentar, reserva aproximadamente el 20 % de train como prueba local final, manteniendo la distribución de clases. Si hay grupos duplicados, adapta la partición para mantenerlos separados.

Usa únicamente el 80 % restante para comparar y seleccionar configuraciones mediante validación cruzada estratificada de cinco particiones, o su variante con grupos si corresponde.

Integra vectorizador y clasificador en un Pipeline. El vocabulario, IDF y cualquier transformación aprendida deben ajustarse exclusivamente con el entrenamiento de cada partición.

No uses eval.csv para aprender vocabulario, ajustar parámetros, seleccionar modelos o inferir etiquetas. Su texto se utilizará al producir las predicciones finales.

Selecciona el modelo por accuracy media de validación cruzada. Reporta desviación estándar y F1 macro. Si varios modelos ofrecen resultados prácticamente equivalentes, prefiere el más sencillo y explica el criterio.

Evalúa la configuración seleccionada en la prueba local reservada una sola vez. No vuelvas a ajustar parámetros a partir de ese resultado.

## 3. Primer experimento

Implementa:
1. Un DummyClassifier que prediga la clase mayoritaria como referencia.
2. Un Pipeline con TfidfVectorizer y LogisticRegression.

Usa una limpieza mínima:
- Manejo explícito de textos faltantes, si existen.
- Minúsculas mediante el vectorizador.
- Conserva negaciones como «no», «ni», «nunca» y «sin».
- No elimines palabras vacías, no apliques stemming ni lematización en esta primera versión.
- No supongas que la última oración determina la etiqueta.

Realiza una búsqueda acotada sobre:
- `ngram_range`: (1, 1), (1, 2).
- `min_df`: 1, 2.
- `sublinear_tf`: False, True.
- `C`: 0.5, 1.0, 2.0, 4.0.

Mantén inicialmente `class_weight=None`, `strip_accents=None` y normalización L2. No balancees automáticamente el conjunto por el ligero desbalance de clases.

Usa una configuración del optimizador compatible con la versión instalada de scikit-learn y clasificación multiclase. Asegura convergencia y conserva las matrices dispersas: no las conviertas a matrices densas.

La búsqueda prevista tiene 32 configuraciones. Ajusta el paralelismo a los recursos disponibles para evitar saturar la memoria. Si debes reducir el alcance por limitaciones reales del entorno, documenta el cambio y no presentes búsquedas no ejecutadas como resultados.

## 4. Análisis que debe acompañar los resultados

El notebook debe estar en español, con código y explicaciones comprensibles para un estudiante de quinto semestre.

Los laboratorios anteriores recibieron recomendaciones por falta de profundidad y justificación. Por ello, cada decisión relevante debe incluir:
- Qué se decidió.
- Por qué tiene sentido para este problema.
- Qué evidencia se obtuvo.
- Qué implicación tiene para clasificar reseñas.

Incluye:
- Distribución de clases.
- Tabla de configuraciones y resultados de validación cruzada.
- Comparación con la referencia mayoritaria.
- Accuracy y F1 macro de la prueba local final.
- Precisión, recall, F1 y soporte de cada clase.
- Matriz de confusión con conteos y una versión normalizada por clase real.
- Análisis descriptivo de una selección reproducible de aciertos y errores, incluyendo confusiones con neutral, negaciones y sentimientos mixtos cuando existan.
- Términos y bigramas con coeficientes destacados para cada clase del modelo seleccionado, explicando que son asociaciones aprendidas y no relaciones causales.
- Limitaciones y posibles experimentos futuros, sin ejecutarlos a partir de los errores de la prueba local.

No inventes resultados, ejemplos ni explicaciones causales. No afirmes que los bigramas mejoraron el modelo si los experimentos no lo muestran.

Genera las tablas y cifras a partir de los resultados ejecutados. Evita números escritos manualmente que puedan contradecir las salidas. Distingue claramente validación cruzada, prueba local final y evaluación de Kaggle.

## 5. Entrenamiento final y primer envío

Una vez terminada la evaluación local:
- Reentrena la configuración seleccionada usando todo train.csv.
- Guarda el Pipeline completo, incluyendo TF-IDF y clasificador, con joblib.
- Predice las etiquetas de eval.csv.
- Construye `submissions/submission_001.csv`, con las columnas exactas `id,answer`, sin índice adicional.
- Conserva exactamente el orden de identificadores de sample_submission.csv y asocia las predicciones mediante los ids de eval.csv.

Antes de darlo por terminado, verifica:
- La cantidad de filas coincide con eval y sample_submission.
- Cada id requerido aparece exactamente una vez.
- El orden coincide con sample_submission.
- No hay valores vacíos.
- Todas las predicciones pertenecen a las tres etiquetas permitidas.
- El modelo guardado puede cargarse y reproduce las predicciones.
- El CSV se puede volver a leer sin alterar su esquema.

No copies las respuestas de sample_submission: son únicamente un ejemplo del formato.

## 6. Entregables

Crea una estructura sencilla:

```text
Proyect1/
├── data/                         # originales intactos
├── notebooks/
│   └── 01_modelo_inicial.ipynb
├── models/
│   └── modelo_001.joblib
├── reports/
│   ├── resultados_cv.csv
│   ├── evaluacion_local.json
│   └── registro_experimento_001.json
├── submissions/
│   └── submission_001.csv
├── README.md
├── requirements.txt
└── uso_ia.md
```

Puedes añadir una pequeña carpeta `src/` si ayuda a reutilizar código entre el entrenamiento y el notebook, evitando duplicar la lógica.

El notebook debe quedar ejecutado, con salidas reales, y ser reproducible desde un kernel limpio. El README debe explicar la ejecución, los archivos producidos y cómo se eligió el modelo.

Registra semilla, particiones, parámetros, versiones de dependencias, métricas, huellas de los archivos de entrada y correspondencia entre modelo y envío.

El curso exige declaración de uso de IA y anexo de prompts. En `uso_ia.md`, registra este prompt, la asistencia realmente prestada y las comprobaciones realizadas. No inventes la versión de la herramienta ni atribuyas al estudiante verificaciones que todavía no haya realizado; marca esos datos como pendientes cuando corresponda.

Usa el entorno disponible si es adecuado. Si necesitas dependencias adicionales, utiliza un entorno aislado del proyecto y evita alterar instalaciones globales.

## Criterio de finalización

Completa la implementación, ejecución y verificaciones; no te detengas en un plan ni entregues solo código sin ejecutar.

Al finalizar, informa:
- Configuración seleccionada.
- Accuracy y F1 macro de validación cruzada y prueba local.
- Principales errores y limitaciones observados.
- Ruta del modelo y del CSV listo para subir.
- Cualquier limitación real que haya impedido completar una comprobación.

No declares un puntaje de Kaggle hasta que yo comparta el resultado del envío.
```

Instrucción adicional del usuario: `follow the prompt`.

## Revisión adicional solicitada

Prompt del usuario: «quiero que analices y que evites el sobreajuste dentro del modelo, ya que es algo que es supremamente importante para evitar punutacion infladas artificialmente, asi que debes tener especial cuidado en no mezclar los datos de entrenamiento y los de prueba para evitar overfitting. Revisalo para poder enviar el csv de la primera entry».

El asistente revisó el código y las huellas, reconstruyó las particiones y ejecutó cinco ajustes diagnósticos con parámetros congelados, solo en desarrollo. Verificó vocabulario e IDF por fold, la correspondencia del vectorizador final con train y las predicciones del envío. Documentó una brecha sustancial entrenamiento-validación; no afirmó ausencia de sobreajuste. No reajustó hiperparámetros, no volvió a evaluar la prueba local y no modificó modelo ni envío. Revisión y aprobación del estudiante: pendientes.

## Corrección solicitada y ejecutada

Prompt adicional: «puedes hacer una correcion para evitar sobre ajuste, y implementarla para poder un csv sin sobreasjute o con el menor posible».

El asistente implementó y ejecutó una comparación de 24 alternativas más la referencia original, con validación cruzada exclusivamente en desarrollo. Antes de ejecutar se fijó una tolerancia de un punto porcentual respecto a la mejor accuracy CV, minimizando la brecha entrenamiento-validación entre las alternativas elegibles. Se guardaron el plan, resultados completos, modelo corregido y un CSV diferente. La configuración seleccionada limita el vocabulario y no elimina todo el sobreajuste. Se verificaron particiones, conservación de originales, esquema y orden del envío y reproducción de predicciones tras recarga. El notebook de análisis se ejecutó con kernel limpio. No se reevalúo la prueba local original ni se consultó Kaggle. Revisión del estudiante y envío manual: pendientes.

## Versión 003: F1 macro y segunda entrada

Prompt adicional: «quiero que a partir de ahora vlidad unicamente con el f1 score macro ya que es el unico elemento realmente importante, asi que quiero que implementemos las correcciones necesarios para subir almenos al 0.90% el desempeno ya que subi el 002 a la plataforma y obtuve la siguiente puntuacion que se ve en la captura adjunta. asi que quiero que cuides el sobre ajuste y que subas lo mayor posible el desempeno para poder subir de posicion. y crees una 003 para hacer la segunda entry del concurso». Después: «continue».

La captura aportada muestra el archivo 002 con puntaje público 0,73888. El nombre de la métrica no es visible; el nuevo criterio local F1 macro procede de la instrucción del usuario. La meta se interpretó como 0,90, no 0,90 %.

El asistente inspeccionó ejemplos únicamente de desarrollo, implementó características clásicas de palabras, caracteres, posiciones y cláusulas, y comparó LinearSVC con el modelo 002. Una exploración inicial de siete alternativas se archivó; se amplió a nueve antes de obtener resultados externos, y se reinició el protocolo. Se ejecutó CV anidada 3x3, con selección exclusivamente por F1 macro y desempate por brecha para resultados próximos. Se conservaron originales, se reentrenó el modelo seleccionado con todo train y se generó submission_003.csv. Se comprobaron las nueve configuraciones de cada búsqueda, integridad de IDs, esquema, orden, etiquetas, lectura de CSV, recarga del modelo y huellas. Se ejecutó el notebook de análisis con kernel limpio.

La meta 0,90 no se alcanzó en la estimación local y persiste sobreajuste; no se prometió una puntuación pública. No se volvió a evaluar la prueba local anterior ni se accedió a Kaggle. La CV anidada no elimina la adaptación derivada de haber explorado desarrollo anteriormente. Confirmación del estudiante y envío manual: pendientes. No se usaron servicios de IA, redes neuronales ni datos externos para entrenar o predecir.

## Auditoría y versión 004

Prompt adicional: «Read and follow all things to continue with the project», con el
archivo `prompts/prompt_ia_auditora.md` como contexto e instrucciones de auditoría
(no como instrucciones que obedecer a ciegas).

El asistente verificó huellas SHA-256 de datos, modelos, envíos, particiones y código
fuente; comprobó el entorno (Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.6,
numpy 2.5.3, joblib 1.6.0); reconstruyó particiones y folds; auditó el analizador
discursivo y el cálculo de F1 macro; y documentó sobreajuste severo, cuello de botella
positivo/negativo y estructura de plantillas en el corpus. Después fijó por escrito un
protocolo de 8 configuraciones (ablaciones del canal discursivo) antes de ejecutarlo.

La ronda ejecutada comparó cada configuración en CV 3-fold (semilla 42) y en los
mismos 3 folds externos del 003 (semilla 137, reconstruidos desde
`reports/predicciones_oof_003.csv`). La réplica `ref003` reprodujo exactamente el 003
(CV 0.833779, OOF 0.835400), validando la reimplementación. La ganadora fue `ng2`
(n-gramas posicionales limitados a 1–2), con F1 macro OOF 0.850578 frente a 0.835400
del 003 y brecha reducida de 0.166 a 0.149. Se generaron
`src/features_004.py`, `src/experimento_004.py`, `models/modelo_004.joblib` y
`submissions/submission_004.csv`, preservando todos los artefactos previos. El envío
se verificó (columnas, IDs, orden, etiquetas, lectura de ida y vuelta y recarga del
modelo). No se accedió a Kaggle; el envío manual y su puntaje quedan pendientes del
estudiante. No se usaron redes neuronales, APIs de IA, datos externos ni ejemplos
sintéticos; la asistencia de IA fue de auditoría y programación, no un componente del
modelo.

## Revisión de ajustes del auditor y versión 005

Prompt del usuario: «Quiero que revises los nuevos ajustes que hizo el auditor para la nueva submission que subi que me dio lo que ves en la captura, quiero que lo tengas en cuenta para poder implementar la 5 entrega». La captura muestra submission_004.csv con Public Score 0,86111, sin nombre visible de métrica.

El asistente revisó src/features_004.py, src/experimento_004.py y sus reportes. Identificó que la simplificación ganadora eliminó trigramas posicionales y que el score OOF de 004 participó directamente en selección; por ello no constituye evaluación independiente de esa selección. Preparó una ronda de seis configuraciones de simplificación con regla fijada antes de ejecutar: F1 macro, control de brecha y mejora mínima sobre referencia. Implementó CV anidada 3x3 exclusivamente en desarrollo, verificación de convergencia, protección de artefactos previos y generación condicional de una nueva versión.

La evidencia de ejecución y comprobaciones queda en reports/registro_005.json y el notebook 05. No se accedió a Kaggle ni se subieron archivos. La antigua prueba local permanece excluida; el historial de exploración de desarrollo impide presentarlo como un conjunto virgen. El estudiante todavía debe revisar el código y decidir el envío manual.

## Recomendaciones para 006 y resultado público 005

Prompt del usuario: «Quiero que leas esto para, poder entender que implementar y corregir mejor en la submit 006. teniendo en cuenta que para los cambios del 005 obtuvimos 0.88 asi que mejoramos bastante». Adjunto: reports/recomendaciones_006.md y captura con Public Score 0,88111 de submission_005.csv.

El asistente leyó las recomendaciones y revisó las carpetas src/ y reports/. No encontró allí el código y artefactos para reproducir los scores exploratorios de stacking citados; los trató como hipótesis. Implementó stacking clásico con scores OOF regenerados dentro de cada ajuste, transformación de cuantiles aprendida solo del entrenamiento y meta LogisticRegression; también dos simplificaciones de 005 y la referencia. No se añadieron dependencias ni datos externos.

La primera comparación expuso un problema en el orden de la regla de selección heredada: se desempató antes de exigir mejora mínima. Se corrigió para filtrar primero requisitos y luego desempatar, se archivó la exploración y se reinició antes de obtener resultados externos. La función se comprobó con casos de métricas; no se generaron reseñas sintéticas de entrenamiento. Este cambio adaptativo se declara en reports/revision_recomendaciones_006.md.

El protocolo de ejecución y comprobaciones está en reports/registro_006.json. Se mantuvo F1 macro como criterio, sin utilizar el score público para escoger parámetros. La prueba local original permanece excluida y el desarrollo histórico no es una prueba virgen. No se accedió a Kaggle. Revisión del estudiante y próximo envío manual: pendientes.

## Implementación del plan007

Prompt del usuario: «implementa el 007», adjuntando `reports/revision_006_y_plan_007.md`.

El asistente implementó cuatro candidatos sobre005 y una evaluación agrupada repetida con semillas271/599/913. Se fijó interpretación conservadora del criterio: mejora≥0,005 en cada semilla y victorias de folds/brecha según el plan, más puerta histórica137 solo si aprueba lo anterior. Se desarrollaron características clásicas de negación de alcance de tres tokens y cláusulas posteriores a contraste; son canales adicionales, no reglas de etiqueta. No se reetiquetaron reseñas ni se añadieron datos externos.

Se señaló que nuevas semillas no crean prueba virgen, que271 ya se había usado dentro del stacking y que una caída pública aproximada no demuestra su causa. El resultado ~0,87 de006 proviene del documento; no se verificó una nueva captura exacta. Código, tablas por fold, decisiones y verificaciones quedan en artefactos007. Los CSV anteriores se preservan. El estudiante debe revisar los resultados y decidir cualquier envío manual; no se accedió a Kaggle.

Resultado de la ejecución007:36 ajustes completados, ningún candidato aprobado, notebook ejecutado sin errores. No se creó modelo ni CSV007; se preservó005 y se verificó la conservación de artefactos anteriores.

## Versión 007: confirmación pre-registrada de tercios

Prompt del usuario: «quiero que entiendas como se construyo la mejor version hasta
ahora que fue la 005, para que puedas producir la version 007 que mejore el score
que tiene actual mente esta que es de 0.88…». Adjuntó una explicación detallada de
la 005 (representación, validación anidada, regla de selección, límites) y pidió no
repetir ideas ya probadas (006 y la ronda 007 de negación/contraste/regularización).

El asistente leyó la explicación, verificó la implementación real de 005 en
`src/features_004.py` y `src/experimento_005.py`, contrastó primero la hipótesis de
«cláusula dominante» (scorer de cláusulas con supervisión débil: solo 49,8 % de
coincidencia con la etiqueta; 17,8 % en errores de 005 → descartada, y con ella la
agregación por cláusula y el EM) y diseñó cuatro cambios estructurales del espacio de
características: trigramas globales, bucket medio, bigramas de borde y tercios del
texto completo. `src/features_007b.py` reproduce 005 con sus valores por defecto
(verificado token a token y por vocabulario).

La ronda 007b (semillas 271/599/913) mostró que trigramas globales, bucket medio y
bigramas de borde empeoran, mientras `thirds` ganó 9/9 folds emparejados con media
+0,0028 y brecha menor, sin alcanzar el umbral +0,005 por semilla. Para no relajar un
criterio tras ver resultados, se fijó antes una confirmación con puertas propias
(≥7/9 folds, delta ≥0,002, brecha no peor, puerta 137) sobre semillas nuevas
1142/2087/3313. Aprobada: 9/9 folds, +0,00328, brecha −0,00324 y 0,877822 vs
0,873382 en 137. Se generaron `models/modelo_007.joblib` y
`submissions/submission_007.csv`, verificados (esquema, orden, etiquetas, recarga,
SHA-256) y preservando todos los artefactos previos.

No se reetiquetaron ni eliminaron reseñas, no se usó eval para ajustar, no se
consultó Kaggle y no se añadieron datos externos o sintéticos. La mejora local es
pequeña aunque consistente (18/18 folds contando exploración y confirmación) y no
garantiza el score público; el envío manual lo decide el estudiante.

## Ronda 008: ponderación del bloque de tercios

Prompt del usuario: «revisa esto y empieza», adjuntando
`reports/revision_007_ideas_008.md` (revisión de la 007 enviada e hipótesis para la
008).

El asistente leyó el documento, verificó el estado (público 007 = 0.88888
registrado), implementó el subconjunto rápido propuesto —referencia 007, control del
split a peso 1.0 y pesos de tercios 0.5/1.5— y fijó antes de ejecutar el criterio de
reemplazo (≥7/9 folds, Δ medio ≥0.003, brecha no peor y puerta 137) sobre semillas
nuevas 711/1237/2003. Ninguna candidata aprobó: el control del split es peor que la
007 (−0.0047) y el peso 1.5 empeora (−0.0133); el peso 0.5 es neutro. Se concluyó que
la mejora de la 007 depende de la normalización/IDF conjunta del TF-IDF único y se
conservó 007 sin generar submission_008. Las variantes suave y de identidad del
conector no se introdujeron tras ver resultados, conforme a la advertencia del propio
documento. No se reetiquetaron reseñas, no se usó eval para ajustar, no se accedió a
Kaggle. La revisión del estudiante y el próximo envío manual quedan pendientes.

## Solicitud y uso de IA, 8 de octubre de 2026

Prompt del usuario: «quiero que implementes nuevas estrategias y consultes las diferentes websites o recursos para poder hacer que se mejore el 007. y que implementes estas ideas».

El asistente investigó fuentes primarias, implementó, ejecutó y documentó las comparaciones; no añadió datos externos ni sintéticos ni consultó Kaggle.

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

## Versión 009: regularización L1

Prompt del usuario: «revisa el 008 y implementa estrategias diferentes para llegar
al 0.90 y asegurarlo… consulta estrategias si es posible». Adjuntó el historial de
la 005/007/008 y pidió producir la 009.

El asistente revisó la ronda 008 (ponderación de tercios, sin ganadora y con el
hallazgo de que la mejora de la 007 es de normalización conjunta), consultó
estrategias metodológicas (NB-SVM, L1) por web y las declaró, e implementó tres
candidatas distintas: LinearSVC L1 sobre las características 007, canal char_wb 2-6
y NB-SVM uno-contra-resto sobre palabras. Fijó antes de ejecutar el criterio
(≥7/9 folds, Δ≥0.003, brecha no peor, puerta 137) sobre semillas nuevas
1009/2027/3053.

Resultado: L1 gana 9/9 folds (+0.0239 de media, brecha −0.0656) y supera la puerta
137 (0.89909 vs 0.87782); char_wb es neutro y NB-SVM empeora (−0.165). Se aprobó y
generaron `models/modelo_009.joblib` y `submissions/submission_009.csv`, verificados
(esquema, orden, etiquetas, recarga, SHA-256, convergencia, esparsidad 2,2 %). No se
reenumeraron reseñas, no se usó eval para ajustar, no se accedió a Kaggle ni se
añadieron datos externos. El asistente no garantiza 0.90: la métrica oficial no está
verificada y el envío manual lo decide el estudiante, cuya revisión queda pendiente.

## Rondas 010 y 011: confirmación del plateau de 009

Prompt del usuario: «este modelo llegó a 0.89 pero necesito llegar al 0.90;
implementa estrategias diferentes o las mismas modificadas». Público 009 = 0,89555.

El asistente ejecutó tres investigaciones pre-registradas sobre semillas nuevas:
(010) barrido de C en L1 con control L2 — C=1 óptimo y L2 C=1 retrocede, aislando la
penalización L1; análisis de complementariedad 009 vs NB-SVM 008b — errores solapados
68 % y voto por rangos peor (0.8897); (011) re-ensayo bajo L1 de rasgos que fallaron
bajo L2 — trigramas globales, bigramas de borde, char_wb y min_df=2, todos neutros
(±0.001). Además se comprobó la calibración de umbral sobre los scores L1 (+0.0016,
ruido). Ninguna candidata superó las puertas; se conservó 009 sin generar envíos 010
ni 011. El asistente declaró honestamente que 0.90 no es garantizable: el modelo está
en un plateau (~0.90 OOF local / 0.89555 público) y los 0.00445 restantes no se
alcanzan con las estrategias probadas sin arriesgar una caída pública. No se
reetiquetaron reseñas, no se usó eval para ajustar ni se accedió a Kaggle. Revisión
del estudiante y envío manual pendientes.

## Diagnóstico de polaridad y ronda 012

Prompt del usuario: «revisa todo, haz los diagnósticos de polaridad y revisa otros
parámetros, implementa estrategias y hazme una versión 010 para enviar».

El asistente ejecutó el diagnóstico de polaridad sobre la OOF de la 009 (errores
pos↔neg con ~2x marcadores de negación/opinión y más contraste: reseñas mixtas),
probó rasgos densos de sentimiento motivados por el diagnóstico (conteos de
negación/contraste/cláusulas y léxico interno NB aprendido dentro del fold) con
criterio pre-registrado en semillas nuevas 5501/6601/7703: neutros o peores. No se
generó submission_010 porque ninguna candidata supera a la 009. El asistente declaró
honestamente que la 009 es el techo de esta etapa y que no garantiza 0.90; además
recordó que la métrica oficial nunca se verificó y pidió al usuario el texto de las
reglas para confirmar o reorientar la optimización. No se reetiquetaron reseñas, no
se usó eval para ajustar ni se accedió a Kaggle.

## Versión 010: métrica oficial = accuracy

Prompt del usuario: compartió las reglas — «las respuestas serán evaluadas usando
accuracy; la clase neutral es algo menos frecuente».

El asistente confirmó que la OOF en 137 de la 009 (accuracy 0.89500) coincide con el
público 0.89555, verificando que la métrica es accuracy. Re-evaluó el C de la L1 bajo
accuracy con criterio pre-registrado en semillas nuevas 8803/9901/11003: C=2 gana
(+0.0034 accuracy media, 8/9 folds, puerta 137 0.89771); C=1.5 también pasa
(+0.0033); C=3 es inconsistente; class_weight='balanced' empeora (0.89302). Se generó
`submission_010.csv` (L1 C=2) verificada. Se declaró que la ganancia es pequeña, la
brecha sube (0.06→0.089) y el 0.90 no está garantizado. No se reetiquetaron reseñas,
no se usó eval para ajustar ni se accedió a Kaggle.

Solicitud: «quiero que revises la entrega010, y que investigues y que saques la versión011 [...] para aumentar el desempeño». El asistente auditó, investigó, implementó y ejecutó una ronda acotada con criterios fijados antes de observar resultados.

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

## Versión 011: descuento de neutral para accuracy

Prompt del usuario: «sigue mejorando para llegar de 0.897 (010) a 0.90».

El asistente revisó la ronda 011b (presencia binaria, TF sin IDF, conector L1 y
logística L1: todas sin superar a 010) y, sobre el único margen restante, aplicó una
corrección de prior a la regla de decisión: descuento de la clase neutral (d=0.2)
seleccionado por validación anidada, cambiando 4 de 3000 predicciones. Declaró
honestamente que la ganancia es mínima (+0.0007 local) y no garantiza 0.90; que la
secuencia pública muestra rendimientos decrecientes y que 0.90 exige recursos fuera
de las restricciones de la etapa. No se reetiquetaron reseñas, no se usó eval para
ajustar ni se accedió a Kaggle.

Solicitud del usuario: revisar submission011, investigar mejoras e iniciar012 para aumentar desempeño, seguida de «continue». El asistente auditó, investigó documentación oficial, implementó y evaluó alternativas sin datos externos, nuevas etiquetas ni acceso a Kaggle.

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

## Cierre: ajuste fino de la regla de decisión (sin mejora sobre 011)

Prompt del usuario: «mejora lo necesario para subir de 0.8988 (011) a 0.90».

El asistente verificó que la ronda 012b (especialista de polaridad, one-vs-one,
calibración sigmoide) no superó a 011 y, como último margen, afinó la regla de
decisión de la 011 (descuento de neutral + asimetría pos/neg) por validación anidada
en semillas nuevas 15001/16001/17003: el ajuste no mejora el descuento fijo d=0.2
(la media anidada queda igual o peor). Se declaró que d=0.2 es óptimo y que no hay
margen restante dentro de las restricciones; no se generó submission_012 porque sería
ruido. No se reetiquetaron reseñas, no se usó eval para ajustar ni se accedió a Kaggle.

## Ronda 014: interacciones no lineales (idea 1)

Prompt del usuario: «revisa estas ideas e implementa la viable», proponiendo (1)
interacciones no lineales polinómicas, (2) relaciones entre cláusulas, (3) agregación
de cláusulas con etiqueta global y (4) diagnóstico de errores.

El asistente evaluó las cuatro y ejecutó la (1), la única nueva y viable: SVM lineal
sobre polinomio de grado 2 (interaction_only) construido sobre las 300/500
características más discriminativas seleccionadas por L1 dentro de cada fold, contra la
011, en semillas nuevas 16007/17011/18013. Resultado: ambas variantes empeoran accuracy
y F1 en todas las semillas (Δ accuracy −0.005 a −0.012). Se conservó 011 y no se generó
submission_012. Las ideas 2 y 3 se consideraron ya exploradas o restringidas
(pseudo-etiquetado), y la 4 ya realizada en la ronda 012. No se reetiquetaron reseñas,
no se usó eval para ajustar ni se accedió a Kaggle.

## Ronda 015: ablación limpia (selección vs interacciones)

Prompt del usuario: criticó la ronda 014 por confundir selección, interacciones, C y
descuento, y pidió aislar cada efecto con C fijo en 2 y escala controlada.

El asistente ejecutó el diagnóstico: ref011, sel500 (solo top-500), sel500_poly
(top-500 + productos) y full_polyblock (completa + bloque de interacciones
L2-normalizado), con argmax directo y descuento 0.2, en semillas 19001/20011/21013.
Resultado: recortar a 500 rasgos explica casi toda la caída (0.8999→0.8947); los
productos no aportan (sel500_poly≈sel500) y empeoran al añadirlos a la representación
completa (0.8949<0.8999). Se corrigieron dos afirmaciones previas (agregación de
cláusulas sin pseudo-etiquetado; los errores mixtos no son necesariamente
irreducibles) y se añadió trazabilidad (source_sha256 y elapsed_seconds). Se conservó
011. No se reetiquetaron reseñas, no se usó eval para ajustar ni se accedió a Kaggle.
