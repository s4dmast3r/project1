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
