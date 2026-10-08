# Revisión de recomendaciones y protocolo 006

La captura del usuario confirma público 005 de 0,88111. `recomendaciones_006.md` fue escrito antes de ese resultado; sus referencias a “pendiente” son históricas. El nombre de la métrica oficial no queda visible en la captura. El criterio local continúa siendo F1 macro.

## Qué se incorpora

- H1: stacking de LinearSVC con representación 005, LogisticRegression y MultinomialNB con palabras 1–2. Meta-modelo LogisticRegression C=1. Se fijan hiperparámetros y no se seleccionan pesos mirando el leaderboard.
- H3: regularización C=1 y min_df=5; otra alternativa elimina únicamente las interacciones `epilogueTrue/False` de las características, preservando tratamiento de cierre, segmentación, posiciones y n-gramas globales. La primera combina dos cambios para limitar presupuesto; no permite atribuir una mejora a uno de ellos aisladamente.
- Referencia 005: mismo C=2, min_df=3, unigramas posicionales y sin caracteres. Cuatro configuraciones en total.

No se ejecutan H2 ni H4 en esta ronda: requerirían más decisiones/dependencias y búsquedas. No se incorporan datos externos, stemming, stopwords, árboles, KNN, embeddings, SMOTE ni ejemplos sintéticos. La decisión limita alcance; no prueba que esas alternativas sean inútiles.

## Qué debe corregirse o verificarse en las recomendaciones

No se hallaron en src/ ni reports/ los scripts, scores por muestra y metadatos que permitan reproducir el stacking exploratorio de 0,8805 citado. Se trata como hipótesis, no como resultado confirmado en esta implementación.

“Meta entrenada solo con folds ajenos” es insuficiente si sus entradas son OOF generadas globalmente: los modelos que produjeron las filas de entrenamiento del meta-modelo podrían haber aprendido con etiquetas del fold externo que se quiere evaluar. Esa fuga indirecta puede existir aunque la fila de validación no aparezca directamente al entrenar el meta-modelo. Sin el código no se afirma que ocurrió; se evita explícitamente en 006.

La complementariedad de errores es una razón para investigar stacking, no prueba de independencia estadística ni garantía de mejora. Un meta-modelo pequeño no vuelve automáticamente bajo el riesgo total del proceso. Tampoco una expectativa de +0,005–0,007 ni una mejora pública anterior son resultados de esta ejecución.

## Protocolo ejecutable

La CV externa tiene tres folds de desarrollo, semilla 137; cada uno vuelve a seleccionar entre cuatro candidatos con tres folds internos, semilla 42. Cada ajuste del stacking crea a su vez tres folds agrupados, semilla 271, EXCLUSIVAMENTE dentro de las filas que recibe ese ajuste. Los modelos base generan scores OOF allí y el meta-modelo se ajusta con esos scores. Las bases se reajustan con todas las filas del entrenamiento de ese ajuste para predecir su validación.

El SVC aporta decision_function; LR y NB aportan predict_proba. En lugar de calcular rangos sobre el lote de validación, se ajusta QuantileTransformer con 100 cuantiles, distribución normal, sobre los scores OOF del entrenamiento del meta-modelo. Los scores de validación/eval solo se transforman con esa función ya fijada. Esto evita que las predicciones dependan de qué otras reseñas se predicen a la vez. No es una reproducción exacta del método de ranking no documentado del auditor.

En producción, si gana el stacking, se generan scores OOF nuevos usando todo train, se entrena el meta-modelo con ellos y se ajustan las bases con todo train. La recomendación de reutilizar meta-scores de desarrollo no se adopta: se regeneran de forma coherente con el conjunto de entrenamiento final, una vez cerrada la selección. eval no participa en ninguno de estos ajustes.

F1 macro exclusivamente, etiquetas explícitas, zero_division=0. Regla previa: candidatos con brecha no mayor que la referencia +0,002; entre los situados a 0,003 del mejor F1 elegible se prefiere menor brecha. Solo se crea un envío diferente si mejora al menos 0,002 en selección. La CV externa no cambia esa selección. Las tolerancias no equivalen a una prueba de significancia.

Datos y artefactos previos protegidos por SHA-256; módulos nuevos para evitar romper joblib anteriores. Convergencia exigida en LR y SVC. Un proceso y un hilo numérico. Sin reusar la antigua prueba local. El desarrollo ya observado y los mismos folds externos no constituyen una evaluación externa virgen; la separación anidada no elimina la adaptación histórica.

## Corrección de la regla durante la exploración inicial

La implementación heredada aplicaba el desempate antes de exigir mejora mínima. La primera búsqueda mostró que podía seleccionar C=1/min_df=5 por brecha y descartarlo después por mejora insuficiente, aunque stacking cumplía ambos requisitos. Se corrigió el orden: filtrar primero mejora >=0,002 y brecha <=referencia+0,002; después aplicar tolerancia y desempate. Es un cambio de selección motivado por la exploración y se declara; no se oculta como regla original.

Se archivaron tabla, plan y log iniciales. Se interrumpió el primer fold externo antes de obtener su resultado y se reinició el protocolo completo. Se verificó la función de selección con casos de métricas de prueba (no se generan ejemplos sintéticos de reseñas ni se usan para entrenar).
