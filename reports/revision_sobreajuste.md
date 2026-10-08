# Revisión de separación y sobreajuste

Se verificó la integridad de los archivos contra las huellas del primer experimento. Las 9600 filas de desarrollo y las 2400 de prueba local son disjuntas; tampoco comparten grupos de texto normalizado. Los cinco folds coinciden con las asignaciones guardadas. La revisión del código confirma que la búsqueda recibe exclusivamente desarrollo y que cada Pipeline aprende vocabulario e IDF dentro de su fold. Se verificó numéricamente la igualdad con vectorizadores ajustados únicamente sobre cada entrenamiento. Los IDs no son características. Las respuestas de sample_submission no se usan.

## Diagnóstico con configuración congelada

- Accuracy media sobre entrenamiento de cada fold: 0.982526.
- Accuracy media sobre validación de cada fold: 0.731979.
- Brecha entrenamiento-validación: 25.05 puntos porcentuales.
- Accuracy de la prueba local original, sin reevaluarla: 0.726250.
- Diferencia CV-prueba local original: 0.57 puntos porcentuales.

La brecha entrenamiento-validación mide adaptación al entrenamiento y debe reconocerse como señal de sobreajuste. No se debe publicar la accuracy de entrenamiento como capacidad de generalización. La cercanía entre CV y prueba local no demuestra ausencia de sobreajuste, pero no muestra una caída grande adicional en la prueba reservada. La CV participó en la selección de 32 configuraciones y puede tener optimismo de selección; la prueba reservada aporta una estimación independiente del procedimiento original. No se ajustan parámetros después de observarla.

La regularización y min_df ya forman parte del modelo seleccionado. No se reduce C automáticamente después de conocer la prueba: eso convertiría esa prueba en otra validación. Cualquier nuevo experimento debe planificarse en desarrollo y requerirá una evaluación independiente para afirmar una mejora.

## Envío

El ajuste final sobre todo train.csv incluye las antiguas filas de prueba local solo DESPUÉS de cerrar selección y evaluación. Esto es correcto para el modelo de producción, pero ese modelo final no puede evaluarse honestamente sobre esas filas. Las métricas publicadas corresponden al modelo previo ajustado solo sobre desarrollo. eval.csv no participa en ningún fit: se verificó además que el vocabulario e IDF finales coinciden con un ajuste exclusivo sobre train.csv.

El modelo guardado reproduce las respuestas de submission_001.csv; esquema, etiquetas, IDs y orden son correctos. Ambos archivos permanecen idénticos, verificados por SHA-256. El CSV está listo para el primer envío manual; no se consultó Kaggle.

## Límites

No hay duplicados exactos o normalizados según la auditoría original. Esto no descarta paráfrasis, plantillas compartidas o dependencias de origen no documentadas. Sin metadatos de autor, producto o procedencia no se puede garantizar independencia por esas entidades. Tampoco se puede garantizar una puntuación privada ni ausencia absoluta de sobreajuste. Esta revisión no usa el texto de eval para decidir cambios en el modelo.

## Regularización: evidencia de la búsqueda original

Con el resto de parámetros seleccionados fijos:

- C=0.5: accuracy CV 0.714896.
- C=1: accuracy CV 0.719271.
- C=2: accuracy CV 0.725938.
- C=4: accuracy CV 0.731979.

Reducir C aumenta la regularización, pero no mejoró la accuracy de validación en esta comparación original. Se conserva el primer envío; esto no elimina la brecha observada ni garantiza que sea la mejor solución posible.
