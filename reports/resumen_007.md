# Experimento 007 — negación, contraste y estabilidad

La ronda se ejecutó con cuatro candidatos, tres semillas y tres folds por semilla. Todos los scores intervienen en la selección; no son una evaluación independiente. Se preservó la prueba local antigua. Cambiar semillas no elimina el historial de exploración.

| Candidato | F1 macro OOF medio entre semillas | Brecha media |
|---|---:|---:|
| c1_min5 | 0.873633 | 0.121717 |
| neg_scope | 0.873354 | 0.126302 |
| post_contraste | 0.873522 | 0.126107 |
| ref005 | 0.874195 | 0.125401 |

## Criterios y decisión

Cada semilla: delta F1 OOF >=0.005 vs005 y ganar 3 folds o >=2 con menor brecha media. Debe cumplir las tres semillas. Luego no perder F1 OOF en semilla histórica137. Entre aprobados mayor F1 medio en semillas nuevas; empate exacto menor brecha.

Ganadora: ninguna. CSV nuevo creado: False.
Conservar005: ninguna candidata cumple todos los criterios previos. No generar un CSV equivalente ni reducir los umbrales después de observar resultados.

## Interpretación y límites

Los canales añadidos son características, no reglas de etiqueta: no se obliga a que una negación invierta la clase ni a que la cláusula posterior al contraste domine la reseña. La negación se aproxima con tres tokens hasta puntuación o corte de contraste; no constituye análisis sintáctico completo. El canal post_contraste abarca el segmento posterior a pero/aunque/sin embargo hasta la puntuación de oración. Ambos conservan toda la representación base de 005. Peso adicional 0,35, min_df=3 y hasta 30000 rasgos se fijaron sin optimizar contra resultados.

C=1/min_df=5 combina dos cambios ya propuestos en el documento; no aísla causalmente cuál de ellos explica diferencias. El opcional char_wb no se ejecutó para mantener el presupuesto. No se borran ni reetiquetan reseñas; no se usan datos externos o sintéticos.

Se exigió +0,005 en CADA semilla, no solo en el promedio. En cada una también se exigieron tres victorias de fold, o al menos dos y brecha menor. La comparación histórica137 solo se ejecuta si una candidata supera esos requisitos. Si se usa, es otra puerta de selección, no prueba confirmatoria. La semilla271 ya se usó dentro del stacking006, aunque no como partición externa de esa versión.

El documento del auditor atribuye la caída pública de006 a adaptación a folds. Es una hipótesis plausible, no causalidad demostrada por un score público. El tamaño del delta respecto a la desviación entre folds tampoco basta para concluir señal o ruido: haría falta analizar diferencias emparejadas e incertidumbre, considerando adaptación. El documento cita ~0,87 sin captura exacta disponible en este turno; no se registra como valor exacto confirmado.

El público confirmado de005 es0,88111. La plataforma no fue consultada. No se extrapolan puntajes ni se garantiza0,90. F1 macro sigue siendo el criterio local por instrucción del usuario; las capturas no verifican la fórmula oficial.

## Reproducibilidad

Tiempo medido del experimento: 159.1 segundos. Archivos anteriores intactos: True.
Entrenamiento: `.venv/bin/python -m src.experimento_007`. Se detiene si ya existe registro_007.json para evitar sobrescribir resultados accidentalmente; archivar explícitamente la ejecución previa antes de repetir. Notebook: `.venv/bin/python src/informe_007.py`. Modelos anteriores y módulos no se modifican.
