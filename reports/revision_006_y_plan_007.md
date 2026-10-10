# Revisión de la versión 006 y plan para la 007

Fecha: 8 de octubre de 2026. Métrica local exclusiva: F1 macro (`zero_division=0`).
No se accedió a Kaggle; los públicos provienen de capturas del usuario.

## 1. Verificación técnica de la 006 — formalmente correcta, decisión equivocada

- Huellas SHA-256 de `modelo_006.joblib`, `submission_006.csv` y fuentes: coinciden con
  `registro_006.json`; originales intactos.
- `submission_006.csv`: 3000 filas, `id,answer`, IDs únicos en orden de
  `sample_submission`, sin vacíos, etiquetas válidas; la recarga del modelo reproduce
  las predicciones. Cambia 116/3000 frente a 005.
- Protocolo: CV anidada 3×3; selección solo dentro del entrenamiento de cada fold
  externo; el `SafeStack` genera sus propios scores OOF internos (semilla 271) y
  aprende `QuantileTransformer` solo con el entrenamiento del meta-modelo (sin fugas
  evidentes). Convergencia exigida. Artefactos previos preservados. El cambio de orden
  de la regla de selección durante la exploración inicial está declarado y archivado.

Localmente el stacking superó a 005: CV 0.878010 vs 0.874163 y OOF 0.878421 vs
0.873382 (+0.0050), con brecha 0.1211 vs 0.1253. **Pero el público cayó de 0.88111
(005) a ~0.87 (006)**: la mejora local no se transfirió a eval.

## 2. Por qué bajó el público a pesar de la mejora local

1. **Folds quemados.** Desde la 003 se compara todo sobre los mismos 3 folds externos
   (semilla 137). Cada ronda elige la config que mejor les va a *esos* folds; la
   ventaja de +0.005 del stack está por debajo de la variación entre folds
   (desviación típica 0.004–0.005) y no era señal, era adaptación.
2. **El stack incluye bases débiles en el espacio de palabras** (LR 0.735 y NB 0.716
   solas). Su contribución ayudaba a los folds de desarrollo, pero no a eval.
3. **La regla de reemplazo era demasiado laxa** (+0.002 de mejora con tolerancia
   0.003): selecciona dentro del ruido de los folds. El resultado público es la
   primera evidencia directa de que a este margen las decisiones locales ya no
   predicen.
4. El público 005→004 había subido 0.02 exactos (0.86111→0.88111) siguiendo la misma
   tendencia local; la 006 rompió esa tendencia por primera vez.

**Decisión recomendada: conservar `submission_005.csv` como mejor envío conocido
(0.88111) y archivar la 006.** No hacer más selecciones sobre los folds 137.

## 3. Veredicto sobre las ideas propuestas

| # | Idea | Veredicto | Razón |
|---|---|---|---|
| 1 | Aislar cláusulas de opinión aprendiendo el "relleno" de las neutrales | **Posible, riesgo medio** | Solo usa etiquetas de train (permitido), pero las neutrales también contienen opinión y descartar cláusulas puede eliminar señal; el canal global ya conserva todo. Hipótesis a contrastar, no regla. |
| 2 | EM por cláusula: reetiquetar con la regla del conector hasta converger | **No en esta etapa** | Reetiquetar ejemplos está restringido («no relabelar a conveniencia») y «no imponer que un conector determine la etiqueta». Es pseudo-etiquetado autorreforzante: riesgo alto de sesgo circular. |
| 3 | Reglas de conector como features explícitas | **Parcialmente** | Como *características* es legítimo (ya hay segmentación por conectores); como *regla de etiqueta* está prohibido. Primero contrastar la hipótesis con evidencia OOF. |
| 4 | Léxico robusto: char n-gramas sin tildes, normalizar «-ísimo», marcar negación | **Sí, acotado** | La marcación de negación es NLP clásico y barata (S11 advierte del «no»). Char n-gramas: ya se quitaron en 005 con mejora; readaptarlos tiene valor esperado bajo. |
| 5 | Stacking SVM-por-oración + cláusulas con LR o HistGB | **Permitido, máxima cautela** | S13 lo cubre, pero la 006 acaba de mostrar que un stack seleccionado en folds quemados no transfiere. HistGB exige matriz densa (falló en prueba). Solo con protocolo de semillas frescas (§5). |
| 6 | Revisar ruido de etiquetas con cross_val_predict | **Análisis sí; corregir/eliminar NO** | Restringido: no modificar CSV ni relabelar ni eliminar filas. El análisis se ejecutó (§4). |
| 7 | Dos vistas: palabras (1–3) + char_wb (2–6) | **Posible, valor esperado bajo** | char_wb es opción estándar de scikit-learn; la evidencia 003→005 dice que el canal de caracteres aporta poco o resta en este corpus. |
| 8 | Limpieza ligera: repetidas, emojis, conservar tildes | **Sí, bajo riesgo** | Normalizar «buenooo»→«bueno» y tokens de emoji es clásico; el corpus casi no tiene emojis. Conservar tildes cambia el colapso actual (pocas tildes en los datos). |
| 9 | Marcar palabras tras «no/nunca/sin» y cláusula posterior a «pero/aunque» | **Sí, prioridad alta** | Ataca directamente el cuello positivo/negativo (§4): características de negación y contraste, sin reglas de etiqueta. |
| 10 | Lineales regularizados: LR, LinearSVC, SGD, NB-SVM | **Sí** | Todo scikit-learn; NB-SVM es un rescaleado lineal de TF-IDF aprendido de train (declarar el método). En espacio de palabras estos modelos rinden ~0.72-0.74 solos; utilidad solo como bases diversas de un stack. |
| 11 | Stacking con diversidad real + meta logístico OOF | **Permitido, máxima cautela** | Igual que #5. La diversidad es necesaria pero no suficiente (S13); la 006 lo demostró en el público. |
| 12 | Corregir o quitar etiquetas erróneas | **Prohibido en esta etapa** | Restricción explícita del prompt: no relabelar ni eliminar filas. |
| 13 | Sesgo de clase sobre probabilidades OOF | **Ya medido: ganancia ≈ 0** | La matriz no tiene problema de neutral (F1 0.988–0.991); la confusión neg↔pos es simétrica y el ajuste de umbral dio +0.0004 (S10). Descartado con evidencia. |

## 4. Auditoría de ruido de etiquetas (solo análisis, sin tocar datos)

Con los scores OOF del 005 (margen = distancia al segundo mejor):

- 1269 errores OOF; 264 (20,8 %) tienen margen > 1,0 y 604 (47,6 %) margen > 0,5.
- Los errores de mayor confianza **no parecen ruido de etiqueta, sino reseñas de
  aspecto mixto** donde el modelo se queda con el aspecto equivocado. Ejemplos:
  *«La funda no vale lo que cuesta. Por otro lado, la velocidad es justo l…»*
  (negativo, predijo positivo); *«Estoy feliz con el sistema de carga. La caja
  traía…»* (positivo, predijo negativo).
- Sí hay casos sospechosos aislados (*«Siento que tiré el dinero con la interfaz»*
  etiquetado positivo), pero son minoría; no autorizan relabelar ni prometer «varios
  puntos». El cuello real es **pesar cláusulas en conflicto**, no etiquetas rotas.
  Esto valida el foco de las ideas #1/#3/#9 sobre el problema correcto, pero sin
  reglas codificadas ni EM.

## 5. Plan 007 (fijado por escrito; requiere aprobación del usuario)

**Objetivo**: recuperar y superar 0.88111 sin repetir el error de la 006.

**Cambio de protocolo obligatorio** (lección de la 006):
- Evaluar candidatos sobre **semillas externas nuevas** (271, 599, 913) nunca usadas
  para selección; los folds 137 solo se reportan como referencia histórica.
- Reemplazar 005 solo si la candidata supera a 005 en **≥0.005** de F1 macro OOF en
  las semillas nuevas **y** no pierde en las 137, **y** gana en los 3 folds (o ≥2/3
  con brecha menor). Si no, se conserva 005 y se documenta.

**Candidatas acotadas (una ablación por eje, todas sobre el diseño ng1 de 005):**
1. `neg_scope`: marcar `neg_` las palabras en el alcance de «no/nunca/sin» (idea #9).
2. `post_contraste`: canal separado para la cláusula posterior a «pero/aunque/sin
   embargo» (idea #9, segunda parte).
3. `c1_min5`: C=1 y min_df=5 (regularización, S7/S8; en 006 dio CV 0.8758 con brecha
   0.1198, marginalmente mejor que 005).
4. `ref005` de referencia. (Opcional si hay presupuesto: char_wb 2–6 como canal
   adicional, idea #7.)

No se incluyen: EM/reeetiquetado, corrección de etiquetas, reglas de conector como
decisión, SMOTE, embeddings, boosting agresivo. Si ninguna candidata cumple el
criterio, se conserva 005 y se aprende de la evidencia.

## 6. Estado de puntajes públicos

| Versión | Público (captura) |
|---|---:|
| 002 | 0.73888 |
| 003 | 0.84333 |
| 004 | 0.86111 |
| 005 | 0.88111 |
| 006 | **~0.87 reportado por el usuario; falta valor exacto y captura para registrarlo** |

El nombre de la métrica oficial sigue sin ser visible en las capturas. La meta 0.90 no
se alcanzó en ninguna evaluación local honesta (máximo OOF medido: 0.8784 de la 006,
que además no transfirió al público). No prometer 0.90–0.95; el techo observado hasta
ahora en público es 0.88111.
