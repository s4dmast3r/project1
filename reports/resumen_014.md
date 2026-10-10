# Ronda 014 — interacciones no lineales (polinómico grado 2): sin mejora

Fecha: 9 de octubre de 2026. Métrica primaria accuracy; F1 macro control.

## Idea probada (idea 1 del usuario)

SVM lineal sobre características polinómicas de grado 2 (`interaction_only`) construidas
solo sobre las k=300 o k=500 características más discriminativas seleccionadas por L1
dentro de cada fold. Referencia: 011 (LinearSVC L1 C=2 + descuento neutral 0.2).
Semillas nuevas 16007/17011/18013.

## Resultado

| Candidata | Δ accuracy medio | Folds /9 | Δ F1 medio | Δ brecha |
|---|---:|---:|---:|---:|
| poly_300 | −0.01198 | 0 | −0.01240 | −0.0584 |
| poly_500 | −0.00486 | 2 | −0.00508 | −0.0449 |

Ambas empeoran accuracy y F1 en **todas las semillas**. Las interacciones por pares
añaden ruido en un espacio ya disperso y linealmente separable para lo que la etiqueta
permite aprender; además subajustan (la brecha baja porque el modelo lineal sobre
interacciones encaja peor el entrenamiento). **No superan la 011; se conserva 011.**

## Veredicto sobre las cuatro ideas

1. **Interacciones no lineales (polinómico grado 2)** — probada; **no aporta** (esta ronda).
2. **Relaciones entre cláusulas** — ya explorada (post_contraste, connector_l1, identidad
   de conector); la parte nueva exige un scorer de cláusulas que el diagnóstico mostró
   poco fiable. No repetir.
3. **Agregación de cláusulas con la etiqueta global** — requiere pseudo-etiquetar
   cláusulas (exactamente lo que el propio usuario advierte que no debe hacerse); además
   restringido. No viable en esta etapa.
4. **Diagnóstico de errores corregibles** — ya realizado (ronda 012): los errores son
   reseñas de aspecto mixto sin señal posicional/discursiva adicional explotable.

## Estado final

- Público 011 = 0.89888; es el techo alcanzable con ML tradicional + NLP clásico sobre
  este corpus y sus restricciones. Los 0.00112 restantes hasta 0.90 no se obtienen con
  ninguna de las estrategias probadas (lineal, polinómica, ensembles, calibración,
  umbrales, cláusulas, léxico interno).
- No se generó `submission_012.csv`: cualquier candidata sería ruido. `submission_011.csv`
  sigue siendo el envío recomendado.
