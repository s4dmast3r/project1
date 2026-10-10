# Ronda 015 (diagnóstico) — aislamiento: la selección, no las interacciones, causó la pérdida

Fecha: 9 de octubre de 2026. C fijado en 2.0; bloque polinómico L2-normalizado por
fila (escala controlada); se reporta accuracy con argmax directo y con descuento
neutral 0.2. Semillas nuevas 19001/20011/21013. Métrica primaria accuracy.

## Tabla (media de 3 semillas)

| Candidata | acc (argmax) | acc (d=0.2) | F1 (argmax) |
|---|---:|---:|---:|
| ref011 (completa) | 0.8999 | 0.9013 | 0.9042 |
| sel500 (solo top-500) | 0.8947 | 0.8956 | 0.8990 |
| sel500_poly (top-500 + productos) | 0.8944 | 0.8942 | 0.8991 |
| full_polyblock (completa + bloque de productos) | 0.8949 | 0.8947 | 0.8996 |

## Lectura rigurosa

1. **Recortar a 500 características explica casi toda la pérdida** de la ronda 014:
   ref011 0.8999 → sel500 0.8947 (−0.0052).
2. **Los productos no aportan nada**: sel500_poly (0.8944) ≈ sel500 (0.8947). Añadir
   interacciones sobre las 500 seleccionadas no recupera ni suma.
3. **Conservando la representación completa, las interacciones empeoran**:
   full_polyblock (0.8949) < ref011 (0.8999), a pesar de la escala controlada.
   El bloque de interacciones añade ruido que la L1 no poda sin costo.
4. **El descuento neutral no transfiere al modelo polinómico**: en ref011 suma
   (+0.0013), en sel500_poly resta (−0.0002). Confirma que la escala de márgenes
   cambia; es un punto menor ahora que la vía está descartada.

## Correcciones a la interpretación previa

- **Correcto**: la 014 confundió selección, interacciones, C y descuento. Este
  diagnóstico la descompone: el culpable de la caída fue el **recorte de rasgos**,
  no las interacciones. Con esto, la hipótesis de interacciones queda rechazada con
  evidencia limpia (no con la confusión anterior).
- **Correcto**: agregar cláusulas no exige pseudo-etiquetarlas; se pueden combinar
  sus representaciones (inicio, fin, posterior a contraste) entrenando solo con la
  etiqueta de la reseña. En nuestro diseño esas señales posicionales/dis cursivas ya
  están presentes (start/end, post_contraste), y no se probó aquí un "agregador" de
  cláusulas distinto; no afirmaré que sea redundante sin contrastarlo.
- **Correcto**: que los errores mixtos sean un grupo difícil no demuestra que sean
  irreducibles; solo lo identifica. La afirmación "techo absoluto" era demasiado
  fuerte.

## Trazabilidad

`registro_015.json` incluye `source_sha256` y `elapsed_seconds` (390.6 s), ausentes en
la 014. Los resultados de esta ronda son reproducibles desde `experimento_015.py`.

## Decisión

Ninguna variante supera a ref011. Se conserva **011** (público 0.89888). La vía de
interacciones polinómicas queda ahora cerrada con una ablación limpia, no por una
prueba confundida.
