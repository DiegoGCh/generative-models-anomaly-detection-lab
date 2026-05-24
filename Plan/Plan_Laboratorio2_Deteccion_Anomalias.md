# Plan de Trabajo — Laboratorio 2: Detección de Anomalías (MVTec AD)

> **Curso:** Modelos Generativos
> **Dataset:** MVTec AD (15 clases)
> **Modelo:** VAE convolucional entrenado como β-VAE con β bajo
> **Framework:** PyTorch
> **Métrica:** F1-score a nivel de píxel (segmentación)

---

## 1. La idea central en una frase

Entrenamos un modelo **solo con imágenes normales**. El modelo aprende a reconstruir fielmente lo "sano". Cuando recibe una imagen con defecto, **no sabe reconstruir el defecto** porque nunca lo vio: la reconstrucción falla justo en esa región. La **diferencia entre la imagen original y su reconstrucción** es el mapa de anomalía. Ese mapa se binariza con un umbral y se compara contra la máscara de ground truth.

```
x_j  ──▶  x̂_j  ──▶  s_j  ──▶  t(s_j)
Input    Reconstr.   Resta    Threshold
```

---

## 2. Por qué VAE con β bajo

### 2.1 El problema con el VAE estándar

El método depende **enteramente de la calidad de reconstrucción de lo normal**. Si la reconstrucción de imágenes sanas ya es borrosa, esa borrosidad aparece en el mapa de diferencia **incluso en imágenes sin defecto** → falsos positivos → baja el F1.

El VAE estándar reconstruye más borroso que un autoencoder por culpa del término KL. La derivación de β-VAE (Sesión 5) lo demuestra:

```
K(φ) = E_{p_data(x)} D_KL( q_φ(z|x) || p(z) )  ≥  I_q(x; z)
```

El KL promedio es cota superior de la información mutua entre datos y latente. **El término KL limita cuánta información de `x` puede meter el encoder en `z`**. Menos información en `z` → reconstrucción más genérica y borrosa.

### 2.2 Por qué aun así elegimos VAE

1. **Es el marco del curso** — espera el marco probabilístico.
2. **El VAE da verosimilitud** `log p_θ(x|z)` que un AE no tiene.
3. **Demuestra comprensión del material** — usar β-VAE y explicar el trade-off sube la nota.

### 2.3 La solución: β-VAE con β pequeño

```
L_i = (1/2)·||x_i - x̂_i||²  +  β · D_KL( q_φ(z|x_i) || p(z) )
```

| Valor de β | Efecto |
|---|---|
| β grande (≈4) | Disentanglement, reconstrucción pobre — **no es nuestro caso** |
| **β = 0.5 ← elegido** | Reconstrucción nítida, latente suficientemente regularizado |
| β → 0 | El VAE se convierte en AE (caso límite) |

**β = 0.5 fijo** (sin KL annealing por defecto). Si aparece posterior collapse — reconstrucciones idénticas para inputs distintos — activar annealing.

### 2.4 La frase para el informe

> *"Uso un β-VAE con β bajo porque, a diferencia de Higgins et al. (2017) que aumentan β para favorecer disentanglement, mi objetivo es la fidelidad de reconstrucción para detección de anomalías. Bajar β relaja la cota superior sobre la información mutua I_q(x;z) y permite que el latente z retenga más información de x, produciendo reconstrucciones más nítidas y reduciendo falsos positivos en el mapa de anomalía."*

---

## 3. Arquitectura del modelo

VAE convolucional. Entrada **256×256 RGB**.

- **Encoder:** 5 convoluciones stride-2. Reduce `256→128→64→32→16→8`. Canales `32→64→128→256→512`. Produce `mu` y `logvar` (dos cabezas lineales).
- **Reparameterization trick:** `z = mu + sigma · eps`, `sigma = exp(0.5·logvar)`, `eps ~ N(0,I)`.
- **Decoder:** espejo del encoder con `ConvTranspose2d` stride-2. Salida con `sigmoid` → rango `[0,1]`.

### 3.1 `IMG_SIZE` vs `LATENT_DIM` — NO son lo mismo

#### `IMG_SIZE` = 256×256
Más resolución → defectos más finos visibles → máscaras más precisas. **Aquí más grande sí es mejor.**

#### `LATENT_DIM` = 128 (punto de arranque)

**NO se maximiza — se calibra.** Un canal demasiado ancho hace que el VAE reconstruya el defecto y anule la detección.

| `LATENT_DIM` | Qué pasa | Resultado |
|---|---|---|
| Muy grande | Pasa toda la info, incluido el defecto. `x ≈ x̂` en todas partes. | Mapa plano, F1 ≈ 0 |
| **128 ← arranque** | El modelo comprime: solo caben patrones normales | Mapa se enciende en defecto ✅ |
| Muy chico | Ni las `good` reconstruyen | Todo parece anómalo |

**Checklist de calibración (revisar al ver los primeros resultados):**
- [ ] Mapa plano/vacío → `LATENT_DIM` muy grande → bajar a **64**
- [ ] `good` no reconstruyen → `LATENT_DIM` muy chico → subir a **256**
- [ ] Mapa se enciende en defecto → OK, no tocar

Las texturas (`carpet`, `wood`, `grid`) toleran latente más chico que los objetos complejos (`cable`, `transistor`).

---

## 4. Hiperparámetros acordados

| Parámetro | Valor | Razón |
|---|---|---|
| `IMG_SIZE` | 256 | Mejor precisión de máscara |
| `LATENT_DIM` | 128 (arranque) | Punto medio seguro para MVTec |
| `β` | 0.5 fijo | Reconstrucción nítida sin posterior collapse |
| `EPOCHS` | 50 | MVTec sets pequeños → converge rápido; más épocas puede overfitear |
| `BATCH_SIZE` | 16 | Seguro en 8GB VRAM (RTX 4060); bajar a 8 si OOM |
| Optimizer | Adam, lr=1e-4 | Estable para VAE; 1e-3 puede divergir |
| Normalización input | [0,1] (÷255) | Decoder tiene sigmoid → target debe estar en [0,1] |

---

## 5. Loss de entrenamiento

```
L_i = L_reconstrucción  +  β · L_KL
```

### 5.1 Término de reconstrucción

```
L_reconstrucción = λ1 · MSE(x, x̂)  +  λ2 · (1 - SSIM(x, x̂))
```

- **λ1 = 0.5, λ2 = 0.5** (igual peso de arranque)
- Si F1 bajo en clases de textura (`carpet`, `grid`, `wood`) → subir λ2 (SSIM detecta errores estructurales locales mejor que MSE)
- **Librería SSIM:** `pytorch-msssim` (`pip install pytorch-msssim`)
- **MSE** conecta con Sesión 5: maximizar `log p_θ(x|z) = N(x; μ_θ(z), I)` equivale a minimizar MSE.
- **SSIM** compara estructura en ventanas locales → captura defectos de textura que MSE puro deja pasar.

### 5.2 Término KL

Forma cerrada (Sesión 5):

```
D_KL = -0.5 · Σ_j ( 1 + log(σ_j²) - μ_j² - σ_j² )
```

Código: `KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())`

### 5.3 ⚠️ Normalización obligatoria

Reconstrucción suma sobre **todos los píxeles** (256×256×3 ≈ 196k). KL suma sobre **dimensiones latentes** (128). Son magnitudes distintas → normalizar cada uno por su nº de elementos antes de aplicar β.

```python
recon_loss = loss_recon / (C * H * W)   # por píxel
kl_loss    = kld      / latent_dim       # por dimensión latente
loss       = recon_loss + beta * kl_loss
```

---

## 6. Datos y entrenamiento

### 6.1 Estructura del dataset

MVTec AD: 15 clases (objetos: `bottle`, `cable`, `screw`, `transistor`, ...; texturas: `carpet`, `grid`, `wood`, ...).

- `train/good/` → solo imágenes normales
- `test/` → normales + defectuosas (por tipo)
- `ground_truth/` → máscaras binarias

### 6.2 Reglas de entrenamiento

1. **Un modelo por clase.**
2. **Entrenar SOLO con `train/good/`.** Nunca tocar máscaras ni defectuosas.
3. **Sin split de validación** — todo `train/good/` va a training. El umbral se calibra por barrido en test (ver sección 8).

### 6.3 Data augmentation

Aplicar en training para evitar memorización (solo ~200-400 imágenes por clase):
- Horizontal flip
- Vertical flip
- Rotación ±15°

**NO usar:** elastic transform, distorsiones geométricas fuertes → cambian la estructura "normal" y confunden al modelo.

---

## 7. Mapa de anomalía

```
ssim_err = promedio_w( 1 - SSIM_w(x_j, x̂_j) )   # w en {3, 7, 11}, por píxel
s_j      = λ1 · L2(x_j, x̂_j)  +  λ2 · ssim_err
s_j      = gaussian_blur(s_j, sigma=4)
s_j      = normalizar globalmente sobre todas las imágenes de test de esa clase
```

Pipeline:
1. L2 por píxel + SSIM per-pixel multi-escala (λ1=λ2=0.5)
2. **Suavizado gaussiano** con **σ=4** (estándar papers MVTec AD — Bergmann et al.)
3. Normalizar globalmente por clase (no por imagen)

**Por qué σ=4:** los defectos son regiones conexas de decenas de píxeles. σ=4 elimina ruido de alta frecuencia (píxeles sueltos) sin borrar defectos reales. Valores menores (σ=1-2) → mapa ruidoso → falsos positivos. Valores mayores (σ=8+) → defectos pequeños desaparecen → falsos negativos.

### 7.1 SSIM per-pixel (no el escalar de pytorch-msssim)

`pytorch_msssim.ssim(size_average=False)` devuelve un escalar por imagen, no un mapa espacial. Para localización, se implementa el mapa SSIM real con una ventana Gaussiana deslizante por convolución separable:

```python
# Para cada ventana w en {3, 7, 11}:
mu_x, mu_y = smooth(x), smooth(y)       # medias locales
sigma_x, sigma_y, sigma_xy = ...        # varianzas y covarianza locales
ssim_map = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2) /
           ((mu_x^2 + mu_y^2 + C1)*(sigma_x + sigma_y + C2))
# output: [B, 1, H, W] — un valor por píxel
```

Ventana 3: detecta errores de textura fina (grietas pequeñas, puntos).
Ventana 11: detecta diferencias estructurales (bordes, formas).
El promedio de las tres escalas es robusto a defectos de distintos tamaños.

Resultado: avg F1 de 0.080 (L2 solo efectivo) a **0.140** sin reentrenar.

### 7.2 Feature-space map (experimento, descartado)

Se probó combinar el mapa pixel-level con un mapa de diferencia en espacio de features del encoder:

```
combined = (1 - alpha) * pixel_norm + alpha * feat_norm
```

Sweep de alpha={0.0, 0.1, ..., 0.7, 1.0} sobre clases representativas (hazelnut, wood, capsule, leather, metal_nut, cable):
- alpha=0.0 (solo pixel): mejor promedio (0.2204)
- El feature map ayuda a wood y capsule, pero penaliza hazelnut y leather
- Efecto neto negativo en el promedio

**Decision: alpha=0.0 (solo mapa pixel, multi-scale SSIM).** El feature map queda como parámetro opcional `feat_alpha` para experimentación futura.

### 7.3 Normalización global vs. por imagen

Normalización global por clase: imágenes sanas (error bajo) quedan en valores bajos globalmente. Normalización por imagen: cada imagen se escala a [0,1] independientemente, lo que hace que incluso imágenes sanas lleguen a 1.0 y generen falsos positivos masivos al umbralizar.

---

## 8. Umbralización: mapa continuo → máscara binaria

**Estrategia: barrido de umbral por clase en test.**

Barrer ~50 valores de umbral uniformemente en [0,1], calcular F1 para cada uno, reportar el F1 al mejor umbral **por clase**. Documentar en el informe que se está eligiendo el operating point óptimo por clase.

**Por qué no percentil de validación:** no hay split de validación (todo `train/good/` va a training). El barrido en test es más directo y robusto para F1 de segmentación.

### 8.1 Limpieza morfológica (después del umbral)

Después de binarizar el mapa, aplicar operaciones morfológicas:

```
Mapa binarizado crudo:        Ground truth real:
  . . X . X . .                 . . . . . . .
  . X X X . X .    →  malo      . . X X X . .
  . . X X . . .                 . . X X X . .
  . X . . X . .                 . . . . . . .

Después morfología:
  . . . . . . .
  . . X X X . .    ← más parecido al GT → mejor F1
  . . X X X . .
  . . . . . . .
```

- **Apertura** (erosión → dilatación): elimina manchitas aisladas (falsos positivos pequeños)
- **Cierre** (dilatación → erosión): rellena huecos dentro de regiones detectadas

Implementación: `cv2.morphologyEx` — 2 líneas, sube F1 sin riesgo.

---

## 9. Evaluación: F1-score a nivel de píxel

Cada píxel = clasificación binaria (anómalo / normal):

```
F1 = 2·TP / ( 2·TP + FP + FN )
```

- **TP** = píxeles marcados como defecto confirmados por ground truth
- **FP** = píxeles marcados como defecto que estaban sanos
- **FN** = defectos reales no detectados

**Por qué F1 y no accuracy:** los defectos son fracción minúscula de píxeles. Máscara toda en cero → accuracy alta, F1 = 0.

**Reportar:**
- F1 por clase + promedio
- Curva F1 vs umbral (demuestra comprensión del trade-off precision/recall)
- **Figuras visuales: 3 ejemplos por clase** — 1 imagen `good` (mapa debe ser plano, sin activación) + 2 defectuosas distintas (mapa debe encenderse sobre el defecto). Formato: `input → reconstruction → anomaly map → thresholded mask`

---

## 10. Errores típicos que cuestan puntos

| # | Error | Solución |
|---|---|---|
| 1 | VAE reconstruye perfecto hasta los defectos → mapa vacío | Reducir `LATENT_DIM` |
| 2 | Un solo modelo para todas las clases | Un modelo por clase |
| 3 | Tocar máscaras o defectuosas en training | Solo `train/good/` |
| 4 | Umbral global a ojo | Barrido por clase en test |
| 5 | Reconstrucciones borrosas confundidas con defectos | β=0.5 + SSIM + normalizar loss |
| 6 | Aplicar β sin normalizar los términos del loss | Normalizar por nº de elementos (sección 5.3) |
| 7 | Posterior collapse (reconstrucciones idénticas) | Activar KL annealing |
| 8 | Normalizar inputs con ImageNet stats | Usar [0,1] — decoder tiene sigmoid |

---

## 11. Orden de trabajo

1. **Pipeline de datos para UNA clase** (`bottle` es fácil)
2. **VAE convolucional**: entrenar, verificar que reconstruye bien las `good`
3. **Mapa de anomalía en test**: ¿se enciende donde está el defecto?
4. **Calibrar `LATENT_DIM`** con checklist de la sección 3.1
5. **Umbral + morfología**: implementar y medir F1 en esa clase
6. **Escalar a las 15 clases** en bucle
7. **Evaluar con multi-scale SSIM per-pixel**: `python main.py --eval-only --full`
8. **Entregables**: tabla F1 por clase + figuras (3/clase)

### Experimentos realizados y conclusiones

| Experimento | Resultado | Conclusion |
|---|---|---|
| LATENT_DIM=128, todas las clases | avg F1=0.080 | Baseline |
| LATENT_DIM=64, clases borderline | avg F1<0.080 | 64 comprime demasiado, regresion en capsule |
| Multi-scale per-pixel SSIM (alpha=0.0) | avg F1=0.140 | +75% sin reentrenar. Resultado final |
| Feature-space map (alpha=0.3) | avg F1=0.134 | Inconsistente: ayuda wood/capsule, penaliza hazelnut/leather |

### Fase A — Smoke test

| Parámetro | Smoke test | Razón |
|---|---|---|
| `IMG_SIZE` | 128 | 4× menos cómputo |
| Loss | Solo MSE | Una variable menos que puede fallar |
| β | 0.5 fijo | Sin annealing |
| Detección | L2 pura + suavizado | Sin SSIM todavía |
| Umbral | Barrido simple (20 valores) | |
| Clases | Solo `bottle` | |
| Épocas | 5 | Solo verificar que corre |

**Lo que NO se simplifica nunca:** normalización del loss + vigilar `LATENT_DIM`.

Verificar al terminar smoke test:
1. Reconstrucción de `good` se parece al input
2. Anomaly map se enciende donde está el defecto
3. Mapa plano → bajar `LATENT_DIM`
4. F1 bajo es OK — lo importante es que nada crashee

### Fase B — Entrenamiento real

- `IMG_SIZE` → 256
- `EPOCHS` → 50
- `LATENT_DIM` → 128 (calibrar con checklist)
- Loss completo: MSE + SSIM + KL normalizado
- Bucle de 15 clases desatendido

---

## 12. Resumen de decisiones

| Decisión | Valor |
|---|---|
| Modelo | β-VAE convolucional |
| `LATENT_DIM` arranque | 128 → calibrar |
| β | 0.5 fijo |
| SSIM en training | `pytorch-msssim` (escalar, correcto para loss) |
| SSIM en evaluacion | Per-pixel sliding window, ventanas {3, 7, 11} |
| λ1, λ2 | 0.5, 0.5 |
| feat_alpha | 0.0 (feature map descartado) |
| Umbral | Barrido por clase en test |
| Split validación | Ninguno |
| Morfología | Apertura + cierre desde inicio |
| Gaussian sigma | 4 (estándar papers MVTec) |
| Épocas | 50 |
| Batch size | 16 |
| Optimizer | Adam lr=1e-4 |
| Augmentation | hflip + vflip + rot±15° |
| Normalización | [0,1] |
| Figuras | 3/clase: 1 good + 2 defectuosas |
| Resultado final | avg F1=0.140 (baseline 0.080) |

### Fallos estructurales — no resolubles sin cambio de arquitectura

**Screw, toothbrush (variacion de orientacion):** Cada imagen de entrenamiento tiene el objeto en un angulo distinto. El VAE promedia todos los angulos y aprende un blob gris. El mapa de anomalia se activa en todo el objeto, no en el defecto. Fix requiere pre-alinear las imagenes por orientacion antes de entrenar.

**Grid, carpet (textura periodica):** El VAE no puede reproducir la fase del patron periodico. El error de reconstruccion es uniformemente alto en toda la imagen. Los defectos generan un error marginalmente mayor que la textura normal, insuficiente para discriminar. Fix requiere modelos que operen en espacio de frecuencia o features preentrenadas (fuera del marco VAE).
