# Demo de identificación del tipo de dron por RF

Página estática que ejecuta en el navegador, y en el procesador del dispositivo que la abre, el modelo propuesto en el
artículo *Identificación de UAVs/Drones por RF con bajo costo computacional para procesadores ARM de 64 bits*:
MobileNetV3-Small congelada con los pesos de ImageNet, seguida de una regresión logística, sobre espectrogramas de 80 ms
de capturas de RF del conjunto DRFF-R2 (vuelo estacionario, 5,745 GHz).

**Abrir la demo:** https://dtl-da.github.io/demo-ID-dron-RF/

## Qué hace

- Cada una de las 113 capturas se clasifica con el modelo del pliegue que dejó fuera a su dron (validación anidada,
  20 pliegues), así que el modelo nunca vio ese dron al entrenar.
- La red corre con onnxruntime-web (WebAssembly) y la regresión logística con los coeficientes de cada pliegue
  (`modelo/pliegues.json`). La decisión de la captura es el promedio de las log-probabilidades de sus 17 segmentos.
- La página calcula la huella SHA-256 de la red que descarga y la compara con la publicada.
- El botón *Evaluar las 113* reproduce en el dispositivo la exactitud del artículo: 111 de 113 capturas (98,2 %),
  F1 macro de 0,984 y exactitud por dron de 96,7 %. Los dos errores son del dron mavicAir2s_7.

## Alcance

El dispositivo recibe el espectrograma ya calculado, como índices de color de 0 a 255 del mapa viridis, porque la señal
de RF cruda pesa cerca de 1 GB por captura. Desde la imagen hasta la decisión, todo corre en el dispositivo. Es
identificación entre tres tipos de dron conocidos (mavicAir2, mavicAir2s y mini4PRO): no detecta si hay un dron ni
reconoce tipos nuevos.

## Archivos

| Ruta | Contenido |
|---|---|
| `modelo.js` | El modelo propuesto en el navegador: red, regresión logística y decisión por captura |
| `interfaz.js`, `index.html`, `estilo.css` | La pantalla de la demo |
| `modelo/mobilenet_v3_small_rasgos.onnx` | MobileNetV3-Small (parte convolucional, IMAGENET1K_V1) con la normalización de ImageNet y el promedio global: 576 rasgos por segmento |
| `modelo/pliegues.json` | Media, escala, coeficientes e intercepto de la regresión de cada uno de los 20 pliegues |
| `datos/capturas.json`, `datos/capturas/*.bin.gz` | Las 113 capturas: metadatos y los 17 espectrogramas de cada una |
| `codigo/modelo_demo.py`, `codigo/construir_demo.py` | El entrenamiento de los pliegues, la exportación a ONNX y las verificaciones contra el artículo |
| `vendor/` | onnxruntime-web 1.31.1 (licencia MIT) |

## Verificación al construir

`construir_demo.py` comprueba, antes de escribir los datos, que las regresiones reproducen las probabilidades por
segmento del artículo (diferencia máxima de 1,7 × 10⁻⁵, por el redondeo a float32 del archivo de resultados) y que la red
exportada a ONNX, partiendo de las imágenes, da la misma decisión que PyTorch en las 113 capturas (111 aciertos).

Datos: DRFF-R2 (Zheng et al., 2026), CC BY 4.0. Autor: Donny Torres, Maestría en Analítica de Datos, Universidad del
Norte, 2026.
