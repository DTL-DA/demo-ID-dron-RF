/* modelo.js · El modelo propuesto, tal como corre en este celular.
 *
 * Artículo: "Identificación de UAVs/Drones por RF con bajo costo computacional para procesadores ARM de 64 bits".
 * Entra:  los 17 espectrogramas de 80 ms de una captura de RF (índices de color de 0 a 255 del mapa viridis).
 * Sale:   la probabilidad de cada tipo de dron y la decisión de la captura.
 * Motor:  onnxruntime-web (WebAssembly). La red corre en el procesador de este dispositivo, sin servidor.
 */

export const CLASES = ["mavicAir2", "mavicAir2s", "mini4PRO"];
const LADO = 224;          // espectrograma de 224 x 224 píxeles
const SEGMENTOS = 17;      // 17 segmentos de 80 ms por captura

// ---------------------------------------------------------------- carga de la red y de los 20 pliegues

export async function cargarModelo(avisar = () => {}) {
  ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
  ort.env.wasm.numThreads = 1;
  const meta = await leerJSON("datos/capturas.json");
  avisar("red");
  const bytesRed = new Uint8Array(await (await fetch(`modelo/${meta.onnx.archivo}`)).arrayBuffer());
  const huella = await sha256(bytesRed);                 // huella de la red calculada aquí
  avisar("motor");
  const sesion = await ort.InferenceSession.create(bytesRed, { executionProviders: ["wasm"] });
  avisar("pliegues");
  const pliegues = await leerJSON("modelo/pliegues.json");
  const viridis = Uint8Array.from((await leerJSON("modelo/viridis.json")).flat());
  return { meta, sesion, pliegues, viridis, huella };
}

// Los 17 espectrogramas de una captura: 17 x 224 x 224 índices de color (archivo comprimido con gzip).
export async function leerCaptura(n) {
  const bytes = new Uint8Array(await (await fetch(`datos/capturas/c${String(n).padStart(3, "0")}.bin.gz`)).arrayBuffer());
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;          // el servidor ya lo descomprimió
  const flujo = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(flujo).arrayBuffer());
}

// Índices de color -> imagen RGB de 224 x 224 con la tabla viridis (la misma del artículo).
export function aRGB(indices, segmento, viridis) {
  const n = LADO * LADO, rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const k = indices[segmento * n + i] * 3;
    rgb[3 * i] = viridis[k]; rgb[3 * i + 1] = viridis[k + 1]; rgb[3 * i + 2] = viridis[k + 2];
  }
  return rgb;
}

// ════════════════════ MODELO PROPUESTO: MobileNetV3-Small + regresión logística ════════════════════

// 1) Red: MobileNetV3-Small con los pesos de ImageNet, congelada. Recibe espectrogramas RGB de 224 x 224 y
//    devuelve 576 rasgos por segmento. La normalización de ImageNet y el promedio global van dentro del ONNX.
export async function rasgosRed(sesion, rgb, n) {
  const entrada = new ort.Tensor("uint8", rgb, [n, LADO, LADO, 3]);
  const salida = await sesion.run({ espectrograma: entrada });
  return salida.rasgos.data;                                   // Float32Array de n x 576
}

// 2) Regresión logística del pliegue:  z = (x - media) / escala;  logits = coef · z + intercepto;  softmax.
export function regresionLogistica(x, p) {
  const logits = p.coef.map((w, c) => {
    let suma = p.intercepto[c];
    for (let j = 0; j < 576; j++) suma += w[j] * (x[j] - p.media[j]) / p.escala[j];
    return suma;
  });
  return normalizar(logits.map((v) => Math.exp(v - Math.max(...logits))));
}

// 3) Decisión de la captura: promedio de las log-probabilidades de sus 17 segmentos, renormalizado.
export function decidirCaptura(probas) {
  const logMedia = CLASES.map((_, c) =>
    probas.reduce((s, p) => s + Math.log(Math.min(Math.max(p[c], 1e-9), 1)), 0) / probas.length);
  return normalizar(logMedia.map(Math.exp));
}

// La captura completa, con el modelo del pliegue que dejó fuera a su dron: ese modelo nunca lo vio.
export async function identificarCaptura(modelo, captura, { alSegmento = async () => {}, lote = false } = {}) {
  const pliegue = modelo.pliegues[captura.unidad];          // regresión entrenada con los otros 19 drones
  const indices = await leerCaptura(captura.n);
  const probas = [];
  if (lote) {                                                 // los 17 segmentos en una sola llamada a la red
    const rgb = new Uint8Array(SEGMENTOS * LADO * LADO * 3);
    for (let s = 0; s < SEGMENTOS; s++) rgb.set(aRGB(indices, s, modelo.viridis), s * LADO * LADO * 3);
    const x = await rasgosRed(modelo.sesion, rgb, SEGMENTOS);
    for (let s = 0; s < SEGMENTOS; s++) probas.push(regresionLogistica(x.subarray(s * 576, (s + 1) * 576), pliegue));
  } else {                                                    // segmento por segmento, para verlo avanzar
    for (let s = 0; s < SEGMENTOS; s++) {
      const t0 = performance.now();
      const rgb = aRGB(indices, s, modelo.viridis);
      const x = await rasgosRed(modelo.sesion, rgb, 1);
      const p = regresionLogistica(x, pliegue);
      probas.push(p);
      await alSegmento({ s, rgb, p, ms: performance.now() - t0 });
    }
  }
  const decision = decidirCaptura(probas);
  return { probas, decision, tipo: argmax(decision) };
}

// ════════════════════ FIN DEL MODELO PROPUESTO ════════════════════

// ---------------------------------------------------------------- utilidades

export const argmax = (v) => v.indexOf(Math.max(...v));

function normalizar(v) {
  const suma = v.reduce((a, b) => a + b, 0);
  return v.map((x) => x / suma);
}

async function leerJSON(ruta) {
  const r = await fetch(ruta);
  if (!r.ok) throw new Error(`No se pudo leer ${ruta} (${r.status})`);
  return r.json();
}

async function sha256(bytes) {
  if (!globalThis.crypto?.subtle) return null;              // solo disponible en https o en localhost
  const h = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
