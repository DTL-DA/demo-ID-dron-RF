// interfaz.js · La pantalla de la demo. Las cuentas del modelo están en modelo.js.
import { CLASES, argmax, cargarModelo, identificarCaptura } from "./modelo.js";

const $ = (id) => document.getElementById(id);
const COLORES = ["var(--c0)", "var(--c1)", "var(--c2)"];
const ARTICULO = { exactitud: 0.982, f1: 0.984, porDron: 0.967 };   // Tabla III y sección V.A del artículo
let modelo = null, detener = false, ocupado = false;

// ---------------------------------------------------------------- formato en español

const fmt = (x, d = 3) => x.toFixed(d).replace(".", ",");
const pct = (x, d = 1) => `${fmt(100 * x, d)} %`;
function cientifica(x) {
  if (x === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(x))), m = x / 10 ** e;
  const sup = String(e).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
  return `${fmt(m, 1)} × 10${sup}`;
}

// ---------------------------------------------------------------- 1. carga

function marcar(id, estado) {
  const li = $(id);
  li.classList.remove("cargando", "listo", "fallo");
  if (estado) li.classList.add(estado);
}

async function iniciar() {
  $("leyenda").innerHTML = CLASES.map((c, i) => `<span><i style="background:${COLORES[i]}"></i>${c}</span>`).join("");
  infoDispositivo().then((t) => ($("dispositivo").textContent = t));
  mostrarCodigo("modelo.js", "js");
  if (typeof DecompressionStream === "undefined") {
    return fallo("Este navegador no puede descomprimir los datos (DecompressionStream). Actualiza el navegador.");
  }
  const etapas = { red: ["e-red"], motor: ["e-motor", "e-red"], pliegues: ["e-pliegues", "e-motor"] };
  try {
    modelo = await cargarModelo((etapa) => {
      const [actual, anterior] = etapas[etapa];
      if (anterior) marcar(anterior, "listo");
      marcar(actual, "cargando");
    });
  } catch (e) {
    ["e-red", "e-motor", "e-pliegues"].forEach((id) => $(id).classList.contains("cargando") && marcar(id, "fallo"));
    return fallo(`No se pudo cargar el modelo: ${e.message}`);
  }
  marcar("e-pliegues", "listo");
  $("tam-red").textContent = `${fmt(modelo.meta.onnx.bytes / 1e6, 1)} MB`;
  const publicada = modelo.meta.onnx.sha256;
  if (modelo.huella === null) {
    $("huella").textContent = "no disponible sin https";
    marcar("e-huella", null);
  } else {
    const coincide = modelo.huella === publicada;
    $("huella").textContent = `${modelo.huella.slice(0, 16)}… ${coincide ? "✓ coincide con la publicada" : "✗ NO coincide"}`;
    $("huella").title = modelo.huella;
    marcar("e-huella", coincide ? "listo" : "fallo");
  }
  const mb = modelo.meta.capturas.reduce((s, c) => s + c.bytes, 0) / 1e6;
  $("mb-total").textContent = `${Math.round(mb)} MB`;
  poblarDrones();
  ["sel-dron", "sel-captura", "btn-azar", "btn-correr", "btn-todas"].forEach((id) => ($(id).disabled = false));
}

function fallo(texto) {
  $("error-carga").textContent = texto;
  $("error-carga").hidden = false;
}

async function infoDispositivo() {
  const partes = [];
  try {
    const v = await navigator.userAgentData?.getHighEntropyValues(["architecture", "bitness", "platform", "model"]);
    if (v?.architecture) partes.push(`arquitectura ${v.architecture}${v.bitness ? ` de ${v.bitness} bits` : ""}`);
    if (v?.platform) partes.push(v.platform);
    if (v?.model) partes.push(v.model);
  } catch { /* el navegador no lo informa */ }
  if (!partes.length && navigator.platform) partes.push(navigator.platform);
  if (navigator.hardwareConcurrency) partes.push(`${navigator.hardwareConcurrency} núcleos`);
  return partes.length ? `Tu dispositivo, según el navegador: ${partes.join(" · ")}.` : "";
}

// ---------------------------------------------------------------- 2. elegir

function capturasDe(unidad) { return modelo.meta.capturas.filter((c) => c.unidad === unidad); }

function poblarDrones() {
  $("sel-dron").innerHTML = modelo.meta.unidades
    .map((u) => `<option value="${u}">${u} · ${capturasDe(u).length} capturas</option>`).join("");
  poblarCapturas();
}

function poblarCapturas() {
  $("sel-captura").innerHTML = capturasDe($("sel-dron").value)
    .map((c) => `<option value="${c.n}">${c.canal} · ${c.receptor} · ${c.dia}</option>`).join("");
}

function alAzar() {
  const c = modelo.meta.capturas[Math.floor(Math.random() * modelo.meta.capturas.length)];
  $("sel-dron").value = c.unidad;
  poblarCapturas();
  $("sel-captura").value = String(c.n);
}

// ---------------------------------------------------------------- 3. identificar una captura

function prepararBarras() {
  $("barras-segmentos").innerHTML = Array.from({ length: 17 },
    () => `<div class="barra-seg">${COLORES.map((c) => `<span style="background:${c};height:0"></span>`).join("")}</div>`).join("");
}

function dibujar(rgb) {
  const ctx = $("lienzo").getContext("2d"), img = ctx.createImageData(224, 224);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    img.data[j] = rgb[i]; img.data[j + 1] = rgb[i + 1]; img.data[j + 2] = rgb[i + 2]; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function barrasDecision(p, ganador) {
  return CLASES.map((c, i) => `
    <div class="fila-barra ${i === ganador ? "ganador" : ""}">
      <span class="nombre">${c}</span>
      <div class="pista"><div style="width:${100 * p[i]}%;background:${COLORES[i]}"></div></div>
      <span class="valor">${fmt(p[i], 4)}</span>
    </div>`).join("");
}

async function correr() {
  if (ocupado) return;
  ocupado = true;
  bloquear(true);
  const captura = modelo.meta.capturas.find((c) => c.n === Number($("sel-captura").value));
  const pliegue = modelo.meta.unidades.indexOf(captura.unidad) + 1;
  $("resultado").hidden = false;
  $("decision").hidden = true;
  prepararBarras();
  $("resultado").scrollIntoView({ behavior: "smooth", block: "start" });
  const barras = $("barras-segmentos").children, tiempos = [];
  try {
    const r = await identificarCaptura(modelo, captura, {
      alSegmento: async ({ s, rgb, p, ms }) => {
        tiempos.push(ms);
        dibujar(rgb);
        [...barras].forEach((b, k) => b.classList.toggle("actual", k === s));
        [...barras[s].children].forEach((span, i) => (span.style.height = `${100 * p[i]}%`));
        $("pie-espectro").textContent = `Segmento ${s + 1} de 17 · 80 ms · ${CLASES[argmax(p)]} (${fmt(Math.max(...p), 2)})`;
        $("tiempo").textContent = `Red + regresión en tu celular: ${Math.round(ms)} ms en este segmento.`;
        await new Promise(requestAnimationFrame);
      },
    });
    const media = tiempos.reduce((a, b) => a + b, 0) / tiempos.length;
    $("tiempo").textContent = `Red + regresión en tu celular: ${Math.round(media)} ms por segmento en promedio ` +
      `(${fmt(tiempos.reduce((a, b) => a + b, 0) / 1000, 2)} s los 17).`;
    [...barras].forEach((b) => b.classList.remove("actual"));
    mostrarDecision(captura, r, pliegue);
  } catch (e) {
    $("tiempo").textContent = `Error: ${e.message}`;
  } finally {
    ocupado = false;
    bloquear(false);
  }
}

function mostrarDecision(captura, r, pliegue) {
  const acierta = r.tipo === captura.real;
  $("barras-decision").innerHTML = barrasDecision(r.decision, r.tipo);
  $("veredicto").innerHTML = `El modelo dice <b>${CLASES[r.tipo]}</b>. El tipo real es <b>${CLASES[captura.real]}</b>: ` +
    (acierta ? `<span class="si">acierta ✓</span>` : `<span class="no">se equivoca ✗</span>`);
  const dif = Math.max(...r.decision.map((v, i) => Math.abs(v - captura.p_articulo[i])));
  $("comparacion").innerHTML = `Probabilidad de ${CLASES[r.tipo]}: en tu celular <b>${fmt(r.decision[r.tipo], 6)}</b> · ` +
    `en la validación del artículo <b>${fmt(captura.p_articulo[r.tipo], 6)}</b>. Diferencia máxima entre las tres ` +
    `clases: ${cientifica(dif)}, por redondeo de punto flotante. La probabilidad no está guardada en la página: ` +
    `se acaba de calcular aquí.` +
    (acierta ? "" : ` Este es uno de los dos errores del modelo en las 113 capturas, ambos del dron mavicAir2s_7.`);
  const p = modelo.pliegues[captura.unidad];
  $("txt-pliegue").textContent = `Modelo del pliegue ${pliegue} de 20: entrenado con 19 drones, sin ${captura.unidad} ` +
    `(C = ${p.C} elegido en la validación interna)`;
  $("lista-entreno").innerHTML = modelo.meta.unidades.filter((u) => u !== captura.unidad).map((u) => `<li>${u}</li>`).join("");
  $("decision").hidden = false;
}

function bloquear(si) {
  ["btn-correr", "btn-azar", "btn-todas", "sel-dron", "sel-captura"].forEach((id) => ($(id).disabled = si));
}

// ---------------------------------------------------------------- 4. evaluar las 113

async function evaluarTodas() {
  if (ocupado) return;
  ocupado = true; detener = false;
  bloquear(true);
  $("btn-parar").hidden = false;
  $("progreso-todas").hidden = false;
  $("metricas").hidden = true;
  const caps = modelo.meta.capturas, res = [], t0 = performance.now();
  try {
    for (const c of caps) {
      if (detener) break;
      const r = await identificarCaptura(modelo, c, { lote: true });
      res.push({ real: c.real, pred: r.tipo, unidad: c.unidad, captura: c });
      const ok = res.filter((x) => x.real === x.pred).length;
      $("barra-todas").style.width = `${(100 * res.length) / caps.length}%`;
      $("txt-todas").textContent = `${res.length} de ${caps.length} capturas · ${ok} aciertos · ` +
        `${fmt((performance.now() - t0) / 1000, 0)} s`;
    }
    mostrarMetricas(res, caps.length, (performance.now() - t0) / 1000);
  } catch (e) {
    $("txt-todas").textContent = `Error: ${e.message}`;
  } finally {
    ocupado = false;
    bloquear(false);
    $("btn-parar").hidden = true;
  }
}

function calcularMetricas(res) {
  const M = CLASES.map(() => CLASES.map(() => 0));
  res.forEach((r) => M[r.real][r.pred]++);
  const f1 = CLASES.map((_, c) => {
    const tp = M[c][c], fp = M.reduce((s, fila) => s + fila[c], 0) - tp, fn = M[c].reduce((a, b) => a + b, 0) - tp;
    const pr = tp + fp ? tp / (tp + fp) : 0, re = tp + fn ? tp / (tp + fn) : 0;
    return pr + re ? (2 * pr * re) / (pr + re) : 0;
  });
  const drones = [...new Set(res.map((r) => r.unidad))];
  const porDron = drones.map((u) => {
    const d = res.filter((r) => r.unidad === u);
    return d.filter((r) => r.real === r.pred).length / d.length;
  });
  return {
    M, aciertos: res.filter((r) => r.real === r.pred).length,
    exactitud: res.filter((r) => r.real === r.pred).length / res.length,
    f1: f1.reduce((a, b) => a + b, 0) / 3,
    porDron: porDron.reduce((a, b) => a + b, 0) / porDron.length,
  };
}

function mostrarMetricas(res, total, segundos) {
  if (!res.length) return;
  const m = calcularMetricas(res), completo = res.length === total;
  const errores = res.filter((r) => r.real !== r.pred);
  $("metricas").innerHTML = `
    <p class="nota">${completo ? "Las 113 capturas" : `Resultado parcial: ${res.length} de ${total} capturas`}, calculadas
      en tu celular en ${fmt(segundos, 0)} s.</p>
    <div class="cifras">
      <div class="cifra"><b>${m.aciertos}/${res.length}</b><span>exactitud ${pct(m.exactitud)}</span>
        <small>artículo: ${pct(ARTICULO.exactitud)}</small></div>
      <div class="cifra"><b>${fmt(m.f1)}</b><span>F1 macro</span><small>artículo: ${fmt(ARTICULO.f1)}</small></div>
      <div class="cifra"><b>${pct(m.porDron)}</b><span>exactitud por dron</span>
        <small>artículo: ${pct(ARTICULO.porDron)}</small></div>
    </div>
    <p class="rotulo">Matriz de confusión por captura (filas: tipo real; columnas: decisión del modelo)</p>
    <table class="confusion"><tr><th></th>${CLASES.map((c) => `<th>${c}</th>`).join("")}</tr>
      ${m.M.map((fila, i) => `<tr><th>${CLASES[i]}</th>${fila.map((v, j) =>
        `<td class="${i === j ? "diag" : v ? "fuera" : ""}">${v}</td>`).join("")}</tr>`).join("")}
    </table>
    <p class="nota">${errores.length ? `Errores: ${errores.map((e) =>
      `${e.unidad} (${e.captura.canal} · ${e.captura.receptor} · ${e.captura.dia}) tomado por ${CLASES[e.pred]}`).join("; ")}.`
      : "Sin errores."}</p>`;
  $("metricas").hidden = false;
}

// ---------------------------------------------------------------- 5. visor de código

const PALABRAS = /\b(const|let|var|return|async|await|function|export|import|from|for|of|if|else|new|throw|try|catch|def|class|with|in|not|and|or|None|True|False|lambda|assert|self)\b/;

function escapar(t) { return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

function colorear(codigo) {
  const patron = new RegExp(`("(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.)*'|\`(?:[^\`\\\\]|\\\\.)*\`)|${PALABRAS.source}|\\b(\\d+(?:\\.\\d+)?(?:e-?\\d+)?)\\b`, "g");
  let salida = "", ultimo = 0;
  for (const m of codigo.matchAll(patron)) {
    salida += escapar(codigo.slice(ultimo, m.index));
    const clase = m[1] ? "cad" : m[2] ? "pal" : "num";
    salida += `<span class="${clase}">${escapar(m[0])}</span>`;
    ultimo = m.index + m[0].length;
  }
  return salida + escapar(codigo.slice(ultimo));
}

function partirComentario(linea, lenguaje) {
  const marca = lenguaje === "js" ? "//" : "#";
  let comilla = null;
  for (let i = 0; i < linea.length; i++) {
    const ch = linea[i];
    if (comilla) { if (ch === "\\") i++; else if (ch === comilla) comilla = null; continue; }
    if (ch === '"' || ch === "'" || ch === "`") comilla = ch;
    else if (linea.startsWith(marca, i)) return [linea.slice(0, i), linea.slice(i)];
  }
  return [linea, ""];
}

async function mostrarCodigo(archivo, lenguaje) {
  document.querySelectorAll(".pestanas button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.archivo === archivo)));
  const visor = $("visor");
  try {
    const texto = await (await fetch(archivo, { cache: "no-store" })).text();
    let enBloque = false, enPropuesto = false;
    const lineas = texto.replace(/\r/g, "").split("\n").map((linea, k) => {
      let html;
      const delimitaBloque = lenguaje === "js" ? "/*" : '"""';
      const t = linea.trim();
      if (enBloque || t.startsWith(delimitaBloque)) {
        const cierra = lenguaje === "js" ? t.includes("*/") : (enBloque ? t.includes('"""') : t.split('"""').length > 2);
        enBloque = !cierra;
        html = `<span class="com">${escapar(linea)}</span>`;
      } else {
        const [codigo, comentario] = partirComentario(linea, lenguaje);
        html = colorear(codigo) + (comentario ? `<span class="com">${escapar(comentario)}</span>` : "");
      }
      const inicio = linea.includes("MODELO PROPUESTO:");
      if (inicio) enPropuesto = true;
      const clase = enPropuesto ? (inicio || linea.includes("FIN DEL MODELO PROPUESTO") ? "linea propuesto marca" : "linea propuesto") : "linea";
      if (linea.includes("FIN DEL MODELO PROPUESTO")) enPropuesto = false;
      return `<span class="${clase}" data-n="${k + 1}">${html || " "}</span>`;
    });
    visor.innerHTML = `<code>${lineas.join("")}</code>`;
    visor.dataset.archivo = archivo;
  } catch (e) {
    visor.innerHTML = `<code>No se pudo leer ${archivo}: ${escapar(e.message)}</code>`;
  }
}

async function irAlModelo() {
  if ($("visor").dataset.archivo !== "modelo.js") await mostrarCodigo("modelo.js", "js");
  const primera = $("visor").querySelector(".propuesto");
  if (!primera) return;
  $("visor").scrollTop = primera.offsetTop - 8;
  $("codigo").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- eventos

$("sel-dron").addEventListener("change", poblarCapturas);
$("btn-azar").addEventListener("click", alAzar);
$("btn-correr").addEventListener("click", correr);
$("btn-todas").addEventListener("click", evaluarTodas);
$("btn-parar").addEventListener("click", () => (detener = true));
$("btn-ir-modelo").addEventListener("click", irAlModelo);
$("btn-ajustar").addEventListener("click", () => {
  const si = $("visor").classList.toggle("ajustar");
  $("btn-ajustar").setAttribute("aria-pressed", String(si));
});
document.querySelectorAll(".pestanas button").forEach((b) =>
  b.addEventListener("click", () => mostrarCodigo(b.dataset.archivo, b.dataset.lenguaje)));

iniciar();
