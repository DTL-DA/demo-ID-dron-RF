"""Construye la demo de identificación que corre en el celular (sitio estático en demo_app/sitio).

Pasos, cada uno verificado contra los resultados del artículo:
  1. Entrena la regresión logística de los 20 pliegues con los rasgos de la caché (MobileNetV3-Small, espectrograma
     de NumPy) y comprueba que reproduce las probabilidades por segmento de reports/espectrograma_numpy.
  2. Exporta la red a ONNX y comprueba con onnxruntime que, partiendo de las imágenes, se obtienen las mismas
     decisiones por captura (111 de 113).
  3. Escribe los datos de la página: índices de color de los 17 espectrogramas de cada captura (gzip), la tabla
     viridis, los parámetros de los 20 pliegues y la huella SHA-256 de la red.

Uso:  DRFFR2_CACHE=<caché de drffr2-green> python construir_demo.py
      (DRFFR2_REPO apunta al repositorio drffr2-green; por defecto, la carpeta hermana ../drffr2-green)
"""
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import modelo_demo as md

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
if "DRFFR2_CACHE" not in os.environ:
    sys.exit("Falta la variable DRFFR2_CACHE: la carpeta de cachés de drffr2-green (imágenes y rasgos).")
CACHE = Path(os.environ["DRFFR2_CACHE"])
REPO = Path(os.environ.get("DRFFR2_REPO", AQUI.parents[1] / "drffr2-green"))
SITIO = AQUI / "sitio"
CLASES = ["mavicAir2", "mavicAir2s", "mini4PRO"]


def cargar():
    seg = pd.read_csv(CACHE / "referencia_numpy" / "segmentos.csv")
    X = np.load(CACHE / "redes_numpy" / "mobilenet_v3_small_espectrograma_numpy_global.npy").astype(np.float64)
    assert len(seg) == len(X) == 1921 and X.shape[1] == 576
    y = seg["modelo"].map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    eleccion = pd.read_csv(REPO / "reports" / "espectrograma_numpy" / "elecciones.csv").set_index("unidad_fuera")
    assert (eleccion["elegido"] == "mobilenet_v3_small|espectrograma_numpy|global|logistica").all()
    articulo = pd.read_csv(REPO / "reports" / "espectrograma_numpy" / "por_segmento.csv")
    articulo = articulo[articulo["modelo_eval"] == "procedimiento"].set_index(["archivo", "segmento"])
    return seg, X, y, eleccion, articulo


def paso1_regresiones(seg, X, y, eleccion, articulo):
    unidad = seg["unidad"].to_numpy()
    pliegues, P = {}, np.zeros((len(seg), 3))
    for h in sorted(eleccion.index):
        C = float(eleccion.loc[h, "hiper"])
        modelo = md.entrenar_pliegue(X, y, unidad, h, C)
        p = md.parametros(modelo, C)
        fuera = unidad == h
        P[fuera] = md.proba_segmentos(X[fuera], p)
        assert np.allclose(P[fuera], modelo.predict_proba(X[fuera]), atol=1e-12)
        pliegues[h] = p
    ref = articulo.loc[list(zip(seg["archivo"], seg["segmento"]))][["p0", "p1", "p2"]].to_numpy()
    print(f"[1] regresiones: diferencia máxima con el artículo por segmento = {np.abs(P - ref).max():.2e}")
    return pliegues, P


def por_captura(seg, P):
    archivos = list(dict.fromkeys(seg["archivo"]))
    D = np.stack([md.decision_captura(P[(seg["archivo"] == a).to_numpy()]) for a in archivos])
    real = seg.groupby("archivo", sort=False)["modelo"].first().map(CLASES.index).to_numpy()
    return archivos, D, real


def paso2_onnx(seg, archivos, pliegues, D_cache):
    import onnxruntime as ort

    destino = SITIO / "modelo"
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / "mobilenet_v3_small_rasgos.onnx"
    md.exportar_onnx(ruta)
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    viridis = None
    unidad_de = seg.groupby("archivo", sort=False)["unidad"].first()
    D, indices, t0 = [], {}, time.perf_counter()
    for k, a in enumerate(archivos, start=1):
        img = np.load(CACHE / "referencia_numpy" / "imagenes" / a.replace(".mat", ".npz"))["espectrograma_numpy"]
        assert img.shape == (17, 224, 224, 3) and img.dtype == np.uint8
        indices[a], viridis = a_indices(img)
        rasgos = sesion.run(None, {"espectrograma": viridis[indices[a]]})[0].astype(np.float64)
        D.append(md.decision_captura(md.proba_segmentos(rasgos, pliegues[unidad_de[a]])))
        if k % 25 == 0:
            print(f"    onnx: {k}/113 capturas | {time.perf_counter() - t0:.0f} s", flush=True)
    D = np.stack(D)
    print(f"[2] onnx: diferencia máxima de probabilidad por captura frente a PyTorch = {np.abs(D - D_cache).max():.2e}; "
          f"misma decisión en {int((D.argmax(1) == D_cache.argmax(1)).sum())}/113")
    return D, indices, viridis, ruta


def a_indices(img):
    """Recupera el índice viridis (0 a 255) de cada píxel; la imagen RGB se reconstruye exacta con la tabla."""
    from matplotlib import colormaps

    viridis = (colormaps["viridis"](np.arange(256))[:, :3] * 255).round().astype(np.uint8)
    clave = viridis.astype(np.int32) @ np.array([65536, 256, 1])
    unicas, primero = np.unique(clave, return_index=True)   # en uint8 hay colores contiguos repetidos: da igual cuál
    codigo = img.astype(np.int32) @ np.array([65536, 256, 1])
    pos = np.clip(np.searchsorted(unicas, codigo), 0, len(unicas) - 1)
    idx = primero[pos].astype(np.uint8)
    assert np.array_equal(viridis[idx], img), "hay píxeles fuera de la tabla viridis"
    return idx, viridis


def paso3_datos(seg, archivos, real, D_cache, D_onnx, pliegues, indices, viridis, ruta_onnx):
    datos = SITIO / "datos" / "capturas"
    datos.mkdir(parents=True, exist_ok=True)
    unidad_de = seg.groupby("archivo", sort=False)["unidad"].first()
    capturas, total = [], 0
    for n, a in enumerate(archivos, start=1):
        bruto = gzip.compress(indices[a].tobytes(), compresslevel=9, mtime=0)
        (datos / f"c{n:03d}.bin.gz").write_bytes(bruto)
        total += len(bruto)
        c, u, d = re.search(r"_hover_(c\d)_(u\d)_(d\d)\.mat$", a).groups()
        capturas.append({"n": n, "archivo": a, "unidad": unidad_de[a], "real": int(real[n - 1]),
                         "canal": c, "receptor": u, "dia": d, "bytes": len(bruto),
                         "p_articulo": [round(float(v), 6) for v in D_cache[n - 1]]})
    huella = hashlib.sha256(ruta_onnx.read_bytes()).hexdigest()
    unidades = sorted(pliegues)
    meta = {"clases": CLASES, "unidades": unidades, "segmentos": 17, "lado": 224,
            "onnx": {"archivo": ruta_onnx.name, "sha256": huella, "bytes": ruta_onnx.stat().st_size,
                     "pesos": "torchvision mobilenet_v3_small IMAGENET1K_V1 (parte convolucional, congelada)"},
            "capturas": capturas,
            "articulo": {"aciertos": int((D_cache.argmax(1) == real).sum()), "total": len(real)},
            "verificacion_onnx_python": {"aciertos": int((D_onnx.argmax(1) == real).sum()),
                                         "max_dif_prob": float(np.abs(D_onnx - D_cache).max())}}
    (SITIO / "datos" / "capturas.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    redondear = lambda v: [float(f"{x:.9g}") for x in v]
    plieg = {h: {"C": p["C"], "media": redondear(p["media"]), "escala": redondear(p["escala"]),
                 "coef": [redondear(f) for f in p["coef"]], "intercepto": redondear(p["intercepto"])}
             for h, p in pliegues.items()}
    (SITIO / "modelo" / "pliegues.json").write_text(json.dumps(plieg), encoding="utf-8")
    (SITIO / "modelo" / "viridis.json").write_text(json.dumps(viridis.tolist()), encoding="utf-8")
    codigo = SITIO / "codigo"
    codigo.mkdir(exist_ok=True)
    for nombre in ("modelo_demo.py", "construir_demo.py"):
        shutil.copy(AQUI / nombre, codigo / nombre)
    print(f"[3] datos: 113 capturas, {total / 1e6:.1f} MB comprimidos (media {total / 113 / 1e6:.2f} MB por captura); "
          f"red ONNX {ruta_onnx.stat().st_size / 1e6:.2f} MB, SHA-256 {huella[:16]}…")


def main():
    seg, X, y, eleccion, articulo = cargar()
    pliegues, P = paso1_regresiones(seg, X, y, eleccion, articulo)
    archivos, D_cache, real = por_captura(seg, P)
    print(f"    por captura (rasgos de la caché): {int((D_cache.argmax(1) == real).sum())}/113 aciertos; "
          f"fallos: {[archivos[i] for i in np.flatnonzero(D_cache.argmax(1) != real)]}")
    D_onnx, indices, viridis, ruta = paso2_onnx(seg, archivos, pliegues, D_cache)
    print(f"    por captura (red ONNX desde las imágenes): {int((D_onnx.argmax(1) == real).sum())}/113 aciertos")
    paso3_datos(seg, archivos, real, D_cache, D_onnx, pliegues, indices, viridis, ruta)


if __name__ == "__main__":
    main()
