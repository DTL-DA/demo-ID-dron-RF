"""Modelo propuesto de la demo: MobileNetV3-Small congelada + regresión logística, un modelo por pliegue.

Es el mismo procedimiento del artículo (validación anidada que deja fuera un dron completo):
  1. Cada segmento de 80 ms se convierte en un espectrograma RGB de 224 x 224 (mapa viridis).
  2. La parte convolucional de MobileNetV3-Small, con los pesos de ImageNet (IMAGENET1K_V1) y congelada,
     da 576 x 7 x 7 activaciones; su promedio global son los 576 rasgos del segmento.
  3. En cada pliegue se estandarizan los rasgos y se ajusta una regresión logística con los segmentos de los
     19 drones de entrenamiento; el C de cada pliegue es el que eligió la validación interna del artículo.
  4. La decisión de una captura es el promedio de las log-probabilidades de sus 17 segmentos.

La red se exporta a ONNX para que corra en el navegador del celular (onnxruntime-web); la regresión de cada
pliegue viaja como números (media, escala, coeficientes e intercepto) en un JSON.
"""
import numpy as np


# ---------------------------------------------------------------- red: espectrograma -> 576 rasgos

def red_rasgos():
    """MobileNetV3-Small de ImageNet, congelada, que recibe imágenes RGB uint8 (n, 224, 224, 3) y da (n, 576).

    Incluye la normalización de ImageNet y el promedio global, igual que `rasgos_red` del artículo.
    """
    import torch
    import torchvision

    class RasgosMobileNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            red = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            self.features = red.features.eval()          # solo la parte convolucional: 576 x 7 x 7
            for p in self.features.parameters():
                p.requires_grad_(False)
            self.register_buffer("media", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("desv", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, imagen_uint8):
            x = imagen_uint8.permute(0, 3, 1, 2).float() / 255.0     # (n, 3, 224, 224) en [0, 1]
            x = (x - self.media) / self.desv                         # normalización de ImageNet
            return self.features(x).mean(dim=(2, 3))                 # promedio global -> (n, 576)

    return RasgosMobileNet().eval()


def exportar_onnx(ruta):
    """Exporta la red de rasgos a ONNX con el lote variable (1 a 17 segmentos por llamada)."""
    import torch

    red = red_rasgos()
    ejemplo = torch.zeros(1, 224, 224, 3, dtype=torch.uint8)
    torch.onnx.export(red, (ejemplo,), str(ruta), input_names=["espectrograma"], output_names=["rasgos"],
                      dynamic_axes={"espectrograma": {0: "segmentos"}, "rasgos": {0: "segmentos"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)


# ---------------------------------------------------------------- regresión logística de cada pliegue

def entrenar_pliegue(X, y, unidad, dron_fuera, C, semilla=42):
    """Estandarización + regresión logística con los segmentos de los 19 drones distintos de `dron_fuera`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    entrena = unidad != dron_fuera
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=semilla))
    return modelo.fit(X[entrena], y[entrena])


def parametros(modelo, C) -> dict:
    """Los números que usa el celular: z = (x - media) / escala; logits = coef · z + intercepto."""
    escalado, logistica = modelo.named_steps["standardscaler"], modelo.named_steps["logisticregression"]
    return {"C": C, "media": escalado.mean_.tolist(), "escala": escalado.scale_.tolist(),
            "coef": logistica.coef_.tolist(), "intercepto": logistica.intercept_.tolist()}


def proba_segmentos(rasgos, p) -> np.ndarray:
    """Probabilidades de los 3 tipos por segmento, con los parámetros de un pliegue (la cuenta del celular)."""
    z = (rasgos - np.asarray(p["media"])) / np.asarray(p["escala"])
    logits = z @ np.asarray(p["coef"]).T + np.asarray(p["intercepto"])
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def decision_captura(proba) -> np.ndarray:
    """Promedio de las log-probabilidades de los segmentos de una captura, renormalizado (regla del artículo)."""
    P = np.exp(np.log(np.clip(proba, 1e-9, 1)).mean(axis=0))
    return P / P.sum()
