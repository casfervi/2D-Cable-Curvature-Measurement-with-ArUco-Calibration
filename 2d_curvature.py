# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:11:08 2026

@author: vinicius.ferreira

curvatura_2d_aruco.py

Mede a curvatura e o raio de curvatura de um cabo (objeto escuro sobre
fundo claro), em video ou webcam, com calibracao por marcadores ArUco.

Pipeline:
    0. Calibracao (opcional): 4 ArUco -> homografia -> imagem retificada
       com escala exata em mm (corrige perspectiva e da a escala)
    1. Segmentacao (Otsu ou threshold manual) + maior componente conexa
    2. Skeleton (afinamento para ~1 px)
    3. Ordenacao dos pixels do skeleton (caminho geodesico mais longo)
    4. Spline parametrica suavizada  x(t), y(t)
    5. Curvatura com sinal:  k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    6. Raio de curvatura:    R = 1 / |k|

Calibracao com ArUco:
    Coloque 4 marcadores no MESMO plano do cabo, nos cantos de um
    retangulo, todos com a mesma orientacao de impressao:

        ID 0 ---------- ID 1        (ids padrao; mude com --aruco-ids,
         |               |           ordem: sup-esq, sup-dir, inf-dir, inf-esq)
        ID 3 ---------- ID 2

    --aruco-size     lado do marcador (quadrado preto externo), em mm
    --aruco-spacing  distancia CENTRO A CENTRO entre marcadores vizinhos, em mm
                     (um valor = retangulo quadrado; dois valores = horizontal vertical)

    Os 16 cantos dos marcadores ajustam a homografia (minimos quadrados).
    O erro RMS de reprojecao e mostrado no painel como medida de qualidade.

Controles:  (tudo dentro do dashboard)
    Botoes -1s / < / PLAY-PAUSE / > / +1s = navegacao no video
    Barra de progresso = clique ou arraste para ir a qualquer instante
    Sliders            = threshold (0 = Otsu) e suavizacao da spline (arraste)
    Clique esquerdo    = mostra R e o circulo osculador naquele ponto do RESULTADO
    Clique direito     = remove a selecao
    C                  = refaz a calibracao ArUco
    ESPACO             = pausa / continua
    S                  = salva snapshot (PNG + CSV + grafico)
    Q ou ESC           = sair

Pose da camera:
    O painel POSE mostra o conjunto de ArUco com os eixos XYZ (X vermelho,
    Y verde, Z azul; origem no centro dos marcadores, Z saindo da folha) e a
    camera com seu frustum e eixos xyz. Sem calibracao de lente a distancia
    focal e assumida (0.8 x largura da imagem): use --focal-px para informar
    o valor real (distancia e altura escalam com ela; a inclinacao muda pouco).

Exemplos:
    python curvatura_2d_aruco.py --video cabo.mp4 --aruco-size 40 --aruco-spacing 300 200
    python curvatura_2d_aruco.py --video cabo.mp4 --aruco-size 40 --aruco-spacing 300 200 --focal-px 1500
    python curvatura_2d_aruco.py --camera 0 --aruco-size 40 --aruco-spacing 250
    python curvatura_2d_aruco.py --video cabo.mp4 --load-calibration
    python curvatura_2d_aruco.py --video cabo.mp4 --mm-per-pixel 0.25
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize


# ============================================================
# CONSTANTES
# ============================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

# Dashboard theme (OpenCV uses BGR colors)
THEME = {
    "background": (18, 21, 27),
    "panel": (28, 33, 42),
    "panel_alt": (34, 40, 50),
    "header": (37, 44, 56),
    "border": (65, 75, 92),
    "text": (235, 239, 244),
    "muted": (155, 166, 181),
    "accent": (235, 154, 47),
    "accent_soft": (92, 65, 35),
    "success": (112, 201, 126),
    "warning": (60, 190, 245),
    "danger": (90, 95, 235),
    "curve": (238, 164, 63),
    "grid": (60, 68, 82),
}

WIN_DASHBOARD = "Analise de Curvatura"
WIN_RESULT = "1 - Resultado"
WIN_MASK = "2 - Mascara"
WIN_RECTIFICATION = "3 - Original e Retificada"
WIN_INFO = "4 - Informacoes"
WIN_CURVATURE = "4 - Curvatura"
WIN_RADIUS = "5 - Raio"
WIN_CONTROLS = "6 - Controles"

TRACKBAR_THRESHOLD = "Threshold (0=auto)"
TRACKBAR_SMOOTHING = "Suavizacao x0.1px"

# Frames CONSECUTIVOS com os 4 marcadores visiveis para calibrar
# (a mediana dos cantos reduz o ruido de subpixel)
CALIBRATION_FRAMES = 15

# Quanto o marcador e "alargado" ao apagar do frame retificado
# (cobre a borda serrilhada e a zona branca em volta)
MARKER_MASK_SCALE = 1.15

TITLE_HEIGHT = 30

# Distancia focal assumida (fracao da largura da imagem) quando --focal-px
# nao e informado. 0.8 equivale a ~65 graus de campo horizontal (webcam/celular).
DEFAULT_FOCAL_FACTOR = 0.8

# A pose usa os 16 cantos dos marcadores; se o erro de reprojecao passar disso
# (marcador impresso ou colado girado), usa so os 4 centros.
POSE_MAX_REPROJECTION_PX = 3.0

# Lado maximo (px) da imagem retificada quando a resolucao e automatica
MAX_RECTIFIED_SIDE_PX = 2400

# Sliders do dashboard (substituem a janela externa de trackbars)
SLIDERS = {
    "threshold": {"min": 0, "max": 255, "step": 1},
    "smoothing": {"min": 0.1, "max": 10.0, "step": 0.1},
}


# ============================================================
# CONFIGURACAO E ESTRUTURAS DE DADOS
# ============================================================

@dataclass
class Settings:
    # Escala. Sem calibracao: 1 pixel = 1 unidade ("px")
    mm_per_pixel: float = 1.0
    calibrated: bool = False

    # Area minima do objeto segmentado, em pixels
    min_object_area: int = 1000

    # Quantidade de pontos da curva final (0 = automatico: ~1 ponto a cada 3 px de arco)
    curve_point_count: int = 0

    # Suavizacao da spline, em PIXELS: erro RMS tolerado entre o
    # skeleton e a curva ajustada.
    #   Aumente se a curvatura estiver ruidosa (picos falsos).
    #   Diminua se curvas fechadas estiverem sendo "achatadas".
    # O erro de curvatura cresce rapido com este valor (2 px ja da ~20% num
    # cabo de raio 40 mm a 2 px/mm), entao comece baixo e suba so se precisar.
    smoothing_px: float = 1.0

    # Fracao das pontas da curva descartada nas estatisticas
    # (o skeleton e a spline sao pouco confiaveis nas extremidades)
    trim_fraction: float = 0.03

    # Espessura da centerline desenhada
    centerline_thickness: int = 5

    @property
    def unit(self) -> str:
        return "mm" if self.calibrated else "px"


@dataclass
class Centerline:
    x: np.ndarray           # coordenadas em pixels
    y: np.ndarray
    normal_x: np.ndarray    # normal unitaria (aponta para o centro de curvatura se k>0)
    normal_y: np.ndarray
    arc_length: np.ndarray  # comprimento de arco na unidade da escala
    curvature: np.ndarray   # 1/unidade, COM sinal
    radius: np.ndarray      # unidade (inf em trechos retos)
    valid: slice            # trecho sem as pontas


@dataclass
class Analysis:
    """Resultado da analise de um frame."""
    mask: np.ndarray
    contours: list
    area: int
    threshold: float
    skeleton: np.ndarray
    centerline: Optional[Centerline]


@dataclass
class Statistics:
    length: float
    max_curvature: float
    min_radius: float
    min_radius_index: int
    max_line_deviation: float


@dataclass
class Views:
    """Imagens prontas para exibir."""
    result: np.ndarray
    info: np.ndarray
    rectification: np.ndarray
    mask: np.ndarray
    curvature_graph: np.ndarray
    radius_graph: np.ndarray
    centerline: Optional[Centerline] = None  # usado pelo snapshot
    pose: Optional[np.ndarray] = None        # vista 3D da pose da camera


@dataclass
class FrameView:
    """Onde o frame aparece dentro da imagem exibida (para converter cliques)."""
    x: float
    y: float
    scale: float
    width: int
    height: int


@dataclass
class Interaction:
    """Estado compartilhado com o callback do mouse."""
    click: Optional[Tuple[float, float]] = None       # em coordenadas do frame
    window_fit: Tuple[float, int, int] = (1.0, 0, 0)  # (escala, x0, y0) de show_fit
    frame_view: Optional[FrameView] = None
    control_regions: dict = None
    progress_region: Optional[Tuple[int, int, int, int]] = None
    action: Optional[str] = None
    seek_fraction: Optional[float] = None
    slider_regions: dict = None      # nome -> (x, y, w, h) da trilha, em coords do dashboard
    slider_values: dict = None       # nome -> valor atual (compartilhado com o app)
    dragging: Optional[str] = None   # slider (ou "progress") sendo arrastado

    def __post_init__(self):
        if self.slider_values is None:
            self.slider_values = {}


# ============================================================
# 1. SEGMENTACAO
# ============================================================

def segment_object(frame, threshold_value, min_area):
    """
    Segmenta um objeto escuro sobre um fundo claro.

    threshold_value = 0 usa Otsu (automatico).

    Retorna:
        clean_mask: mascara binaria com somente o maior objeto
        contours:   contornos externos do objeto (para desenhar)
        area:       area do objeto em pixels
        used_threshold: limiar efetivamente usado
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if threshold_value <= 0:
        used_threshold, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        used_threshold, mask = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Kernel 3x3: um kernel 5x5 na abertura apaga cabos finos
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    clean_mask = np.zeros_like(mask)

    # Maior componente conexa (preserva buracos, ao contrario de
    # preencher o contorno externo)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if count <= 1:
        return clean_mask, [], 0, used_threshold

    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[best, cv2.CC_STAT_AREA])

    if area < min_area:
        return clean_mask, [], area, used_threshold

    clean_mask[labels == best] = 255

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return clean_mask, contours, area, used_threshold


def create_skeleton(mask):
    """
    Reduz a mascara para uma linha de ~1 pixel de espessura.
    Processa apenas a caixa que envolve o objeto (bem mais rapido).

    Retorna um array booleano com o mesmo tamanho da mascara.
    """

    skeleton = np.zeros(mask.shape, dtype=bool)

    if not np.any(mask):
        return skeleton

    x, y, w, h = cv2.boundingRect(mask)
    pad = 2
    y0 = max(y - pad, 0)
    y1 = min(y + h + pad, mask.shape[0])
    x0 = max(x - pad, 0)
    x1 = min(x + w + pad, mask.shape[1])

    skeleton[y0:y1, x0:x1] = skeletonize(mask[y0:y1, x0:x1] > 0)

    return skeleton


# ============================================================
# 2. CENTERLINE: ORDENACAO, SPLINE E CURVATURA
# ============================================================

_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def order_skeleton_pixels(skeleton):
    """
    Devolve os pixels do skeleton ORDENADOS ao longo do cabo,
    como array Nx2 [x, y], ou None se houver pontos insuficientes.

    Constroi um grafo de vizinhanca-8 e pega o caminho mais longo
    entre duas extremidades (dois Dijkstra). Isso:
      - funciona para qualquer orientacao (curvas em U, trechos verticais);
      - ignora ramificacoes curtas ("espinhos") do skeleton.
    """

    ys, xs = np.nonzero(skeleton)
    count = len(xs)

    if count < 20:
        return None

    # Trabalha numa caixa pequena para nao alocar a imagem inteira
    y_min, x_min = ys.min(), xs.min()
    box_h = ys.max() - y_min + 1
    box_w = xs.max() - x_min + 1

    local_y = ys - y_min
    local_x = xs - x_min

    index_map = np.full((box_h, box_w), -1, dtype=np.int64)
    index_map[local_y, local_x] = np.arange(count)

    rows, cols, weights = [], [], []

    for dy, dx in _NEIGHBOR_OFFSETS:
        ny = local_y + dy
        nx = local_x + dx

        inside = (ny >= 0) & (ny < box_h) & (nx >= 0) & (nx < box_w)

        neighbor = np.full(count, -1, dtype=np.int64)
        neighbor[inside] = index_map[ny[inside], nx[inside]]

        connected = neighbor >= 0

        rows.append(np.nonzero(connected)[0])
        cols.append(neighbor[connected])
        weights.append(np.full(connected.sum(), np.hypot(dy, dx)))

    graph = coo_matrix(
        (
            np.concatenate(weights),
            (np.concatenate(rows), np.concatenate(cols)),
        ),shape=(count, count)).tocsr()

    # 1o Dijkstra: ponto mais distante de um pixel qualquer = uma ponta
    distance = dijkstra(graph, directed=False, indices=0)
    distance[~np.isfinite(distance)] = -1
    start = int(np.argmax(distance))

    # 2o Dijkstra: a partir da ponta, acha a ponta oposta
    distance, predecessors = dijkstra(
        graph, directed=False, indices=start, return_predecessors=True)
    distance[~np.isfinite(distance)] = -1
    end = int(np.argmax(distance))

    path = [end]
    while path[-1] != start:
        previous = predecessors[path[-1]]
        if previous < 0:
            break
        path.append(int(previous))

    if len(path) < 10:
        return None

    path = path[::-1]

    return np.column_stack((xs[path], ys[path])).astype(np.float64)


def compute_centerline(points, settings):
    """
    Ajusta uma spline parametrica suavizada aos pontos ordenados e
    calcula curvatura (com sinal) e raio em cada ponto.

    Nao assume y = f(x), entao funciona para qualquer formato de cabo.
    """

    # Remove pontos repetidos
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], steps > 1e-9))
    points = points[keep]

    if len(points) < 10:
        return None

    chord = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))

    if chord[-1] < 1e-9:
        return None

    u = chord / chord[-1]

    # splprep minimiza a rugosidade sujeito a sum(residuos^2) <= s.
    # Com s = N * sigma^2, sigma e o erro RMS tolerado em pixels.
    smoothing = len(points) * settings.smoothing_px ** 2

    try:
        tck, _ = splprep([points[:, 0], points[:, 1]], u=u, s=smoothing, k=3)

        point_count = settings.curve_point_count
        if point_count <= 0:
            point_count = int(np.clip(chord[-1] / 3.0, 100, 1000))

        u_new = np.linspace(0.0, 1.0, point_count)

        x, y = splev(u_new, tck)
        dx, dy = splev(u_new, tck, der=1)
        ddx, ddy = splev(u_new, tck, der=2)

    except Exception as error:
        print(f"[ERRO] Falha no ajuste da spline: {error}", flush=True)
        return None

    speed = np.hypot(dx, dy)

    if np.any(speed < 1e-9):
        return None

    # Curvatura com sinal em 1/pixel
    curvature_px = (dx * ddy - dy * ddx) / speed ** 3

    # 1/pixel -> 1/mm:  k_mm = k_px / (mm/pixel)
    curvature = curvature_px / settings.mm_per_pixel

    radius = np.full_like(curvature, np.inf)
    nonzero = np.abs(curvature) > 1e-9
    radius[nonzero] = 1.0 / np.abs(curvature[nonzero])

    segment_lengths = np.hypot(np.diff(x), np.diff(y))
    arc_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    arc_length *= settings.mm_per_pixel

    trim = int(round(settings.trim_fraction * len(x)))
    valid = slice(trim, len(x) - trim)

    return Centerline(
        x=np.asarray(x),
        y=np.asarray(y),
        normal_x=-dy / speed,
        normal_y=dx / speed,
        arc_length=arc_length,
        curvature=curvature,
        radius=radius,
        valid=valid)


def calculate_line_deviation(centerline, settings):
    """
    Distancia de cada ponto da centerline ate a reta formada pelo
    primeiro e pelo ultimo ponto (na unidade da escala).
    """

    curve = np.column_stack((centerline.x, centerline.y))

    reference = curve[-1] - curve[0]
    length = np.linalg.norm(reference)

    if length < 1e-12:
        return np.zeros(len(curve))

    unit_vector = reference / length
    relative = curve - curve[0]

    along = relative @ unit_vector
    perpendicular = relative - np.outer(along, unit_vector)

    return np.linalg.norm(perpendicular, axis=1) * settings.mm_per_pixel


def compute_statistics(centerline, settings) -> Statistics:
    v = centerline.valid
    index = v.start + int(np.argmax(np.abs(centerline.curvature[v])))
    deviation = calculate_line_deviation(centerline, settings)

    return Statistics(
        length=float(centerline.arc_length[-1]),
        max_curvature=float(abs(centerline.curvature[index])),
        min_radius=float(centerline.radius[index]),
        min_radius_index=index,
        max_line_deviation=float(np.max(deviation)))


def nearest_valid_index(centerline, click):
    v = centerline.valid
    distance = np.hypot(centerline.x[v] - click[0], centerline.y[v] - click[1])
    return v.start + int(np.argmin(distance))


def analyze_frame(frame, settings, threshold_value) -> Analysis:
    """Segmentacao -> skeleton -> ordenacao -> spline -> curvatura."""

    mask, contours, area, used_threshold = segment_object(frame, threshold_value, settings.min_object_area)
    skeleton = create_skeleton(mask)
    points = order_skeleton_pixels(skeleton)
    centerline = compute_centerline(points, settings) if points is not None else None

    return Analysis(mask, contours, area, used_threshold, skeleton, centerline)


def empty_analysis(shape) -> Analysis:
    height, width = shape[:2]
    return Analysis(
        mask=np.zeros((height, width), np.uint8),
        contours=[],
        area=0,
        threshold=0.0,
        skeleton=np.zeros((height, width), bool),
        centerline=None)


# ============================================================
# 3. CALIBRACAO COM ARUCO
# ============================================================

@dataclass(frozen=True)
class ArucoLayout:
    """
    Disposicao fisica dos 4 marcadores (medidas em mm).

    Os centros formam um retangulo: sup-esq (0,0), sup-dir (sx,0),
    inf-dir (sx,sy), inf-esq (0,sy), na ordem de `ids`.
    """

    marker_size_mm: float
    spacing_x_mm: float
    spacing_y_mm: float
    ids: Tuple[int, ...] = (0, 1, 2, 3)
    dictionary: str = "DICT_5X5_250"
    pixels_per_mm: float = 0.0   # 0 = automatico (resolucao nativa da camera)
    margin_mm: float = 0.0
    

    def marker_corners_mm(self) -> np.ndarray:
        """
        Cantos dos 4 marcadores no plano, shape (4 marcadores, 4 cantos, 2).
        Origem no canto externo do marcador superior-esquerdo.
        Ordem dos cantos = a do OpenCV: TL, TR, BR, BL do proprio marcador.
        """

        half = self.marker_size_mm / 2.0

        centers = np.array([
            [0.0, 0.0],
            [self.spacing_x_mm, 0.0],
            [self.spacing_x_mm, self.spacing_y_mm],
            [0.0, self.spacing_y_mm],
        ]) + half

        offsets = np.array([
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
        ])

        return centers[:, None, :] + offsets[None, :, :]

    # @property
    # def output_size(self) -> Tuple[int, int]:
    #     """Tamanho (largura, altura) da imagem retificada em pixels."""
    #     width = (self.spacing_x_mm + self.marker_size_mm) * self.pixels_per_mm
    #     height = (self.spacing_y_mm + self.marker_size_mm) * self.pixels_per_mm
    #     return int(round(width)), int(round(height))
    
    @property
    def output_size(self) -> Tuple[int, int]:
        width_mm = (self.spacing_x_mm + self.marker_size_mm + 2.0 * self.margin_mm)    
        height_mm = (self.spacing_y_mm + self.marker_size_mm + 2.0 * self.margin_mm)
        width_px = width_mm * self.pixels_per_mm
        height_px = height_mm * self.pixels_per_mm
    
        return (int(round(width_px)),int(round(height_px)))


@dataclass
class PlanarCalibration:
    homography: np.ndarray       # imagem original -> imagem retificada
    output_size: Tuple[int, int]  # (largura, altura) da imagem retificada
    mm_per_pixel: float          # escala da imagem retificada
    marker_polygons: np.ndarray  # (4, 4, 2): marcadores na imagem retificada, em px
    rms_error_mm: float          # erro RMS de reprojecao (mm)
    method: str = "4 centros"     # "16 cantos" (minimos quadrados) ou "4 centros"


def create_aruco_detector(dictionary_name):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError(
            "A calibracao ArUco requer OpenCV >= 4.7 "
            "(pip install -U opencv-contrib-python)."
        )

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)

    if dictionary_id is None:
        raise RuntimeError(f"Dicionario ArUco desconhecido: {dictionary_name}")

    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_markers(detector, frame):
    """Retorna {id: cantos (4, 2)} de todos os marcadores detectados."""

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        return {}

    return {
        int(marker_id): marker_corners.reshape(4, 2)
        for marker_corners, marker_id in zip(corners, ids.ravel())}


def native_pixels_per_mm(image_corners, layout) -> float:
    """
    Resolucao (px/mm) do plano na PROPRIA imagem, medida pelo lado dos marcadores.

    Retificar em resolucao menor que a da camera joga detalhe fora, e a
    suavizacao (em pixels) passa a valer mais milimetros: o ajuste do cabo
    piora sem que a escala mude. Por isso o padrao e a resolucao nativa.
    """

    sides = np.linalg.norm(image_corners - np.roll(image_corners, -1, axis=1), axis=2)
    ppm = float(np.median(sides)) / layout.marker_size_mm

    largest_mm = (max(layout.spacing_x_mm, layout.spacing_y_mm)
                  + layout.marker_size_mm + 2.0 * layout.margin_mm)
    ppm = min(ppm, MAX_RECTIFIED_SIDE_PX / largest_mm)

    return max(ppm, 0.5)


def compute_calibration(samples, layout) -> Optional[PlanarCalibration]:
    """
    Homografia imagem -> plano retificado (em mm).

    1. Homografia exata pelos 4 centros (robusta a marcadores girados).
    2. Se os 16 cantos concordam com ela, refaz por minimos quadrados com os
       16 cantos (mais preciso, principalmente em vista obliqua). Nesse caso o
       erro RMS e uma medida real de qualidade (com 4 pontos ele e sempre ~0).

    A ordem e definida por layout.ids: superior esquerdo, superior direito,
    inferior direito e inferior esquerdo.
    """

    image_corners = np.median(np.stack(samples), axis=0)              # (4, 4, 2)
    image_centers = np.mean(image_corners, axis=1).astype(np.float32)

    ppm = layout.pixels_per_mm
    if ppm <= 0:
        ppm = native_pixels_per_mm(image_corners, layout)
    layout = replace(layout, pixels_per_mm=ppm)

    half_marker_px = layout.marker_size_mm * ppm / 2.0
    margin_px = (layout.margin_mm * ppm)
    spacing_x_px = layout.spacing_x_mm * ppm
    spacing_y_px = layout.spacing_y_mm * ppm

    # destination_centers = np.array([
    #     [half_marker_px, half_marker_px],
    #     [half_marker_px + spacing_x_px, half_marker_px],
    #     [half_marker_px + spacing_x_px, half_marker_px + spacing_y_px],
    #     [half_marker_px, half_marker_px + spacing_y_px]], dtype=np.float32)

    destination_centers = np.array([
        [margin_px + half_marker_px, margin_px + half_marker_px],
        [margin_px + half_marker_px + spacing_x_px, margin_px + half_marker_px],
        [margin_px + half_marker_px + spacing_x_px, margin_px + half_marker_px + spacing_y_px],
        [margin_px + half_marker_px, margin_px + half_marker_px + spacing_y_px]], dtype=np.float32)

    homography = cv2.getPerspectiveTransform(image_centers, destination_centers)
    if homography is None:
        return None

    image_points = image_corners.reshape(-1, 1, 2).astype(np.float32)
    # expected_corners = (layout.marker_corners_mm() * ppm).reshape(-1, 1, 2).astype(np.float32)
    expected_corners = ((layout.marker_corners_mm() + layout.margin_mm) * ppm).reshape(-1, 1, 2).astype(np.float32)

    method = "4 centros"
    projected = cv2.perspectiveTransform(image_points, homography)
    deviation_mm = np.linalg.norm((projected - expected_corners).reshape(-1, 2), axis=1) / ppm

    # Cantos coerentes com a homografia dos centros (marcadores no mesmo
    # sentido)? Entao o ajuste por minimos quadrados vale.
    if deviation_mm.max() < 0.25 * layout.marker_size_mm:
        refined, _ = cv2.findHomography(image_points, expected_corners, 0)
        if refined is not None:
            homography = refined
            method = "16 cantos"

    if method == "16 cantos":
        projected = cv2.perspectiveTransform(image_points, homography)
        errors_mm = np.linalg.norm((projected - expected_corners).reshape(-1, 2), axis=1) / ppm
    else:
        projected = cv2.perspectiveTransform(image_centers.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors_mm = np.linalg.norm(projected - destination_centers, axis=1) / ppm

    rms_error_mm = float(np.sqrt(np.mean(errors_mm ** 2)))

    offsets = np.array([
        [-half_marker_px, -half_marker_px],
        [ half_marker_px, -half_marker_px],
        [ half_marker_px,  half_marker_px],
        [-half_marker_px,  half_marker_px],
    ], dtype=np.float32)
    marker_polygons = destination_centers[:, None, :] + offsets[None, :, :]

    return PlanarCalibration(
        homography=homography,
        output_size=layout.output_size,
        mm_per_pixel=1.0 / ppm,
        marker_polygons=marker_polygons,
        rms_error_mm=rms_error_mm,
        method=method,
    )


class ArucoCalibrator:
    """Acumula deteccoes de frames consecutivos e calcula a calibracao."""

    def __init__(self, layout: ArucoLayout, frames_required=CALIBRATION_FRAMES):
        self.layout = layout
        self.frames_required = frames_required
        self.detector = create_aruco_detector(layout.dictionary)
        self.samples: List[np.ndarray] = []

    @property
    def progress(self) -> int:
        return len(self.samples)

    def reset(self):
        self.samples = []

    def update(self, frame):
        """Retorna (marcadores detectados, calibracao ou None)."""

        found = detect_markers(self.detector, frame)

        if all(marker_id in found for marker_id in self.layout.ids):
            self.samples.append(
                np.stack([found[marker_id] for marker_id in self.layout.ids])
            )
        else:
            self.samples = []  # exige frames CONSECUTIVOS

        if len(self.samples) < self.frames_required:
            return found, None

        calibration = compute_calibration(self.samples, self.layout)
        self.samples = []

        return found, calibration


def rectify_frame(frame, calibration):
    """Vista de topo do plano dos marcadores, na escala da calibracao."""

    return cv2.warpPerspective(
        frame,
        calibration.homography,
        calibration.output_size,
        flags=cv2.INTER_LINEAR,
        borderValue=WHITE,  # fora da imagem = branco (nao vira "objeto")
    )


def whiten_markers(rectified_frame, calibration):
    """
    Pinta os marcadores de branco. Sem isso, os padroes preto e branco dos
    ArUco seriam segmentados como objeto (e podem ate ganhar do cabo).
    """

    output = rectified_frame.copy()

    for polygon in calibration.marker_polygons:
        center = polygon.mean(axis=0)
        enlarged = center + (polygon - center) * MARKER_MASK_SCALE
        cv2.fillConvexPoly(output, np.round(enlarged).astype(np.int32), WHITE)

    return output


def save_calibration(path, calibration, layout=None):
    data = {
        "homography": calibration.homography.tolist(),
        "output_size": list(calibration.output_size),
        "mm_per_pixel": calibration.mm_per_pixel,
        "marker_polygons": calibration.marker_polygons.tolist(),
        "rms_error_mm": calibration.rms_error_mm,
        "method": calibration.method,
    }

    if layout is not None:
        data["layout"] = asdict(layout)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_calibration(path) -> PlanarCalibration:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return PlanarCalibration(
        homography=np.asarray(data["homography"], dtype=np.float64),
        output_size=tuple(int(v) for v in data["output_size"]),
        mm_per_pixel=float(data["mm_per_pixel"]),
        marker_polygons=np.asarray(data.get("marker_polygons", []), dtype=np.float64).reshape(-1, 4, 2),
        rms_error_mm=float(data.get("rms_error_mm", float("nan"))),
        method=str(data.get("method", "4 centros")),
    )


# ============================================================
# 3b. POSE DA CAMERA (ArUco -> posicao e orientacao)
# ============================================================
# Referencial do ArUco (mm): origem no centro do retangulo dos marcadores,
# X para a direita, Y para o topo da folha, Z saindo da folha (para a camera).

AXIS_COLORS = {
    "X": (60, 60, 235),    # vermelho (BGR)
    "Y": (90, 200, 90),    # verde
    "Z": (235, 140, 40),   # azul
}


@dataclass
class CameraPose:
    rvec: np.ndarray            # (3, 1) referencial ArUco -> camera
    tvec: np.ndarray            # (3, 1) mm
    camera_matrix: np.ndarray   # (3, 3)
    rotation: np.ndarray        # (3, 3) ArUco -> camera
    position: np.ndarray        # (3,) camera no referencial do ArUco, mm
    reprojection_px: float
    focal_px: float
    focal_assumed: bool
    image_size: Tuple[int, int]  # (largura, altura)


def pose_object_points(layout):
    """Cantos (4 marcadores x 4 cantos, ordem TL TR BR BL) e centros no referencial do ArUco."""

    half = layout.marker_size_mm / 2.0
    cx = layout.spacing_x_mm / 2.0
    cy = layout.spacing_y_mm / 2.0

    centers = np.array([[-cx, cy], [cx, cy], [cx, -cy], [-cx, -cy]])
    offsets = np.array([[-half, half], [half, half], [half, -half], [-half, -half]])
    corners = centers[:, None, :] + offsets[None, :, :]

    def to_3d(points):
        return np.concatenate((points, np.zeros(points.shape[:-1] + (1,))), axis=-1)

    return to_3d(corners), to_3d(centers)


def estimate_camera_pose(found, layout, frame_shape, focal_px=None) -> Optional[CameraPose]:
    """Pose da camera a partir dos 4 marcadores (None se algum nao foi detectado)."""

    if not all(marker_id in found for marker_id in layout.ids):
        return None

    height, width = frame_shape[:2]
    focal = float(focal_px) if focal_px else DEFAULT_FOCAL_FACTOR * width
    camera_matrix = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0]])
    no_distortion = np.zeros(5)

    object_corners, object_centers = pose_object_points(layout)
    image_corners = np.stack([found[marker_id] for marker_id in layout.ids]).astype(np.float64)

    candidates = (
        (object_corners.reshape(-1, 3), image_corners.reshape(-1, 2)),
        (object_centers, image_corners.mean(axis=1)),
    )

    result = None
    for object_points, image_points in candidates:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, no_distortion,
            flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue

        rvec, tvec = cv2.solvePnPRefineLM(
            object_points, image_points, camera_matrix, no_distortion, rvec, tvec)

        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, no_distortion)
        error = float(np.sqrt(np.mean(
            np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))

        result = (rvec, tvec, error)
        if error <= POSE_MAX_REPROJECTION_PX:
            break

    if result is None:
        return None

    rvec, tvec, error = result
    rotation, _ = cv2.Rodrigues(rvec)
    position = (-rotation.T @ tvec).ravel()

    return CameraPose(
        rvec=rvec, tvec=tvec, camera_matrix=camera_matrix, rotation=rotation,
        position=position, reprojection_px=error, focal_px=focal,
        focal_assumed=not focal_px, image_size=(width, height))


def draw_pose_axes(image, pose, layout):
    """Desenha os eixos XYZ do ArUco sobre a imagem original da camera."""

    length = 0.5 * min(layout.spacing_x_mm, layout.spacing_y_mm)
    cv2.drawFrameAxes(
        image, pose.camera_matrix, np.zeros(5), pose.rvec, pose.tvec, length, 4)


def _view_basis(azimuth_deg=28.0, elevation_deg=30.0):
    """Base (direita, cima) da vista ortografica: observador pelo lado de baixo da folha, olhando de cima."""

    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)

    to_viewer = np.array([
        np.cos(elevation) * np.sin(azimuth),
        -np.cos(elevation) * np.cos(azimuth),
        np.sin(elevation)])
    forward = -to_viewer

    up = np.array([0.0, 0.0, 1.0])
    up = up - forward * np.dot(up, forward)
    up /= np.linalg.norm(up)

    return np.cross(forward, up), up


def _put_centered(image, lines, color=(200, 200, 200)):
    height, width = image.shape[:2]
    y = height // 2 - 12 * (len(lines) - 1)

    for text in lines:
        size = cv2.getTextSize(text, FONT, 0.55, 1)[0]
        cv2.putText(image, text, ((width - size[0]) // 2, y),
                    FONT, 0.55, color, 1, cv2.LINE_AA)
        y += 24


def _dashed_line(image, p0, p1, color, dash=6):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(p1 - p0))

    if length < 1.0:
        return

    for start in np.arange(0.0, length, 2 * dash):
        a = p0 + (p1 - p0) * (start / length)
        b = p0 + (p1 - p0) * (min(start + dash, length) / length)
        cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                 color, 1, cv2.LINE_AA)


def build_pose_view(pose, layout, live=True, width=480, height=330):
    """
    Vista 3D esquematica: folha com os 4 ArUco, eixos XYZ e a camera
    (frustum + eixos xyz) na posicao estimada.
    """

    image = np.full((height, width, 3), 38, dtype=np.uint8)

    if layout is None:
        _put_centered(image, ["Pose indisponivel", "use --aruco-size e --aruco-spacing"])
        return image

    if pose is None:
        _put_centered(
            image,
            ["Marcadores ArUco nao detectados", f"ids esperados: {list(layout.ids)}"],
            (0, 200, 255))
        return image

    right, up = _view_basis()
    half = layout.marker_size_mm / 2.0
    cx = layout.spacing_x_mm / 2.0
    cy = layout.spacing_y_mm / 2.0

    cam_x, cam_y, cam_z = pose.rotation.T[:, 0], pose.rotation.T[:, 1], pose.rotation.T[:, 2]
    position = pose.position
    distance = float(np.linalg.norm(position))

    # Frustum com o campo de visao real (assumido ou informado)
    depth = max(0.25 * distance, 30.0)
    half_w = depth * (pose.image_size[0] / 2.0) / pose.focal_px
    half_h = depth * (pose.image_size[1] / 2.0) / pose.focal_px
    lens = position + cam_z * depth
    frustum = [lens + sx * half_w * cam_x + sy * half_h * cam_y
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    axis_length = 0.6 * max(cx, cy)
    camera_axis_length = 0.6 * depth
    origin = np.zeros(3)
    foot = np.array([position[0], position[1], 0.0])
    board = [np.array([sx * (cx + half), sy * (cy + half), 0.0])
             for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]

    unit = np.eye(3)
    all_points = np.array(
        board + frustum + [position, foot, origin]
        + [axis_length * unit[i] for i in range(3)]
        + [position + camera_axis_length * v for v in (cam_x, cam_y, cam_z)])

    projected = np.column_stack((all_points @ right, all_points @ up))

    text_height = 84
    margin = 28
    scene_height = height - text_height
    low, high = projected.min(axis=0), projected.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    scale = min((width - 2 * margin) / span[0], (scene_height - 2 * margin) / span[1])
    screen_center = np.array([width / 2.0, scene_height / 2.0])
    scene_center = (low + high) / 2.0

    def to_px(point):
        s = np.array([np.dot(point, right), np.dot(point, up)])
        p = screen_center + (s - scene_center) * scale * np.array([1.0, -1.0])
        return int(round(p[0])), int(round(p[1]))

    def polygon(points):
        return np.array([to_px(p) for p in points], dtype=np.int32)

    def arrow(start, end, color, label, thickness, font_scale):
        p0, p1 = to_px(start), to_px(end)
        cv2.arrowedLine(image, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.14)
        cv2.putText(image, label, (p1[0] + 5, p1[1] - 5), FONT, font_scale,
                    color, 1, cv2.LINE_AA)

    # folha e marcadores
    board_poly = polygon(board)
    cv2.fillConvexPoly(image, board_poly, (66, 66, 66), cv2.LINE_AA)
    cv2.polylines(image, [board_poly], True, (120, 120, 120), 1, cv2.LINE_AA)

    marker_px = 2.0 * half * scale
    centers = [(-cx, cy), (cx, cy), (cx, -cy), (-cx, -cy)]
    corner_signs = ((-1, 1), (1, 1), (1, -1), (-1, -1))

    for marker_id, (mx, my) in zip(layout.ids, centers):
        outer = polygon([np.array([mx + sx * half, my + sy * half, 0.0]) for sx, sy in corner_signs])
        inner = polygon([np.array([mx + sx * half * 0.62, my + sy * half * 0.62, 0.0]) for sx, sy in corner_signs])
        cv2.fillConvexPoly(image, outer, (15, 15, 15), cv2.LINE_AA)
        cv2.polylines(image, [outer], True, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.fillConvexPoly(image, inner, (235, 235, 235), cv2.LINE_AA)

        if marker_px > 24:
            label = str(marker_id)
            size = cv2.getTextSize(label, FONT, 0.45, 1)[0]
            center = to_px(np.array([mx, my, 0.0]))
            cv2.putText(image, label, (center[0] - size[0] // 2, center[1] + size[1] // 2),
                        FONT, 0.45, (15, 15, 15), 1, cv2.LINE_AA)

    # linha de visada, altura e camera
    _dashed_line(image, to_px(position), to_px(origin), (130, 130, 130))
    _dashed_line(image, to_px(position), to_px(foot), (90, 160, 200))
    cv2.circle(image, to_px(foot), 3, (90, 160, 200), -1, cv2.LINE_AA)

    frustum_poly = polygon(frustum)
    apex = to_px(position)
    for corner in frustum_poly:
        cv2.line(image, apex, tuple(corner), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.polylines(image, [frustum_poly], True, (230, 230, 230), 1, cv2.LINE_AA)

    for name, vector in zip("xyz", (cam_x, cam_y, cam_z)):
        arrow(position, position + camera_axis_length * vector,
              AXIS_COLORS[name.upper()], name, 1, 0.42)
    cv2.circle(image, apex, 4, WHITE, -1, cv2.LINE_AA)

    # eixos do ArUco por cima
    for i, name in enumerate("XYZ"):
        arrow(origin, axis_length * unit[i], AXIS_COLORS[name], name, 2, 0.55)

    # legenda e medidas
    cv2.putText(image, "XYZ = ArUco   xyz = camera", (10, 18), FONT, 0.42,
                (170, 170, 170), 1, cv2.LINE_AA)
    if not live:
        cv2.putText(image, "SEM DETECCAO - ultima pose valida", (10, 38), FONT, 0.5,
                    (0, 165, 255), 1, cv2.LINE_AA)

    tilt = float(np.degrees(np.arccos(np.clip(-cam_z[2], -1.0, 1.0))))
    lines = [
        f"Distancia ao centro: {distance:.0f} mm",
        f"Altura Z: {position[2]:.0f} mm | X, Y: {position[0]:.0f}, {position[1]:.0f} mm",
        f"Inclinacao: {tilt:.1f} graus (0 = olhando de cima)",
        f"f = {pose.focal_px:.0f} px ({'assumida' if pose.focal_assumed else 'informada'})"
        f" | reproj. {pose.reprojection_px:.2f} px",
    ]
    y = height - text_height + 22
    for text in lines:
        cv2.putText(image, text, (12, y), FONT, 0.44, (225, 225, 225), 1, cv2.LINE_AA)
        y += 19

    return image


# ============================================================
# 4. DESENHO
# ============================================================

def create_curvature_colors(curvature):
    """
    |curvatura| -> cor BGR (verde = reta, amarelo, vermelho = mais curvo).
    Retorna array Nx3.
    """

    magnitude = np.abs(curvature)
    reference = np.percentile(magnitude, 95)

    if reference <= 1e-12:
        reference = 1.0

    value = np.clip(magnitude / reference, 0.0, 1.0)

    red = np.clip(2.0 * value, 0.0, 1.0) * 255
    green = np.clip(2.0 - 2.0 * value, 0.0, 1.0) * 255
    blue = np.zeros_like(value)

    return np.column_stack((blue, green, red)).astype(int)


def draw_centerline(frame, centerline, settings):
    colors = create_curvature_colors(centerline.curvature[centerline.valid])
    offset = centerline.valid.start

    points = np.column_stack((centerline.x, centerline.y))
    points = np.round(points).astype(int)

    for i in range(len(points) - 1):
        color_index = min(max(i - offset, 0), len(colors) - 1)
        color = tuple(int(c) for c in colors[color_index])

        cv2.line(
            frame, tuple(points[i]), tuple(points[i + 1]),
            color, settings.centerline_thickness, cv2.LINE_AA,
        )


def mark_minimum_radius(frame, centerline, index):
    cv2.circle(
        frame,
        (int(round(centerline.x[index])), int(round(centerline.y[index]))),
        9, WHITE, 2, cv2.LINE_AA,
    )


def draw_selection(frame, centerline, index, settings):
    """Marca o ponto selecionado e desenha o circulo osculador."""

    magenta = (255, 0, 255)

    px = centerline.x[index]
    py = centerline.y[index]

    cv2.circle(frame, (int(round(px)), int(round(py))), 6, magenta, -1, cv2.LINE_AA)

    curvature_px = centerline.curvature[index] * settings.mm_per_pixel

    if abs(curvature_px) > 1e-6:
        radius_px = 1.0 / abs(curvature_px)

        if radius_px < 20000:
            # centro = P + N / k  (k com sinal)
            cx = px + centerline.normal_x[index] / curvature_px
            cy = py + centerline.normal_y[index] / curvature_px

            cv2.circle(
                frame, (int(round(cx)), int(round(cy))),
                int(round(radius_px)), magenta, 1, cv2.LINE_AA,
            )

    radius = centerline.radius[index]
    label = f"R = {radius:.2f} {settings.unit}" if np.isfinite(radius) else "R = inf (reto)"

    cv2.putText(
        frame, label, (int(px) + 10, int(py) - 10),
        FONT, 0.6, magenta, 2, cv2.LINE_AA,
    )


def draw_information_panel(frame, lines):
    panel = frame.copy()
    height = 20 + 30 * len(lines)

    cv2.rectangle(panel, (10, 10), (560, height), (0, 0, 0), cv2.FILLED)
    output = cv2.addWeighted(panel, 0.5, frame, 0.5, 0.0)

    for i, text in enumerate(lines):
        cv2.putText(
            output, text, (25, 40 + i * 30),
            FONT, 0.65, WHITE, 2, cv2.LINE_AA,
        )

    return output


def draw_rounded_rectangle(image, pt1, pt2, color, radius=10, thickness=-1):
    """Draws a rounded rectangle using only OpenCV primitives."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = int(max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
        for center in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(image, center, radius, color, cv2.FILLED, cv2.LINE_AA)
    else:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def put_text_right(image, text, right_x, y, scale, color, thickness=1):
    size = cv2.getTextSize(text, FONT, scale, thickness)[0]
    cv2.putText(image, text, (right_x - size[0], y), FONT, scale,
                color, thickness, cv2.LINE_AA)


def create_info_view(lines, width=480, height=330):
    """Creates a card-based information panel with visual hierarchy."""
    # Altura necessaria (mesmos passos verticais de baixo): nada fica fora da imagem
    needed = 55
    for row in lines:
        kind = "plain" if isinstance(row, str) else row[0]
        if kind == "section":
            needed += (5 if needed > 60 else 0) + 15
        elif kind in ("status_ok", "status_warn"):
            needed += 32
        elif kind in ("metric", "highlight"):
            needed += 31
        else:
            needed += 24
    height = max(height, needed + 4)

    image = np.full((height, width, 3), THEME["background"], dtype=np.uint8)
    margin = 14
    y = 14

    # Panel heading
    cv2.putText(image, "LIVE ANALYSIS", (margin, 25), FONT, 0.52,
                THEME["text"], 1, cv2.LINE_AA)
    cv2.circle(image, (width - 24, 20), 5, THEME["success"], cv2.FILLED, cv2.LINE_AA)
    cv2.line(image, (margin, 36), (width - margin, 36), THEME["border"], 1)
    y = 55

    for row in lines:
        if isinstance(row, str):
            # Backwards compatibility for calibration-progress messages.
            row = ("plain", row)
        kind = row[0]

        if kind == "section":
            if y > 60:
                y += 5
            cv2.putText(image, row[1], (margin, y), FONT, 0.38,
                        THEME["accent"], 1, cv2.LINE_AA)
            y += 15
            continue

        if kind in ("status_ok", "status_warn"):
            color = THEME["success"] if kind == "status_ok" else THEME["warning"]
            draw_rounded_rectangle(image, (margin, y - 11), (width - margin, y + 13),
                                   THEME["panel_alt"], radius=7)
            cv2.circle(image, (margin + 12, y + 1), 4, color, cv2.FILLED, cv2.LINE_AA)
            cv2.putText(image, row[1], (margin + 24, y + 6), FONT, 0.42,
                        THEME["text"], 1, cv2.LINE_AA)
            y += 32
            continue

        if kind in ("metric", "highlight"):
            label, value = row[1], row[2]
            card_color = THEME["accent_soft"] if kind == "highlight" else THEME["panel"]
            value_color = THEME["accent"] if kind == "highlight" else THEME["text"]
            draw_rounded_rectangle(image, (margin, y - 12), (width - margin, y + 14),
                                   card_color, radius=7)
            cv2.putText(image, label, (margin + 10, y + 6), FONT, 0.39,
                        THEME["muted"], 1, cv2.LINE_AA)
            put_text_right(image, value, width - margin - 10, y + 6, 0.41,
                           value_color, 1)
            y += 31
            continue

        cv2.putText(image, str(row[1]), (margin, y), FONT, 0.40,
                    THEME["text"], 1, cv2.LINE_AA)
        y += 24

        if y > height - 16:
            break
    return image

def build_rectification_view(original_frame, corrected_frame, calibrated):
    """Compara a perspectiva original com a imagem corrigida pelo ArUco."""
    target_height = 360

    def fit_height(image):
        scale = target_height / image.shape[0]
        width = max(1, int(round(image.shape[1] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, (width, target_height), interpolation=interpolation)

    original = fit_height(original_frame)
    if calibrated:
        corrected = fit_height(corrected_frame)
    else:
        corrected = np.zeros_like(original)
        cv2.putText(corrected, "AGUARDANDO CALIBRACAO ArUco",
                    (20, target_height // 2), FONT, 0.65,
                    (0, 255, 255), 2, cv2.LINE_AA)

    label_h = 34
    left = np.zeros((target_height + label_h, original.shape[1], 3), np.uint8)
    right = np.zeros((target_height + label_h, corrected.shape[1], 3), np.uint8)
    left[:label_h] = (45, 45, 45)
    right[:label_h] = (45, 45, 45)
    left[label_h:] = original
    right[label_h:] = corrected
    cv2.putText(left, "ORIGINAL / PERSPECTIVA", (10, 23), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(right, "RETIFICADA POR ArUco", (10, 23), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return np.hstack((left, right))


def build_info_lines(analysis, stats, settings, calibration, selected_radius):
    """Creates structured rows for the information panel."""
    unit = settings.unit
    rows = [
        ("section", "MEASUREMENT"),
        ("metric", "Total length", f"{stats.length:.2f} {unit}"),
        ("metric", "Maximum curvature", f"{stats.max_curvature:.5f} 1/{unit}"),
        ("metric", "Minimum radius", f"{stats.min_radius:.2f} {unit}"),
        ("metric", "Maximum deviation", f"{stats.max_line_deviation:.2f} {unit}"),
        ("metric", "Segmented area", f"{analysis.area:,} px2"),
    ]

    if selected_radius is not None:
        value = (f"{selected_radius:.2f} {unit}"
                 if np.isfinite(selected_radius) else "infinite")
        rows.extend([
            ("section", "SELECTION"),
            ("highlight", "Local radius", value),
        ])

    rows.append(("section", "CALIBRATION"))
    if calibration is not None:
        ppm = 1.0 / calibration.mm_per_pixel
        rows.extend([
            ("status_ok", "ArUco calibration active"),
            ("metric", "Method", calibration.method),
            ("metric", "Scale", f"{calibration.mm_per_pixel:.5f} mm/px"),
            ("metric", "Resolution", f"{ppm:.2f} px/mm"),
            ("metric", "Reprojection RMS", f"{calibration.rms_error_mm:.3f} mm"),
        ])
    elif settings.calibrated:
        rows.extend([
            ("status_warn", "Fixed scale active"),
            ("metric", "Scale", f"{settings.mm_per_pixel:.5f} mm/px"),
            ("metric", "Perspective", "not corrected"),
        ])
    else:
        rows.extend([
            ("status_warn", "No metric calibration"),
            ("metric", "Output unit", "pixels"),
        ])
    return rows

def build_mask_view(mask, threshold_value, used_threshold):
    """Mascara em BGR com o limiar escrito (a mascara original nao e alterada)."""

    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mode = "auto" if threshold_value <= 0 else "manual"

    cv2.putText(
        view, f"Threshold: {int(used_threshold)} ({mode})",
        (20, 30), FONT, 0.7, THEME["success"], 2, cv2.LINE_AA,
    )

    return view


def create_graph(
    arc_length, values, title, y_limits, x_unit,
    marker_x=None, width=700, height=300,
):
    """Dark themed OpenCV graph with grid, labels, and selection marker."""
    graph = np.full((height, width, 3), THEME["background"], dtype=np.uint8)
    left, right, top, bottom = 74, 24, 42, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    cv2.putText(graph, title, (18, 27), FONT, 0.55,
                THEME["text"], 1, cv2.LINE_AA)
    cv2.putText(graph, f"Position along cable ({x_unit})",
                (left + max(10, plot_w // 2 - 90), height - 13), FONT, 0.42,
                THEME["muted"], 1, cv2.LINE_AA)

    # Plot card and soft grid
    draw_rounded_rectangle(graph, (left, top), (width - right, height - bottom),
                           THEME["panel"], radius=8)
    for i in range(1, 5):
        gy = top + int(i * plot_h / 5)
        cv2.line(graph, (left, gy), (width - right, gy), THEME["grid"], 1, cv2.LINE_AA)
    for i in range(1, 6):
        gx = left + int(i * plot_w / 6)
        cv2.line(graph, (gx, top), (gx, height - bottom), THEME["grid"], 1, cv2.LINE_AA)

    if arc_length is None or values is None:
        cv2.putText(graph, "Waiting for a valid centerline...",
                    (left + 24, top + plot_h // 2), FONT, 0.48,
                    THEME["muted"], 1, cv2.LINE_AA)
        return graph

    y_min, y_max = y_limits
    x_span = arc_length[-1] - arc_length[0]
    if x_span <= 1e-12 or y_max - y_min <= 1e-12:
        return graph

    def to_graph_y(value):
        fraction = (np.clip(value, y_min, y_max) - y_min) / (y_max - y_min)
        return height - bottom - fraction * plot_h

    for value in (y_min, (y_min + y_max) / 2, y_max):
        cv2.putText(graph, f"{value:.3g}", (8, int(to_graph_y(value)) + 4),
                    FONT, 0.36, THEME["muted"], 1, cv2.LINE_AA)

    if y_min < 0.0 < y_max:
        zero_y = int(to_graph_y(0.0))
        cv2.line(graph, (left, zero_y), (width - right, zero_y),
                 (105, 112, 125), 1, cv2.LINE_AA)

    graph_x = left + (arc_length - arc_length[0]) / x_span * plot_w
    points = np.column_stack((graph_x, to_graph_y(values))).astype(np.int32)
    cv2.polylines(graph, [points], False, THEME["curve"], 2, cv2.LINE_AA)

    # Subtle fill under the curve when it is non-negative.
    if y_min >= 0 and len(points) > 1:
        polygon = np.vstack((points, [points[-1, 0], height - bottom],
                             [points[0, 0], height - bottom])).astype(np.int32)
        overlay = graph.copy()
        cv2.fillPoly(overlay, [polygon], THEME["accent_soft"], cv2.LINE_AA)
        graph = cv2.addWeighted(overlay, 0.24, graph, 0.76, 0)
        cv2.polylines(graph, [points], False, THEME["curve"], 2, cv2.LINE_AA)

    if marker_x is not None:
        mx = int(left + (marker_x - arc_length[0]) / x_span * plot_w)
        cv2.line(graph, (mx, top), (mx, height - bottom),
                 (220, 75, 210), 1, cv2.LINE_AA)
        cv2.circle(graph, (mx, int(to_graph_y(np.interp(marker_x, arc_length, values)))),
                   4, (220, 75, 210), cv2.FILLED, cv2.LINE_AA)
    return graph

def build_graphs(unit, centerline=None, stats=None, selected=None):
    """Retorna (grafico de curvatura, grafico de raio); vazios se nao houver curva."""

    curvature_graph = create_graph(
        None, None, f"Curvatura (1/{unit})", (-1, 1), unit
    )
    radius_graph = create_graph(
        None, None, f"Raio de curvatura ({unit})", (0, 1), unit
    )

    if centerline is None:
        return curvature_graph, radius_graph

    v = centerline.valid
    s = centerline.arc_length[v]
    k_valid = centerline.curvature[v]
    marker = centerline.arc_length[selected] if selected is not None else None

    k_limit = max(float(np.percentile(np.abs(k_valid), 98)) * 1.1, 1e-9)
    curvature_graph = create_graph(
        s, k_valid, f"Curvatura (1/{unit})", (-k_limit, k_limit), unit,
        marker_x=marker,
    )

    r_limit = 10.0 * stats.min_radius if np.isfinite(stats.min_radius) else 1.0
    radius_graph = create_graph(
        s, centerline.radius[v],
        f"Raio de curvatura ({unit}) - cortado em 10x o minimo",
        (0.0, r_limit), unit, marker_x=marker,
    )

    return curvature_graph, radius_graph


def render_analysis(original_frame, display_frame, analysis, settings,
                    calibration, click, threshold_value) -> Views:
    """Monta as seis imagens do dashboard sem cobrir o video com legenda."""
    result_frame = display_frame.copy()
    cv2.drawContours(result_frame, analysis.contours, -1, (255, 255, 0), 1)
    centerline = analysis.centerline
    stats = None
    selected = None
    selected_radius = None

    if centerline is None:
        cv2.putText(result_frame, "Objeto nao detectado", (20, 40),
                    FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        lines = ["Objeto nao detectado"]
        if calibration is not None:
            lines.append(f"Calibracao ArUco ({calibration.method}): RMS {calibration.rms_error_mm:.3f} mm")
        elif settings.calibrated:
            lines.append(f"Escala fixa: {settings.mm_per_pixel:.5f} mm/px")
        else:
            lines.append("Sem calibracao: resultados em pixels")
    else:
        stats = compute_statistics(centerline, settings)
        draw_centerline(result_frame, centerline, settings)
        mark_minimum_radius(result_frame, centerline, stats.min_radius_index)
        if click is not None:
            selected = nearest_valid_index(centerline, click)
            draw_selection(result_frame, centerline, selected, settings)
            selected_radius = centerline.radius[selected]
        lines = build_info_lines(
            analysis, stats, settings, calibration, selected_radius
        )

    curvature_graph, radius_graph = build_graphs(
        settings.unit, centerline, stats, selected
    )
    return Views(
        result=result_frame,
        info=create_info_view(lines),
        rectification=build_rectification_view(
            original_frame, display_frame, calibration is not None
        ),
        mask=build_mask_view(analysis.mask, threshold_value, analysis.threshold),
        curvature_graph=curvature_graph,
        radius_graph=radius_graph,
        centerline=centerline,
    )

def render_calibration(frame, found, calibrator, settings) -> Views:
    """Tela mostrada enquanto os marcadores ainda nao foram calibrados."""

    result_frame = frame.copy()
    layout = calibrator.layout

    for marker_id, corners in found.items():
        in_layout = marker_id in layout.ids
        color = (0, 255, 0) if in_layout else (0, 0, 255)  # vermelho = id ignorado
        polygon = np.round(corners).astype(np.int32)

        cv2.polylines(result_frame, [polygon], True, color, 3, cv2.LINE_AA)
        cv2.putText(
            result_frame, f"ID {marker_id}", tuple(polygon[0] + [5, -8]),
            FONT, 0.8, color, 2, cv2.LINE_AA,
        )

    missing = [marker_id for marker_id in layout.ids if marker_id not in found]

    if missing:
        message = f"CALIBRACAO ArUco: faltam os ids {missing}"
    else:
        message = (
            f"CALIBRACAO ArUco: {calibrator.progress}/{calibrator.frames_required} "
            f"- mantenha a cena parada"
        )

    cv2.putText(
        result_frame, message, (20, result_frame.shape[0] - 25),
        FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )

    curvature_graph, radius_graph = build_graphs(settings.unit)
    blank = np.zeros(frame.shape[:2], np.uint8)
    info_lines = [
        "Calibracao ArUco em andamento",
        f"Progresso: {calibrator.progress}/{calibrator.frames_required} frames",
        f"IDs esperados: {list(layout.ids)}",
        f"IDs detectados: {sorted(found.keys())}",
        f"IDs faltando: {missing}",
        "Mascara e graficos continuam ativos em pixels.",
    ]
    return Views(
        result=result_frame,
        info=create_info_view(info_lines),
        rectification=build_rectification_view(frame, frame, False),
        mask=blank,
        curvature_graph=curvature_graph,
        radius_graph=radius_graph,
    )


# ============================================================
# 5. EXPORTACAO
# ============================================================

def save_snapshot(output_dir, result_frame, centerline, settings):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.join(output_dir, "curvatura_" + time.strftime("%Y%m%d_%H%M%S"))
    unit = settings.unit

    cv2.imwrite(base + "_resultado.png", result_frame)

    data = np.column_stack((
        centerline.arc_length,
        centerline.x * settings.mm_per_pixel,
        centerline.y * settings.mm_per_pixel,
        centerline.curvature,
        centerline.radius,
    ))

    np.savetxt(
        base + ".csv", data, delimiter=",",
        header=f"s_{unit},x_{unit},y_{unit},curvatura_1/{unit},raio_{unit}",
        comments="", fmt="%.6g",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[AVISO] matplotlib nao instalado: grafico PNG nao gerado.")
    else:
        v = centerline.valid
        s = centerline.arc_length[v]
        radius = np.where(np.isfinite(centerline.radius), centerline.radius, np.nan)

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        axes[0].plot(s, centerline.curvature[v])
        axes[0].axhline(0, color="gray", lw=0.8)
        axes[0].set_ylabel(f"Curvatura (1/{unit})")
        axes[0].grid(alpha=0.3)

        axes[1].semilogy(s, radius[v])
        axes[1].set_ylabel(f"Raio de curvatura ({unit})")
        axes[1].set_xlabel(f"Posicao ao longo do cabo ({unit})")
        axes[1].grid(alpha=0.3, which="both")

        fig.tight_layout()
        fig.savefig(base + "_graficos.png", dpi=150)
        plt.close(fig)

    print(f"[OK] Snapshot salvo: {base}.*", flush=True)


# ============================================================
# 6. INTERFACE: JANELAS, DASHBOARD E MOUSE
# ============================================================

def resize_with_letterbox(image, width, height):
    """
    Redimensiona a imagem para caber em (width, height) sem alterar a
    proporcao, completando com barras pretas.

    Retorna (canvas BGR, escala, x0, y0).
    """

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    img_h, img_w = image.shape[:2]

    scale = min(width / img_w, height / img_h)
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas, scale, x0, y0


def show_fit(window, image):
    """
    imshow que preserva a proporcao mesmo quando a janela e redimensionada
    (KEEPRATIO nao funciona no backend Windows do OpenCV).

    Retorna (escala, x0, y0), usados para converter cliques.
    """

    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window)
    except cv2.error:
        win_w = win_h = 0

    if win_w < 2 or win_h < 2:
        cv2.imshow(window, image)
        return 1.0, 0, 0

    canvas, scale, x0, y0 = resize_with_letterbox(image, win_w, win_h)
    cv2.imshow(window, canvas)

    return scale, x0, y0


def make_panel(image, title, width, height):
    """Creates a polished dashboard panel without covering its content."""
    gutter = 4
    inner_w = max(1, width - 2 * gutter)
    inner_h = max(1, height - 2 * gutter)
    content_h = max(1, inner_h - TITLE_HEIGHT)
    content, scale, x0, y0 = resize_with_letterbox(image, inner_w, content_h)

    panel = np.full((height, width, 3), THEME["background"], dtype=np.uint8)
    x1, y1 = gutter, gutter
    x2, y2 = width - gutter - 1, height - gutter - 1
    draw_rounded_rectangle(panel, (x1, y1), (x2, y2), THEME["panel"], radius=9)
    cv2.rectangle(panel, (x1 + 1, y1 + 1), (x2 - 1, y1 + TITLE_HEIGHT),
                  THEME["header"], cv2.FILLED)
    cv2.rectangle(panel, (x1 + 1, y1 + TITLE_HEIGHT - 2),
                  (x2 - 1, y1 + TITLE_HEIGHT), THEME["accent"], cv2.FILLED)
    panel[y1 + TITLE_HEIGHT:y1 + TITLE_HEIGHT + content.shape[0],
          x1:x1 + content.shape[1]] = content
    cv2.putText(panel, title, (x1 + 12, y1 + 20), FONT, 0.46,
                THEME["text"], 1, cv2.LINE_AA)
    draw_rounded_rectangle(panel, (x1, y1), (x2, y2), THEME["border"],
                           radius=9, thickness=1)
    return panel, scale, x0 + x1, y0 + y1

def _format_time(seconds):
    seconds = max(float(seconds), 0.0)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def slider_value(name, fraction):
    """Fracao (0..1) da trilha -> valor do slider (arredondado ao passo)."""

    spec = SLIDERS[name]
    fraction = min(max(fraction, 0.0), 1.0)
    raw = spec["min"] + fraction * (spec["max"] - spec["min"])
    value = round(raw / spec["step"]) * spec["step"]

    return int(value) if isinstance(spec["step"], int) else round(float(value), 1)


def slider_fraction(name, value):
    spec = SLIDERS[name]
    return min(max((value - spec["min"]) / (spec["max"] - spec["min"]), 0.0), 1.0)


def make_controls_panel(width, height, paused, frame_index, total_frames,
                        fps, slider_values, has_video):
    """
    Painel CONTROLES: botoes e barra de progresso do video + sliders.

    Retorna (imagem, botoes, sliders, barra); as regioes (x, y, w, h) sao
    RELATIVAS ao painel.
    """

    panel = np.full((height, width, 3), THEME["panel"], dtype=np.uint8)
    panel[:TITLE_HEIGHT] = THEME["header"]
    cv2.putText(panel, "CONTROLES", (10, 21), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    light = THEME["text"]
    dim = THEME["muted"]
    accent = THEME["accent"]
    margin = 24

    # ---- reproducao ----
    if has_video and total_frames > 0:
        seconds = frame_index / fps
        total_seconds = max(total_frames - 1, 0) / fps
        status = f"{_format_time(seconds)} / {_format_time(total_seconds)}   frame {frame_index}/{max(total_frames - 1, 0)}"
    else:
        status = "webcam (sem navegacao)"
    cv2.putText(panel, status, (margin, 52), FONT, 0.45, light, 1, cv2.LINE_AA)

    labels = [
        ("back_1s", "-1s", 56, has_video),
        ("prev", "<", 44, has_video),
        ("toggle", "PLAY" if paused else "PAUSE", 84, True),
        ("next", ">", 44, has_video),
        ("forward_1s", "+1s", 56, has_video),
    ]
    gap = 6
    button_h = 34
    button_y = 64
    total_width = sum(item[2] for item in labels) + gap * (len(labels) - 1)
    x = max(margin, (width - total_width) // 2)

    buttons = {}
    for action, label, button_w, enabled in labels:
        color = (70, 120, 70) if action == "toggle" else (65, 65, 65)
        text_color = WHITE if enabled else dim
        cv2.rectangle(panel, (x, button_y), (x + button_w, button_y + button_h), color, cv2.FILLED)
        cv2.rectangle(panel, (x, button_y), (x + button_w, button_y + button_h),
                      (220, 220, 220) if enabled else (90, 90, 90), 1)
        size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
        cv2.putText(panel, label,
                    (x + (button_w - size[0]) // 2, button_y + (button_h + size[1]) // 2),
                    FONT, 0.5, text_color, 1, cv2.LINE_AA)
        if enabled:
            buttons[action] = (x, button_y, button_w, button_h)
        x += button_w + gap

    progress = None
    if has_video and total_frames > 1:
        bar_x, bar_y, bar_w, bar_h = margin, 116, width - 2 * margin, 12
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (85, 85, 85), cv2.FILLED)
        fraction = min(max(frame_index / (total_frames - 1), 0.0), 1.0)
        filled = int(round(bar_w * fraction))
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), accent, cv2.FILLED)
        cv2.circle(panel, (bar_x + filled, bar_y + bar_h // 2), 8, light, -1, cv2.LINE_AA)
        progress = (bar_x, bar_y, bar_w, bar_h)

    # ---- sliders ----
    cv2.line(panel, (margin, 150), (width - margin, 150), (80, 80, 80), 1)
    cv2.putText(panel, "SEGMENTACAO E AJUSTE", (margin, 168), FONT, 0.42, dim, 1, cv2.LINE_AA)

    threshold = int(slider_values.get("threshold", 0))
    smoothing = float(slider_values.get("smoothing", 1.0))
    slider_specs = [
        ("threshold", "Threshold: auto (Otsu)" 
         if threshold <= 0 else f"Threshold: {threshold}", 206),
        ("smoothing", f"Suavizacao da spline: {smoothing:.1f} px", 254),
    ]

    sliders = {}
    track_x, track_w = margin, width - 2 * margin
    for name, label, track_y in slider_specs:
        cv2.putText(panel, label, (track_x, track_y - 14), FONT, 0.48, light, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (track_x, track_y), (track_x + track_w, track_y + 6), (85, 85, 85), cv2.FILLED)
        filled = int(round(track_w * slider_fraction(name, slider_values.get(name, 0))))
        cv2.rectangle(panel, (track_x, track_y), (track_x + filled, track_y + 6), accent, cv2.FILLED)
        cv2.circle(panel, (track_x + filled, track_y + 3), 9, light, -1, cv2.LINE_AA)
        cv2.circle(panel, (track_x + filled, track_y + 3), 9, accent, 2, cv2.LINE_AA)
        sliders[name] = (track_x, track_y, track_w, 6)

    # ---- dicas ----
    cv2.putText(panel, "Clique no cabo: mostra R | Botao direito: limpa",
                (margin, height - 74), FONT, 0.4, dim, 1, cv2.LINE_AA)
    cv2.putText(panel, "ESPACO pausa | C calibra | S snapshot | Q sai",
                (margin, height - 54), FONT, 0.4, dim, 1, cv2.LINE_AA)
    cv2.putText(panel, "Threshold controla a separacao entre objeto escuro e fundo claro",
                (margin, height - 34), FONT, 0.4, dim, 1, cv2.LINE_AA)
    cv2.putText(panel, "Suavizacao da spline ajusta a curva de ajuste da mascara/skeleton",
                (margin, height - 14), FONT, 0.4, dim, 1, cv2.LINE_AA)


    return panel, buttons, sliders, progress


def create_dashboard(views, width, height, paused=False, frame_index=0,
                     total_frames=0, fps=30.0, slider_values=None, has_video=True):
    """Dashboard.

        RESULTADO (metade)    | INFORMACOES | POSE DA CAMERA
        ORIGINAL E CORRIGIDA  | MASCARA     | CONTROLES
        CURVATURA             | RAIO

    Retorna (imagem, frame_view, botoes, barra de progresso, sliders), com
    todas as regioes em coordenadas do dashboard.
    """
    half_w = width // 2
    quarter_w = (width - half_w) // 2
    last_w = width - half_w - quarter_w
    right_w = width - half_w
    row_h = height // 3
    last_h = height - 2 * row_h

    result_panel, scale, x0, y0 = make_panel(
        views.result, "RESULTADO", half_w, row_h
    )
    info_panel = make_panel(
        views.info, "INFORMACOES", quarter_w, row_h
    )[0]

    pose_image = views.pose
    if pose_image is None:
        pose_image = np.full((330, 480, 3), 38, dtype=np.uint8)
    pose_panel = make_panel(
        pose_image, "POSE DA CAMERA (ArUco XYZ / camera xyz)", last_w, row_h
    )[0]

    rectification_panel = make_panel(
        views.rectification, "ORIGINAL E CORRIGIDA POR ArUco", half_w, row_h
    )[0]
    mask_panel = make_panel(
        views.mask, "MASCARA", quarter_w, row_h
    )[0]

    controls_panel, buttons, sliders, progress = make_controls_panel(
        last_w, row_h, paused, frame_index, total_frames, fps,
        slider_values or {}, has_video,
    )

    curvature_panel = make_panel(
        views.curvature_graph, "CURVATURA", half_w, last_h
    )[0]
    radius_panel = make_panel(
        views.radius_graph, "RAIO DE CURVATURA", right_w, last_h
    )[0]

    dashboard = np.vstack((
        np.hstack((result_panel, info_panel, pose_panel)),
        np.hstack((rectification_panel, mask_panel, controls_panel)),
        np.hstack((curvature_panel, radius_panel)),
    ))

    # regioes do painel de controles -> coordenadas do dashboard
    origin_x, origin_y = half_w + quarter_w, row_h

    def shift(rect):
        rx, ry, rw, rh = rect
        return (rx + origin_x, ry + origin_y, rw, rh)

    control_regions = {name: shift(rect) for name, rect in buttons.items()}
    slider_regions = {name: shift(rect) for name, rect in sliders.items()}
    progress_region = shift(progress) if progress is not None else None

    frame_view = FrameView(
        x=x0,
        y=TITLE_HEIGHT + y0,
        scale=scale,
        width=views.result.shape[1],
        height=views.result.shape[0],
    )
    return dashboard, frame_view, control_regions, progress_region, slider_regions


def window_to_frame(x, y, interaction):
    """Converte um clique na janela para coordenadas do frame (ou None)."""

    view = interaction.frame_view
    fit_scale, fit_x0, fit_y0 = interaction.window_fit

    if view is None or fit_scale <= 0 or view.scale <= 0:
        return None

    # janela -> imagem exibida (dashboard ou resultado)
    shown_x = (x - fit_x0) / fit_scale
    shown_y = (y - fit_y0) / fit_scale

    # imagem exibida -> frame
    frame_x = (shown_x - view.x) / view.scale
    frame_y = (shown_y - view.y) / view.scale

    if not (0 <= frame_x < view.width and 0 <= frame_y < view.height):
        return None

    return frame_x, frame_y


def _shown_coordinates(x, y, interaction):
    """Janela -> imagem exibida (dashboard). None se ainda nao ha escala."""

    fit_scale, fit_x0, fit_y0 = interaction.window_fit
    if fit_scale <= 0:
        return None
    return (x - fit_x0) / fit_scale, (y - fit_y0) / fit_scale


def _drag_progress(interaction, shown_x):
    rx, ry, rw, rh = interaction.progress_region
    interaction.seek_fraction = min(max((shown_x - rx) / rw, 0.0), 1.0)
    interaction.action = "seek"


def _drag_slider(interaction, name, shown_x):
    rx, ry, rw, rh = interaction.slider_regions[name]
    interaction.slider_values[name] = slider_value(name, (shown_x - rx) / rw)


def on_mouse(event, x, y, flags, interaction):
    if event == cv2.EVENT_RBUTTONDOWN:
        interaction.click = None
        return

    if event == cv2.EVENT_LBUTTONUP:
        interaction.dragging = None
        return

    shown = _shown_coordinates(x, y, interaction)
    if shown is None:
        return
    shown_x, shown_y = shown

    # arrastando slider ou barra de progresso
    if event == cv2.EVENT_MOUSEMOVE:
        if interaction.dragging is None:
            return
        if not (flags & cv2.EVENT_FLAG_LBUTTON):
            interaction.dragging = None
            return
        if interaction.dragging == "progress" and interaction.progress_region is not None:
            _drag_progress(interaction, shown_x)
        elif interaction.dragging in (interaction.slider_regions or {}):
            _drag_slider(interaction, interaction.dragging, shown_x)
        return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for action, rect in (interaction.control_regions or {}).items():
        rx, ry, rw, rh = rect
        if rx <= shown_x <= rx + rw and ry <= shown_y <= ry + rh:
            interaction.action = action
            return

    for name, rect in (interaction.slider_regions or {}).items():
        rx, ry, rw, rh = rect
        if rx - 10 <= shown_x <= rx + rw + 10 and ry - 14 <= shown_y <= ry + rh + 14:
            interaction.dragging = name
            _drag_slider(interaction, name, shown_x)
            return

    if interaction.progress_region is not None:
        rx, ry, rw, rh = interaction.progress_region
        if rx - 10 <= shown_x <= rx + rw + 10 and ry - 8 <= shown_y <= ry + rh + 8:
            interaction.dragging = "progress"
            _drag_progress(interaction, shown_x)
            return

    point = window_to_frame(x, y, interaction)
    if point is not None:
        interaction.click = point


# ============================================================
# 7. APLICACAO
# ============================================================

class CurvatureApp:
    """Loop principal: captura, calibracao, analise e exibicao."""

    def __init__(self, args, layout: Optional[ArucoLayout]):
        self.args = args
        self.base_scale = args.mm_per_pixel  # escala fixa opcional (None = px)

        self.settings = Settings(
            mm_per_pixel=args.mm_per_pixel or 1.0,
            calibrated=args.mm_per_pixel is not None,
            smoothing_px=args.smoothing,
        )

        # Estado do mouse/sliders. Precisa existir ANTES de set_calibration().
        smoothing_spec = SLIDERS["smoothing"]
        self.interaction = Interaction()
        self.interaction.slider_values = {
            "threshold": 0,
            "smoothing": min(max(round(args.smoothing, 1), smoothing_spec["min"]),
                             smoothing_spec["max"]),
        }

        self.calibrator = ArucoCalibrator(layout, frames_required=args.calibration_frames) if layout is not None else None
        self.layout = layout
        self.calibration: Optional[PlanarCalibration] = None

        if args.load_calibration:
            self.set_calibration(load_calibration(args.calibration_file))
            print(f"[OK] Calibracao carregada: {args.calibration_file}")

        self.paused = False
        self.ended = False          # video chegou ao fim (PLAY volta ao inicio)
        self.capture = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.source_fps = 30.0
        self.last_found = {}
        self.last_views: Optional[Views] = None

        # Pose da camera (so com layout ArUco)
        self.pose_detector = create_aruco_detector(layout.dictionary) if layout is not None else None
        self.pose: Optional[CameraPose] = None
        self.pose_is_live = False
        self._pose_frame = None

    # ---------- calibracao ----------

    @property
    def needs_calibration(self) -> bool:
        return self.calibrator is not None and self.calibration is None

    def set_calibration(self, calibration):
        self.calibration = calibration
        self.settings.mm_per_pixel = calibration.mm_per_pixel
        self.settings.calibrated = True
        self.interaction.click = None

    def restart_calibration(self):
        if self.calibrator is None:
            print("[AVISO] Informe --aruco-size e --aruco-spacing para calibrar.")
            return

        self.calibration = None
        self.calibrator.reset()
        self.settings.mm_per_pixel = self.base_scale or 1.0
        self.settings.calibrated = self.base_scale is not None
        self.interaction.click = None
        self.paused = False
        print("[CALIBRACAO] Refazendo: mantenha os 4 marcadores visiveis.")

    def update_calibration(self, frame):
        """Alimenta o calibrador com o frame atual e conclui se possivel."""

        self.last_found, calibration = self.calibrator.update(frame)

        if calibration is None:
            return

        self.set_calibration(calibration)
        save_calibration(self.args.calibration_file, calibration, self.layout)

        print("[OK] Calibracao ArUco concluida")
        print(f"     Escala: {calibration.mm_per_pixel:.6f} mm/pixel "
              f"({1.0 / calibration.mm_per_pixel:.2f} px/mm)")
        print(f"     Erro RMS de reprojecao: {calibration.rms_error_mm:.3f} mm")
        print(f"     Imagem retificada: {calibration.output_size[0]}x{calibration.output_size[1]} px")
        print(f"     Salva em: {self.args.calibration_file}")

    # ---------- processamento ----------

    def read_controls(self):
        """(threshold, suavizacao em px) vindos dos sliders do dashboard ou das trackbars."""

        if self.args.display == "dashboard":
            values = self.interaction.slider_values
            return int(values["threshold"]), float(values["smoothing"])

        threshold = cv2.getTrackbarPos(TRACKBAR_THRESHOLD, WIN_CONTROLS)
        smoothing = max(cv2.getTrackbarPos(TRACKBAR_SMOOTHING, WIN_CONTROLS), 1) / 10.0
        return threshold, smoothing

    def update_pose(self, frame):
        """Atualiza a pose da camera. Com a cena pausada reaproveita o resultado."""

        if self.layout is None or frame is self._pose_frame:
            return

        self._pose_frame = frame

        if self.needs_calibration and not self.paused:
            found = self.last_found  # o calibrador acabou de detectar neste frame
        else:
            found = detect_markers(self.pose_detector, frame)

        pose = estimate_camera_pose(
            found, self.layout, frame.shape, getattr(self.args, "focal_px", None))

        self.pose_is_live = pose is not None
        if pose is not None:
            self.pose = pose

    def build_views(self, frame) -> Views:
        threshold_value, smoothing_px = self.read_controls()
        self.settings.smoothing_px = smoothing_px

        if self.needs_calibration and not self.paused:
            self.update_calibration(frame)

        self.update_pose(frame)

        # Eixos XYZ so na copia exibida; a analise usa o frame limpo
        original_frame = frame
        if self.pose is not None and self.pose_is_live:
            original_frame = frame.copy()
            draw_pose_axes(original_frame, self.pose, self.layout)

        display_frame = frame
        segmentation_frame = frame
        if self.calibration is not None:
            display_frame = rectify_frame(frame, self.calibration)
            segmentation_frame = whiten_markers(display_frame, self.calibration)

        analysis = analyze_frame(segmentation_frame, self.settings, threshold_value)
        views = render_analysis(
            original_frame, display_frame, analysis, self.settings,
            self.calibration, self.interaction.click, threshold_value,
        )
        views.pose = build_pose_view(self.pose, self.layout, self.pose_is_live)

        if self.needs_calibration:
            layout = self.calibrator.layout
            missing = [m for m in layout.ids if m not in self.last_found]
            info_lines = [
                "Calibracao ArUco em andamento",
                f"Progresso: {self.calibrator.progress}/{self.calibrator.frames_required}",
                f"IDs esperados: {list(layout.ids)}",
                f"IDs detectados: {sorted(self.last_found.keys())}",
                f"IDs faltando: {missing}",
                "Medidas temporarias em pixels.",
            ]
            views.info = create_info_view(info_lines)
            for marker_id, corners in self.last_found.items():
                color = (0, 255, 0) if marker_id in layout.ids else (0, 0, 255)
                polygon = np.round(corners).astype(np.int32)
                cv2.polylines(views.result, [polygon], True, color, 3, cv2.LINE_AA)
                cv2.putText(views.result, f"ID {marker_id}",
                            tuple(polygon[0] + [5, -8]), FONT, 0.7,
                            color, 2, cv2.LINE_AA)
        return views

    # ---------- exibicao ----------

    def create_windows(self):
        args = self.args

        if args.display == "dashboard":
            cv2.namedWindow(WIN_DASHBOARD, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_DASHBOARD, args.dashboard_width, args.dashboard_height)
            mouse_window = WIN_DASHBOARD
            # Controles (playback, threshold, suavizacao) ficam DENTRO do dashboard
        else:
            for name in (WIN_RESULT, WIN_INFO, WIN_RECTIFICATION, WIN_MASK, WIN_CURVATURE, WIN_RADIUS):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_RESULT, 900, 600)
            cv2.resizeWindow(WIN_MASK, 500, 350)
            cv2.resizeWindow(WIN_INFO, 600, 350)
            cv2.resizeWindow(WIN_RECTIFICATION, 800, 350)
            cv2.resizeWindow(WIN_CURVATURE, 700, 300)
            cv2.resizeWindow(WIN_RADIUS, 700, 300)
            mouse_window = WIN_RESULT

            cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)
            cv2.createTrackbar(TRACKBAR_THRESHOLD, WIN_CONTROLS, 0, 255, lambda v: None)
            cv2.createTrackbar(
                TRACKBAR_SMOOTHING, WIN_CONTROLS,
                int(round(self.settings.smoothing_px * 10)), 100, lambda v: None,
            )
            cv2.resizeWindow(WIN_CONTROLS, 500, 100)

        cv2.setMouseCallback(mouse_window, on_mouse, self.interaction)

    def show(self, views: Views):
        result_height, result_width = views.result.shape[:2]

        if self.args.display == "dashboard":
            dashboard, frame_view, controls, progress, sliders = create_dashboard(
                views, self.args.dashboard_width, self.args.dashboard_height,
                paused=self.paused,
                frame_index=self.current_frame_index,
                total_frames=self.total_frames,
                fps=self.source_fps,
                slider_values=self.interaction.slider_values,
                has_video=self.args.video is not None,
            )
            self.interaction.window_fit = show_fit(WIN_DASHBOARD, dashboard)
            self.interaction.frame_view = frame_view
            self.interaction.control_regions = controls
            self.interaction.progress_region = progress
            self.interaction.slider_regions = sliders
            return

        self.interaction.window_fit = show_fit(WIN_RESULT, views.result)
        self.interaction.frame_view = FrameView(0, 0, 1.0, result_width, result_height)

        show_fit(WIN_INFO, views.info)
        show_fit(WIN_RECTIFICATION, views.rectification)
        show_fit(WIN_MASK, views.mask)
        cv2.imshow(WIN_CURVATURE, views.curvature_graph)
        cv2.imshow(WIN_RADIUS, views.radius_graph)

    # ---------- reproducao ----------

    def read_frame_at(self, index):
        if self.capture is None or self.total_frames <= 0:
            return None
        index = int(min(max(index, 0), self.total_frames - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self.capture.read()
        if not success or frame is None:
            return None
        self.current_frame_index = index
        self.ended = False
        return frame

    def apply_dashboard_action(self, frame):
        """Executa o clique pendente (botoes, barra de progresso). Retorna o frame a exibir."""

        action = self.interaction.action
        self.interaction.action = None
        if action is None:
            return frame
        if action == "toggle":
            self.paused = not self.paused
            return frame
        if self.args.video is None:
            return frame

        step_second = max(1, int(round(self.source_fps)))
        if action == "prev":
            target = self.current_frame_index - 1
        elif action == "next":
            target = self.current_frame_index + 1
        elif action == "back_1s":
            target = self.current_frame_index - step_second
        elif action == "forward_1s":
            target = self.current_frame_index + step_second
        elif action == "seek" and self.interaction.seek_fraction is not None:
            target = int(round(self.interaction.seek_fraction * max(self.total_frames - 1, 0)))
        else:
            return frame

        new_frame = self.read_frame_at(target)
        if new_frame is not None:
            self.paused = True
            return new_frame
        return frame

    # ---------- teclado ----------

    def handle_key(self, key) -> bool:
        """Retorna True para encerrar."""

        if key in (ord("q"), 27):
            return True

        if key == ord(" "):
            self.paused = not self.paused
        elif key == ord("c"):
            self.restart_calibration()
        elif key == ord("s"):
            self.save_snapshot()

        return False

    def save_snapshot(self):
        views = self.last_views

        if views is None or views.centerline is None:
            print("[AVISO] Nada para salvar: objeto nao detectado.")
            return

        save_snapshot(self.args.output, views.result, views.centerline, self.settings)

    # ---------- loop ----------

    def run(self):
        capture = open_capture(self.args)

        # O dashboard depende destes tres valores (barra de progresso, botoes
        # de frame/tempo e a busca por instante): sem eles os controles ficam mortos.
        self.capture = capture
        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self.args.video else 0

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.source_fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        frame_duration_seconds = 1.0 / self.source_fps

        print(f"FPS original: {self.source_fps:.2f}")
        if self.args.video:
            print(f"Frames no video: {self.total_frames}")

        try:
            success, frame = capture.read()

            if not success or frame is None:
                raise RuntimeError("A fonte abriu, mas nao entregou o primeiro frame.")

            print(f"[OK] Primeiro frame: {frame.shape}")
            print(f"Resolucao original: {frame.shape[1]}x{frame.shape[0]}")
            print(f"Unidade dos resultados: {self.settings.unit}")
            print(__doc__.split("Controles:")[1].split("Exemplos:")[0])

            self.create_windows()

            if self.needs_calibration:
                print("Calibracao ArUco: mostre os 4 marcadores para a camera.")

            self.current_frame_index = 0

            while True:
                loop_start = time.perf_counter()

                self.last_views = self.build_views(frame)
                self.show(self.last_views)

                if self.current_frame_index % 100 == 0 and not self.paused:
                    print(f"Frame {self.current_frame_index}", flush=True)

                processing_seconds = time.perf_counter() - loop_start

                if self.args.video is not None and not self.paused:
                    remaining_seconds = frame_duration_seconds - processing_seconds
                    wait_ms = max(1, int(round(remaining_seconds * 1000.0)))
                else:
                    wait_ms = 20

                if self.handle_key(cv2.waitKey(wait_ms) & 0xFF):
                    break

                frame = self.apply_dashboard_action(frame)

                if self.paused:
                    continue

                if self.ended:
                    # PLAY depois do fim do video: recomeca do inicio
                    rewound = self.read_frame_at(0)
                    if rewound is not None:
                        frame = rewound
                    continue

                success, next_frame = capture.read()

                if not success or next_frame is None:
                    if self.args.video is not None:
                        # Fica no ultimo frame para os controles continuarem uteis
                        print("[INFO] Fim do video. PLAY recomeca; Q sai.", flush=True)
                        self.ended = True
                        self.paused = True
                        continue
                    print("[INFO] Falha ao capturar frame.", flush=True)
                    break

                frame = next_frame
                self.current_frame_index += 1
        finally:
            capture.release()
            cv2.destroyAllWindows()

        print("\nPrograma encerrado.", flush=True)


# ============================================================
# 8. ENTRADA DE VIDEO E LINHA DE COMANDO
# ============================================================

def open_capture(args):
    if args.video is not None:
        print(f"Abrindo video: {args.video}", flush=True)
        capture = cv2.VideoCapture(args.video)

        if not capture.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir o video: {args.video}")

        return capture

    print(f"Abrindo webcam {args.camera}...", flush=True)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not capture.isOpened():
        print("[AVISO] DirectShow falhou. Tentando backend padrao...", flush=True)
        capture.release()
        capture = cv2.VideoCapture(args.camera)

    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a webcam {args.camera}.")

    return capture


def build_parser():
    default = Settings()

    parser = argparse.ArgumentParser(
        description="Medicao de curvatura 2D por visao computacional",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_argument_group("entrada")
    source.add_argument("--video", type=str, default=None,
        help="Arquivo de video. Se omitido, usa a webcam.")
    source.add_argument("--camera", type=int, default=0,
        help="Indice da webcam.")

    analysis = parser.add_argument_group("analise")
    analysis.add_argument("--smoothing", type=float, default=default.smoothing_px,
        help="Suavizacao inicial da spline, em pixels.")
    analysis.add_argument("--mm-per-pixel", type=float, default=None,
        help="Escala FIXA em mm/pixel (sem correcao de perspectiva). "
             "Sem ela e sem ArUco, os resultados saem em pixels.")
    analysis.add_argument("--output", type=str, default="snapshots",
        help="Pasta dos snapshots.")

    aruco = parser.add_argument_group("calibracao ArUco")
    aruco.add_argument("--aruco-size", type=float, default=None, metavar="MM",
        help="Lado do marcador ArUco (quadrado preto externo), em mm.")
    aruco.add_argument("--aruco-spacing", type=float, nargs="+", default=None, metavar="MM",
        help="Distancia CENTRO A CENTRO entre marcadores vizinhos, em mm. "
             "Um valor = retangulo quadrado; dois valores = horizontal vertical.")
    aruco.add_argument("--aruco-ids", type=int, nargs=4, default=[0, 1, 2, 3], metavar="ID",
        help="Ids dos marcadores: sup-esq sup-dir inf-dir inf-esq.")
    aruco.add_argument("--aruco-dict", type=str, default="DICT_5X5_250",
        help="Dicionario ArUco do OpenCV.")
    aruco.add_argument("--calibration-px-per-mm", type=float, default=None,
        help="Resolucao da imagem retificada, em pixels/mm. Padrao: automatico "
             "(resolucao nativa da camera). Valores menores aceleram, mas "
             "perdem detalhe e pioram o ajuste do cabo.")
    aruco.add_argument("--focal-px", type=float, default=None, metavar="PX",
        help="Distancia focal da camera em pixels (so para a pose no painel POSE). "
             "Padrao: 0.8 x largura da imagem.")
    aruco.add_argument("--calibration-frames", type=int, default=5,
        help="Frames consecutivos com os 4 marcadores para concluir a calibracao.")
    aruco.add_argument("--calibration-file", type=str, default="calibration_aruco.json",
        help="JSON onde a calibracao e salva/carregada.")
    aruco.add_argument("--load-calibration", action="store_true",
        help="Carrega --calibration-file em vez de detectar os marcadores.")
    aruco.add_argument("--aruco-margin", type=float, default=0.0, metavar="MM",
        help=("Margem adicional ao redor dos marcadores na imagem retificada, em mm."))
    
    display = parser.add_argument_group("exibicao")
    display.add_argument("--display", choices=("separate", "dashboard"), default="dashboard",
        help="Modo visual.")
    display.add_argument("--dashboard-width", type=int, default=1920,
        help="Largura do dashboard.")
    display.add_argument("--dashboard-height", type=int, default=1080,
        help="Altura do dashboard.")

    return parser


def build_layout(args, parser) -> Optional[ArucoLayout]:
    """Valida os argumentos de ArUco. Retorna None se nao foram pedidos."""

    if args.aruco_size is None and args.aruco_spacing is None:
        return None

    if args.aruco_size is None or args.aruco_spacing is None:
        parser.error("--aruco-size e --aruco-spacing devem ser usados juntos.")

    if len(args.aruco_spacing) > 2:
        parser.error("--aruco-spacing aceita 1 valor (quadrado) ou 2 (horizontal vertical).")

    spacing_x = args.aruco_spacing[0]
    spacing_y = args.aruco_spacing[-1]

    if args.aruco_size <= 0 or spacing_x <= 0 or spacing_y <= 0:
        parser.error("As medidas dos ArUco devem ser positivas.")

    if min(spacing_x, spacing_y) <= args.aruco_size:
        parser.error(
            "--aruco-spacing e a distancia CENTRO A CENTRO e deve ser maior "
            "que --aruco-size (senao os marcadores se sobrepoem)."
        )
    
    if args.aruco_margin < 0:
        parser.error("--aruco-margin nao pode ser negativa.")
    
    if len(set(args.aruco_ids)) != 4:
        parser.error("--aruco-ids exige 4 ids diferentes.")

    if args.calibration_px_per_mm is not None and args.calibration_px_per_mm <= 0:
        parser.error("--calibration-px-per-mm deve ser positivo.")
    if args.focal_px is not None and args.focal_px <= 0:
        parser.error("--focal-px deve ser positivo.")
    if args.calibration_frames <= 0:
        parser.error("--calibration-frames deve ser positivo.")

    return ArucoLayout(
        marker_size_mm=args.aruco_size,
        spacing_x_mm=spacing_x,
        spacing_y_mm=spacing_y,
        ids=tuple(args.aruco_ids),
        dictionary=args.aruco_dict,
        pixels_per_mm=args.calibration_px_per_mm or 0.0,
        margin_mm=args.aruco_margin,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    layout = build_layout(args, parser)

    print("\n======================================")
    print(" CURVATURA 2D - PROVA DE CONCEITO ")
    print("======================================\n")

    try:
        CurvatureApp(args, layout).run()
    except (RuntimeError, FileNotFoundError) as error:
        print(f"[ERRO] {error}")
        return 1

    return 0


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    sys.exit(main())
