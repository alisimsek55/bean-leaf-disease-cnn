# -*- coding: utf-8 -*-
"""Kural tabanlı şiddet düzeyi çekirdeği (içe aktarılabilir): lezyon_maskesi, lezyon_orani, duzey,
siddet_duzeyi. Ayrıntılı yöntem açıklaması src/06_siddet_tahmini.py başındadır."""
import cv2
import numpy as np

A_KIRMIZI = 4             # a* - 128 bunun üstündeyse kırmızımsı/kahverengi
H_SARI = 27               # H < 27 (≈ 54°) sarı-turuncu-kahve tonları
S_RENKLI = 60             # sarılık ölçütü için en az doygunluk (gri pikseller tonsuzdur)
V_KOYU = 45               # bunun altı koyu nekroz
DUZEYLER = ["sağlıklı", "az hastalıklı", "çok hastalıklı"]
BELIRSIZ = "belirsiz"     # değerlendirilebilir yaprak pikseli yok/çok az: "sağlıklı" denmez (DUZEYLER'de değil)
MIN_YAPRAK_PIKSEL = 50    # aşındırma sonrası bundan az yaprak pikseli varsa düzey belirsizdir
T1_VARSAYILAN = 0.005     # docs/RESULTS.tr.md §7 (06 betiği veriden aynı değeri türetir)
T2_VARSAYILAN = 0.13


def lezyon_maskesi(rgb: np.ndarray, maske: np.ndarray):
    """rgb: HxWx3 uint8, maske: HxW (0/255). Döndürür: (lezyon ikili maske, değerlendirilen yaprak maskesi)."""
    yap = cv2.erode((maske > 127).astype(np.uint8), np.ones((5, 5), np.uint8))
    rgb = cv2.medianBlur(rgb, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1].astype(int) - 128
    lez = ((a > A_KIRMIZI) | ((h < H_SARI) & (s > S_RENKLI)) | (v < V_KOYU)).astype(np.uint8)
    lez = cv2.morphologyEx(lez, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) & yap
    return lez, yap


def lezyon_orani(rgb, maske) -> float:
    """Lezyon pikseli / değerlendirilen yaprak pikseli. Değerlendirilen yaprak boşsa 0.0 döner (eski davranış
    korunur); bu 0.0 "sağlıklı" anlamına gelmez, düzey için siddet_duzeyi kullanılmalıdır."""
    lez, yap = lezyon_maskesi(rgb, maske)
    return float(lez.sum() / max(yap.sum(), 1))


def duzey(oran, t1, t2):
    return DUZEYLER[0] if oran < t1 else (DUZEYLER[1] if oran < t2 else DUZEYLER[2])


def siddet_duzeyi(rgb, maske, t1: float = T1_VARSAYILAN, t2: float = T2_VARSAYILAN):
    """Tek görüntü için (lezyon_orani, düzey) döndürür. rgb: HxWx3 uint8 (PIL/numpy), maske: HxW (0/255).
    Girdi 256x256 ön işlenmiş görüntü/maske ile eşik ayarlıdır; PIL görüntüsü de verilebilir.
    Değerlendirilebilir yaprak pikseli < MIN_YAPRAK_PIKSEL ise (0.0, BELIRSIZ) döner."""
    rgb = np.asarray(rgb.convert("RGB") if hasattr(rgb, "convert") else rgb, dtype=np.uint8)
    maske = np.asarray(maske.convert("L") if hasattr(maske, "convert") else maske)
    lez, yap = lezyon_maskesi(rgb, maske)
    if int(yap.sum()) < MIN_YAPRAK_PIKSEL:  # boş/aşınma sonrası boşalan maske: yanlış "sağlıklı" verme
        return 0.0, BELIRSIZ
    oran = float(lez.sum() / yap.sum())
    return oran, duzey(oran, t1, t2)
