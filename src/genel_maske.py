# -*- coding: utf-8 -*-
"""Kaynağı bilinmeyen (ham) bir fotoğraf için ön işleme: zemin türünü tahmin et, yaprağı maskele,
zemini siyaha boya, yaprağa kırp, 256x256 yap. `tahmin.py` ve web verisi (`08_web_veri.py`) kullanır.

Zemin türü, görüntünün dış çerçevesindeki piksellerden kestirilir:
  * beyaz  : çerçeve parlak ve renksiz  -> 01_on_isleme.maske_beyaz_zemin (kaynak B yöntemi)
  * siyah  : çerçeve koyu               -> 01_on_isleme.maske_siyah_zemin (kaynak A yöntemi, GrabCut)
  * genel  : diğer her şey (tarla, el, masa, kalibrasyon kartı...) -> maske_genel (yeni yöntem)
`maske_genel`: yeşil-sarı-kahverengi ve yeterince renkli pikseller "olası yaprak", görüntü merkezine
yakın olanlar "kesin yaprak" çekirdeği, dış çerçeve "kesin zemin" olacak şekilde GrabCut; ardından
morfoloji + en büyük bileşen + küçük delik doldurma.
Maskeleme başarısız olursa (boş/çok küçük maske ya da istisna) görüntünün tamamı kare olarak
kullanılır (maske = her yer yaprak) ve `basarili=False` döner; çağıran yine tahmin yapabilir.
"""
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ortak import KAYDET_BOYUT  # noqa: E402

_on = importlib.import_module("01_on_isleme")

MIN_ALAN_ORANI = 0.02  # maske bu oranın altında kalırsa "başarısız" sayılır


def zemin_turu(rgb: np.ndarray) -> str:
    """Dış çerçevedeki (%4) piksellerin medyan parlaklık/doygunluğuna göre 'beyaz' | 'siyah' | 'genel'."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cer = _on.kenar_cercevesi(rgb.shape[:2], 0.04) == 1
    s, v = np.median(hsv[..., 1][cer]), np.median(hsv[..., 2][cer])
    if v > 200 and s < 30:
        return "beyaz"
    if v < 70:
        return "siyah"
    return "genel"


def maske_genel(rgb: np.ndarray) -> np.ndarray:
    """Bilinmeyen zemin için renk tohumlu + merkez öncüllü GrabCut (512 px'te hesaplanır)."""
    H, W = rgb.shape[:2]
    s = min(1.0, 512 / max(H, W))
    k = cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    h, w = k.shape[:2]
    hsv = cv2.cvtColor(k, cv2.COLOR_RGB2HSV)
    ton, doy, par = [hsv[..., i].astype(int) for i in range(3)]
    chroma, _ = _on.renk_doygunlugu_lab(k)
    # Yaprak renkleri: yeşil-sarı (H 15-95) ya da kahverengi/nekrotik (H < 15 veya > 160, orta parlaklık)
    renkli = (chroma > 12) & (doy > 40) & (par > 35)
    yaprak_rengi = renkli & (((ton >= 15) & (ton <= 95)) | (((ton < 15) | (ton > 160)) & (par < 200)))
    yy, xx = np.mgrid[0:h, 0:w]
    merkez = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2 < 0.35 ** 2
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[yaprak_rengi] = cv2.GC_PR_FGD
    cekirdek = cv2.erode(_on.morfoloji(yaprak_rengi & merkez, 5), np.ones((5, 5), np.uint8))
    gc[cekirdek == 1] = cv2.GC_FGD
    gc[_on.kenar_cercevesi((h, w), 0.02) == 1] = cv2.GC_BGD
    if not ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).any():
        return np.zeros((H, W), np.uint8)
    if (gc == cv2.GC_FGD).sum() < 50:  # merkezde renkli yaprak yoksa merkez dairesini olası yaprak say
        gc[merkez & (gc != cv2.GC_BGD)] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(k, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        m = yaprak_rengi.astype(np.uint8)
    m = _on.morfoloji(m, 5)
    m = _on.en_buyuk_bilesen(m)
    m = _on.delik_doldur(m, _on.KUCUK_DELIK_ORANI * m.sum())
    return (cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8)


def maske_hesapla(rgb: np.ndarray, zemin: str = "otomatik"):
    """(maske 0/1, kullanılan_zemin) döndürür. zemin: otomatik|beyaz|siyah|kart|genel."""
    if zemin == "otomatik":
        zemin = zemin_turu(rgb)
    f = {"beyaz": _on.maske_beyaz_zemin, "siyah": _on.maske_siyah_zemin,
         "kart": _on.maske_kart_zemin, "genel": maske_genel}[zemin]
    return f(rgb).astype(np.uint8), zemin


def tam_goruntu_kare(rgb: np.ndarray):
    """Maskeleme başarısızsa: görüntüyü siyahla kareye tamamla, 256'ya küçült; maske = görüntünün kendisi."""
    H, W = rgb.shape[:2]
    m = np.ones((H, W), np.uint8)
    g, gm = _on.kirp_ve_boyutlandir(rgb, m)
    return g, gm


def on_isle(rgb: np.ndarray, zemin: str = "otomatik"):
    """Ham RGB (uint8, HxWx3) -> (goruntu 256x256x3 uint8 zemin siyah, maske 256x256 uint8 0/255, bilgi dict).

    bilgi = {"zemin": kullanılan yöntem, "basarili": bool, "alan_orani": maske alanı / görüntü}.
    Hiçbir durumda istisna fırlatmaz (maskeleme hatasında tüm görüntüye geri döner)."""
    kullanilan, boyut_tamam = "hata", True
    try:
        H, W = rgb.shape[:2]
        s = _on.MAKS_KENAR / max(H, W)
        if s < 1:  # aşırı en-boy oranında bir kenar 0'a yuvarlanmasın
            rgb = cv2.resize(rgb, (max(1, int(round(W * s))), max(1, int(round(H * s)))),
                             interpolation=cv2.INTER_AREA)
    except Exception as e:  # noqa: BLE001 - boyutlandırma hatasında mevcut "başarısız" yoluna git
        kullanilan, boyut_tamam = f"hata:{type(e).__name__}", False
    try:
        if not boyut_tamam:
            raise RuntimeError("boyutlandırma başarısız")
        m, kullanilan_z = maske_hesapla(rgb, zemin)
        kullanilan = kullanilan_z
        oran = float(m.mean())
        if oran >= MIN_ALAN_ORANI and int(m.sum()) >= _on.MIN_YAPRAK_PIKSEL:
            g, gm = _on.kirp_ve_boyutlandir(rgb, m)
            if g is not None:
                return g, gm, {"zemin": kullanilan, "basarili": True, "alan_orani": round(oran, 4)}
    except Exception as e:  # noqa: BLE001 - tahmin her koşulda çalışmalı
        if boyut_tamam:
            kullanilan = f"hata:{type(e).__name__}"
    try:
        g, gm = tam_goruntu_kare(rgb)
    except Exception:  # noqa: BLE001 - son çare: boş kare (istisna fırlatma)
        g, gm = np.zeros((KAYDET_BOYUT, KAYDET_BOYUT, 3), np.uint8), np.full((KAYDET_BOYUT, KAYDET_BOYUT), 255, np.uint8)
    return g, gm, {"zemin": kullanilan, "basarili": False, "alan_orani": 1.0}


__all__ = ["zemin_turu", "maske_genel", "maske_hesapla", "on_isle", "KAYDET_BOYUT"]
