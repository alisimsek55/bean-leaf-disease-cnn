# -*- coding: utf-8 -*-
"""Telefon fotoğrafında yaprağı zeminden ayırma (zemin tipi bilinmiyor).

NOT: Uygulamanın ana akışı tek hat olarak `src/genel_maske.on_isle` kullanır (model onunla
değerlendirildi); bu modülden yalnız `kucult` kullanılıyor. Diğer fonksiyonlar karşılaştırma/deneme içindir.

Strateji (mevcut ön işleme kodu src/01_on_isleme.py yeniden kullanılır):
  1) Görüntünün kenar şeridine bakarak zemini tahmin et: açık/renksiz -> beyaz zemin yöntemi,
     koyu -> siyah zemin yöntemi (GrabCut'lı), diğer -> genel yöntem (kenar rengine uzaklık + GrabCut).
  2) Sonucu doğrula (alan oranı makul mu?). Olmazsa genel yöntemi dene.
  3) Hâlâ olmazsa tüm görüntüyle devam et ve uyarı döndür.
"""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parent.parent
SRC = KOK / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _on_isleme():
    ad = "_on_isleme_01"
    if ad in sys.modules:
        return sys.modules[ad]
    spec = importlib.util.spec_from_file_location(ad, SRC / "01_on_isleme.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ad] = mod
    spec.loader.exec_module(mod)
    return mod


MIN_ALAN, MAKS_ALAN = 0.04, 0.92


def kucult(rgb: np.ndarray, maks_kenar: int = 1024) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Boş görüntü.")
    s = maks_kenar / max(h, w)
    if s < 1:  # aşırı en-boy oranında kısa kenar 0'a yuvarlanmasın (OpenCV assertion)
        boyut = (max(1, int(round(w * s))), max(1, int(round(h * s))))
        try:
            rgb = cv2.resize(rgb, boyut, interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            raise ValueError(f"Görüntü küçültülemedi ({w}x{h}).") from e
    return rgb


def _kenar_serit(rgb, oran=0.04):
    k = max(3, int(oran * max(rgb.shape[:2])))
    return np.concatenate([rgb[:k].reshape(-1, 3), rgb[-k:].reshape(-1, 3),
                           rgb[:, :k].reshape(-1, 3), rgb[:, -k:].reshape(-1, 3)])


def maske_genel(rgb: np.ndarray) -> np.ndarray:
    """Zemin rengi kenar şeridinden öğrenilir; ona uzak pikseller yaprak adayıdır, GrabCut inceltir."""
    H, W = rgb.shape[:2]
    s = 512 / max(H, W)
    k = cv2.resize(rgb, (max(int(W * s), 8), max(int(H * s), 8)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(k, (7, 7), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    ser = _kenar_serit(lab.astype(np.float32))
    zemin = np.median(ser, axis=0)
    # kenar şeridinin kendi yayılımı: zemin dokusu/gölgesi için tolerans
    yayilim = np.linalg.norm(ser - zemin, axis=1)
    tol = float(np.percentile(yayilim, 90))
    uzak = np.linalg.norm(lab - zemin, axis=2)
    uzak8 = np.clip(uzak / max(uzak.max(), 1) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(uzak8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    esik = max(t / 255 * max(uzak.max(), 1), 1.3 * tol, 12.0)
    aday = (uzak > esik).astype(np.uint8)
    hsv = cv2.cvtColor(k, cv2.COLOR_RGB2HSV)
    yesil = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)).astype(np.uint8)
    aday = ((aday > 0) | (yesil > 0)).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    aday = cv2.morphologyEx(cv2.morphologyEx(aday, cv2.MORPH_OPEN, ker), cv2.MORPH_CLOSE, ker)
    cek = cv2.erode(aday, np.ones((9, 9), np.uint8))
    gc = np.full(aday.shape, cv2.GC_PR_BGD, np.uint8)
    gc[aday == 1] = cv2.GC_PR_FGD
    gc[(cek == 1) & ((yesil == 1) | (uzak > 1.8 * esik))] = cv2.GC_FGD
    f = max(2, int(0.02 * max(aday.shape)))
    gc[:f, :] = gc[-f:, :] = gc[:, :f] = gc[:, -f:] = cv2.GC_BGD
    if not ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).any():
        return np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(k, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        m = aday
    op = _on_isleme()
    m = op.morfoloji(m, 5)
    m = op.en_buyuk_bilesen(m)
    m = op.delik_doldur(m, None)
    return (cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8)


def _gecerli(m: np.ndarray) -> bool:
    o = float((m > 0).mean())
    return MIN_ALAN <= o <= MAKS_ALAN


def yaprak_ayir(rgb: np.ndarray):
    """rgb (HxWx3 uint8, <=1024 px önerilir) -> (maske 0/1, yöntem adı, uyarı metni veya None)."""
    op = _on_isleme()
    ser = _kenar_serit(rgb)
    hsv = cv2.cvtColor(ser.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    v, s = float(np.median(hsv[:, 2])), float(np.median(hsv[:, 1]))
    denemeler = []
    if v > 200 and s < 40:
        denemeler.append(("beyaz zemin", op.maske_beyaz_zemin))
    elif v < 70:
        denemeler.append(("koyu zemin", op.maske_siyah_zemin))
    denemeler.append(("genel (kenar rengi + GrabCut)", maske_genel))
    for ad, f in denemeler:
        try:
            m = f(rgb)
        except Exception:  # bir yöntem çökerse sıradakine geç
            continue
        if _gecerli(m):
            uyari = None
            if float((m > 0).mean()) > 0.75:
                uyari = "Yaprak görüntünün büyük kısmını kaplıyor; ayırma doğru olmayabilir."
            return m.astype(np.uint8), ad, uyari
    tam = np.ones(rgb.shape[:2], np.uint8)
    return tam, "ayırma başarısız (tüm görüntü)", (
        "Yaprak zeminden ayrılamadı; tüm fotoğraf kullanıldı. Sonuç güvenilir olmayabilir. "
        "Yaprağı sade (beyaz/koyu) bir zemine koyup tekrar deneyin.")


def kirp(rgb: np.ndarray, m: np.ndarray):
    """Arka planı siyaha boyar, kareye kırpar, 256x256 yapar (eğitimdeki ön işlemeyle aynı)."""
    op = _on_isleme()
    g, gm = op.kirp_ve_boyutlandir(rgb, m)
    return g, gm
