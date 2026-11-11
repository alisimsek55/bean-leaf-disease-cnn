# -*- coding: utf-8 -*-
"""
01 - ÖN İŞLEME: yaprağı arka plandan ayırma, arka planı siyaha çevirme, kırpma, boyutlandırma.

Neden? Her sınıf büyük ölçüde farklı bir kaynaktan ve farklı bir zeminden geliyor
(hale = beyaz zemin PNG, CBB = siyah kumaş JPEG, sağlıklı = ikisi karışık). Model zemini
öğrenmesin diye tüm görüntülerde yaprak dışındaki her şey siyaha boyanır ve her görüntü
aynı biçimde (256x256 PNG, yaprak sınır kutusuna kırpılmış, kare olacak şekilde siyahla
doldurulmuş) kaydedilir.

Zemin türüne göre maske yöntemi (kaynak bilgisi etiketler.csv'den gelir):
  * Kaynak B (beyaz zemin): "beyaz olmayan" pikseller (S > 25 veya V < 225) yapraktır.
    RGBA dosyalar önce beyaz zemine yapıştırılır.
  * Kaynak A (siyah/lacivert, bazen gri kıvrımlı kumaş): parlaklık Otsu eşiği + renk
    doygunluğu çekirdeği ile başlatılan GrabCut (ayrıntı: maske_siyah_zemin).
  * Kaynak C (renkli kalibrasyon kartı): yeşil-sarı ton (HSV) çekirdek maskesi + GrabCut.
Hepsinde ortak: morfolojik açma/kapama, en büyük bağlı bileşen, delik doldurma.
  - A'da tüm delikler doldurulur (deliklerden görünen siyah kumaş zaten siyah = arka planla aynı).
  - B ve C'de yalnızca küçük delikler (< yaprak alanının %1'i) doldurulur; böylece böcek
    delikleri gibi gerçek boşluklardan beyaz zemin/kart görünmez, siyah kalır.

Çıktılar:
  data/islenmis/goruntu/<sinif>/<ad>.png   (256x256 RGB, arka plan siyah)
  data/islenmis/maske/<sinif>/<ad>.png     (256x256, 255 = yaprak)
  data/islenmis/on_isleme_raporu.csv       (dosya başına maske istatistikleri ve kalite bayrakları)
Kullanım:  python src/01_on_isleme.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ortak import HAM, ISLENMIS, SINIFLAR, VERI, KAYDET_BOYUT, tohum_ayarla  # noqa: E402

MAKS_KENAR = 1024          # hız için maske hesaplamadan önce uzun kenar bu boyuta küçültülür
KIRPMA_PAYI = 0.04         # sınır kutusuna eklenen pay (kutu kenarının oranı)
KUCUK_DELIK_ORANI = 0.01   # B/C'de doldurulacak deliklerin yaprak alanına göre üst sınırı


def goruntu_oku(yol: Path) -> np.ndarray:
    """Görüntüyü RGB numpy dizisi olarak okur; RGBA ise beyaz zemine yapıştırır."""
    im = Image.open(yol)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        zemin = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(zemin, im)
    im = im.convert("RGB")
    a = np.asarray(im)
    h, w = a.shape[:2]
    s = MAKS_KENAR / max(h, w)
    if s < 1:
        a = cv2.resize(a, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return a


def en_buyuk_bilesen(m: np.ndarray) -> np.ndarray:
    """İkili maskedeki en büyük bağlı bileşeni döndürür."""
    n, et, ist, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m.astype(np.uint8)
    i = 1 + int(np.argmax(ist[1:, cv2.CC_STAT_AREA]))
    return (et == i).astype(np.uint8)


def delik_doldur(m: np.ndarray, maks_alan=None) -> np.ndarray:
    """Maskedeki delikleri doldurur. maks_alan verilirse yalnız ondan küçük delikler."""
    m = m.astype(np.uint8)
    ters = 1 - m
    n, et, ist, _ = cv2.connectedComponentsWithStats(ters, connectivity=4)
    h, w = m.shape
    sonuc = m.copy()
    for i in range(1, n):
        x, y, bw, bh, alan = ist[i]
        kenara_degiyor = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if kenara_degiyor:
            continue  # dış arka plan
        if maks_alan is None or alan < maks_alan:
            sonuc[et == i] = 1
    return sonuc


def morfoloji(m: np.ndarray, k: int) -> np.ndarray:
    """Açma (gürültü temizleme) + kapama (küçük boşlukları kapatma)."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    return m


def maske_beyaz_zemin(rgb):
    """Kaynak B: beyaz zemin. Doygunluğu düşük ve çok parlak pikseller zemindir."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = ((s > 25) | (v < 225)).astype(np.uint8)
    m = morfoloji(m, 5)
    m = en_buyuk_bilesen(m)
    return delik_doldur(m, KUCUK_DELIK_ORANI * m.sum())


def renk_doygunlugu_lab(rgb):
    """CIELab renk doygunluğu (chroma) = sqrt(a^2 + b^2); gri/siyah/beyaz yüzeylerde ~0."""
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2), lab[..., 0]


def kenar_cercevesi(shape, oran=0.02):
    f = np.zeros(shape, np.uint8)
    k = max(2, int(oran * max(shape)))
    f[:k, :] = f[-k:, :] = f[:, :k] = f[:, -k:] = 1
    return f


def maske_siyah_zemin(rgb):
    """Kaynak A: koyu (siyah/lacivert) kumaş; bazı karelerde ışık alan gri kumaş kıvrımları var.

    Görsel denetimde yalnız parlaklık (V, Otsu) eşiğinin gri kumaş kıvrımlarını yaprak sandığı,
    yalnız renk doygunluğu (Lab chroma) eşiğinin ise soluk/grimsi alt yüzlü yaprakları kaçırdığı
    görüldü. Bu yüzden iki aşamalı yöntem:
      1) Parlaklık (V) Otsu eşiği (40..120'ye kısılmış) -> olası yaprak.
      2) GrabCut (512 px'te, 5 tur): görüntü çerçevesi ve çok koyu pikseller kesin zemin,
         'parlak VE renkli (chroma>10)' pikseller kesin yaprak çekirdeği, kalan parlak pikseller
         olası yaprak. GrabCut renk modelini görüntüye göre öğrenir; gri kıvrımlar zemin modeline düşer.
      3) En büyük bileşen + delik doldurma (delikten görünen kumaş zaten koyu = yeni arka plan).
    """
    H, W = rgb.shape[:2]
    s = 512 / max(H, W)
    kucuk = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(kucuk, cv2.COLOR_RGB2HSV)
    v = cv2.GaussianBlur(hsv[..., 2], (5, 5), 0)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(np.clip(t, 40, 120))
    chroma, _ = renk_doygunlugu_lab(kucuk)
    parlak = morfoloji((v > t).astype(np.uint8), 5)
    cekirdek = cv2.erode(morfoloji((parlak > 0) & (chroma > 10), 5), np.ones((7, 7), np.uint8))
    gc = np.full(v.shape, cv2.GC_PR_BGD, np.uint8)
    gc[parlak == 1] = cv2.GC_PR_FGD
    gc[cekirdek == 1] = cv2.GC_FGD
    gc[(v < 0.5 * t) | (kenar_cercevesi(v.shape) == 1)] = cv2.GC_BGD
    if (gc == cv2.GC_FGD).sum() < 50:  # renkli çekirdek yoksa yalnız parlaklıkla
        gc[parlak == 1] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(kucuk, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        m = parlak
    m = morfoloji(m, 5)
    m = en_buyuk_bilesen(m)
    m = delik_doldur(m, None)
    return (cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8)


def maske_kart_zemin(rgb):
    """Kaynak C: renkli kalibrasyon kartı. Yeşil-sarı ton çekirdeği + GrabCut ile iyileştirme."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    cekirdek = ((h >= 22) & (h <= 90) & (s > 60) & (v > 40)).astype(np.uint8)
    cekirdek = morfoloji(cekirdek, 9)
    cekirdek = en_buyuk_bilesen(cekirdek)
    cekirdek = delik_doldur(cekirdek, None)
    gc = np.full(cekirdek.shape, cv2.GC_BGD, np.uint8)
    genis = cv2.dilate(cekirdek, np.ones((41, 41), np.uint8))
    dar = cv2.erode(cekirdek, np.ones((15, 15), np.uint8))
    gc[genis == 1] = cv2.GC_PR_BGD
    gc[cekirdek == 1] = cv2.GC_PR_FGD
    gc[dar == 1] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    # Yaprakçıklar arasında kalan beyaz/gri kart karelerini çıkar (renksiz ve parlak pikseller)
    chroma, L = renk_doygunlugu_lab(rgb)
    m[(chroma < 8) & (L > 90)] = 0
    m = morfoloji(m, 7)
    m = en_buyuk_bilesen(m)
    return delik_doldur(m, KUCUK_DELIK_ORANI * m.sum())


def kirp_ve_boyutlandir(rgb, m):
    """Arka planı siyaha boyar, sınır kutusuna (pay ile) kırpar, kare yapar, KAYDET_BOYUT'a getirir."""
    rgb = rgb.copy()
    rgb[m == 0] = 0
    ys, xs = np.where(m > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bh, bw = y1 - y0, x1 - x0
    kenar = int(max(bh, bw) * (1 + 2 * KIRPMA_PAYI))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    kare = np.zeros((kenar, kenar, 3), np.uint8)
    kare_m = np.zeros((kenar, kenar), np.uint8)
    # kaynak dikdörtgeni kareye kopyala (kare görüntü sınırlarını aşabilir; aşan kısım siyah kalır)
    sy0, sx0 = cy - kenar // 2, cx - kenar // 2
    H, W = m.shape
    ay0, ax0 = max(sy0, 0), max(sx0, 0)
    ay1, ax1 = min(sy0 + kenar, H), min(sx0 + kenar, W)
    kare[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0] = rgb[ay0:ay1, ax0:ax1]
    kare_m[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0] = m[ay0:ay1, ax0:ax1] * 255
    g = cv2.resize(kare, (KAYDET_BOYUT, KAYDET_BOYUT), interpolation=cv2.INTER_AREA)
    gm = cv2.resize(kare_m, (KAYDET_BOYUT, KAYDET_BOYUT), interpolation=cv2.INTER_NEAREST)
    g[gm == 0] = 0  # yeniden boyutlandırma sonrası kenarda kalan zemin karışımını temizle
    return g, gm


def main():
    tohum_ayarla()
    et = pd.read_csv(VERI / "etiketler.csv")
    et = et[(et["kume"] == "raw") & (et["sinif"].isin(SINIFLAR))].reset_index(drop=True)
    print(f"Toplam ham görüntü: {len(et)}")
    for s in SINIFLAR:
        (ISLENMIS / "goruntu" / s).mkdir(parents=True, exist_ok=True)
        (ISLENMIS / "maske" / s).mkdir(parents=True, exist_ok=True)

    kayitlar = []
    t0 = time.time()
    for i, r in et.iterrows():
        yol = VERI / r["dosya"]
        rgb = goruntu_oku(yol)
        if r["kaynak"] == "mendeley_3km9d246z2":
            yontem, m = "beyaz_zemin", maske_beyaz_zemin(rgb)
        elif r["kaynak"] == "mendeley_ykvcrjffzd":
            yontem, m = "siyah_zemin", maske_siyah_zemin(rgb)
        else:
            yontem, m = "kart_grabcut", maske_kart_zemin(rgb)

        H, W = m.shape
        alan_orani = float(m.mean())
        # Kalite göstergeleri: alan oranı, dışbükey örtüye oranı (solidity), kenara değme
        kont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(kont, key=cv2.contourArea)
        solidity = float(cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1))
        kenar_piksel = int(m[0, :].sum() + m[-1, :].sum() + m[:, 0].sum() + m[:, -1].sum())
        kenar_orani = kenar_piksel / (2 * (H + W))
        # Bayraklar: çok küçük/çok büyük maske, düşük solidity, kenara çok değme (zemin sızması ya da kesik yaprak)
        bayraklar = []
        if alan_orani < 0.05:
            bayraklar.append("kucuk_maske")
        if alan_orani > 0.85:
            bayraklar.append("buyuk_maske")
        if solidity < 0.80:
            bayraklar.append("dusuk_solidity")
        if kenar_orani > 0.10:
            bayraklar.append("kenara_degiyor")
        # Maske içi ortalama renk doygunluğu çok düşükse seçilen bölge büyük olasılıkla gri kumaştır
        chroma, _ = renk_doygunlugu_lab(rgb)
        ort_chroma = float(chroma[m > 0].mean())
        # Eşik 4: görsel denetimde chroma <= 3.1 olan 8 maskenin tamamı kumaştı; 4.9-6.4 arasındakiler
        # gerçek (grimsi alt yüzlü) yapraklardı. Arada kalan 1 hatalı görüntü elle dışlandı.
        if ort_chroma < 4:
            bayraklar.append("renksiz_maske")

        g, gm = kirp_ve_boyutlandir(rgb, m)
        ad = Path(r["dosya"]).stem + ".png"
        Image.fromarray(g).save(ISLENMIS / "goruntu" / r["sinif"] / ad)
        Image.fromarray(gm).save(ISLENMIS / "maske" / r["sinif"] / ad)
        kayitlar.append({
            "dosya": r["dosya"], "islenmis": f"goruntu/{r['sinif']}/{ad}", "maske": f"maske/{r['sinif']}/{ad}",
            "sinif": r["sinif"], "kaynak": r["kaynak"], "grup": r["grup"], "yontem": yontem,
            "alan_orani": round(alan_orani, 4), "solidity": round(solidity, 4),
            "kenar_orani": round(kenar_orani, 4), "ort_chroma": round(ort_chroma, 2), "bayraklar": ";".join(bayraklar),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(et)}  ({time.time() - t0:.0f} sn)", flush=True)

    rap = pd.DataFrame(kayitlar)
    # Elle yapılan görsel denetimde kötü maskelendiği belirlenen dosyalar (varsa) dışlanır
    dis = VERI / "islenmis_dislanan.csv"
    # Otomatik dışlama kuralı: renksiz_maske bayrağı (yaprak yerine kumaş seçilmiş)
    rap["dislandi"] = rap["bayraklar"].str.contains("renksiz_maske")
    if dis.exists():
        d = pd.read_csv(dis)
        rap.loc[rap["dosya"].isin(d["dosya"]), "dislandi"] = True  # elle denetimde kötü bulunanlar
    rap.to_csv(ISLENMIS / "on_isleme_raporu.csv", index=False)
    print(f"Bitti: {len(rap)} görüntü, {time.time() - t0:.0f} sn")
    print("Bayrak sayıları:")
    print(rap["bayraklar"].replace("", "yok").value_counts().to_string())
    print(f"Dışlanan: {int(rap['dislandi'].sum())}")


if __name__ == "__main__":
    main()
