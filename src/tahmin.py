# -*- coding: utf-8 -*-
"""Tek görüntü tahmini (web uygulaması ve komut satırı için sade arayüz).

    from tahmin import tahmin_et
    sonuc = tahmin_et("yaprak.jpg")            # ya da PIL.Image / numpy RGB dizisi
    sonuc["sinif"]          -> "adi_yaprak_yanikligi" | "hale_yanikligi" | "saglikli"
    sonuc["sinif_tr"]       -> "Adi yaprak yanıklığı (CBB)" | "Hale yanıklığı" | "Sağlıklı"
    sonuc["olasiliklar"]    -> {"adi_yaprak_yanikligi": 0.91, "hale_yanikligi": 0.05, "saglikli": 0.04}
    sonuc["on_isleme"]      -> {"zemin": "beyaz|siyah|kart|genel", "basarili": bool, "alan_orani": float}

Ham fotoğraf alınır ve eğitimdeki ön işleme uygulanır: zemin türü otomatik kestirilir, yaprak maskelenir,
zemin siyaha boyanır, yaprağa kırpılıp 256x256 yapılır (`genel_maske.on_isle`). Maskeleme başarısız olursa
görüntünün tamamıyla yine tahmin yapılır (`on_isleme.basarili = False`; sonuç daha az güvenilirdir).
Zaten ön işlenmiş (zemini siyah, 256x256) bir görüntü için `on_islenmis=True` verilebilir.

Model: `ciktilar/genel_model/model.pt` (ya da GENEL_MODEL_YOLU ortam değişkeni / `model_yolu` argümanı).
İlk çağrıda güvenli kipte (`torch.load(weights_only=True)`) yüklenir, anahtarları ve sınıf sırası
doğrulanır, (yol, değişiklik zamanı) anahtarıyla önbellekte tutulur. CPU'da da çalışır.

Uyarı: Model 4 açık veri kaynağıyla (A, B, D=PlantWild, E=Bugwood) eğitildi; başka koşullarda (tarla fotoğrafı, farklı tür) güvenilirliği
sınırlıdır — bkz. docs/RESULTS.tr.md §12 ve §13.
Komut satırı: python src/tahmin.py <goruntu_yolu> [...]
"""
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, SINIFLAR, SINIF_ADLARI_TR  # noqa: E402
import genel_maske  # noqa: E402

VARSAYILAN_MODEL = CIKTI / "genel_model" / "model.pt"
GEREKLI_ANAHTARLAR = ("model", "omurga", "boyut", "ortalama", "std", "siniflar")
_onbellek = {}  # çözümlenmiş yol -> (mtime_ns, (model, ck, cihaz, gm))


class ModelDosyasiHatasi(ValueError):
    """Model dosyası güvenli kipte okunamadı ya da beklenen yapıda değil."""


def kontrol_noktasi_oku(yol) -> dict:
    """`torch.load(weights_only=True)`: yalnız tensör/temel türler açılır, dosyadaki kod çalıştırılmaz."""
    try:
        ck = torch.load(yol, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise
    except Exception as e:  # pickle.UnpicklingError vb.: güvenli kipte açılamayan (değiştirilmiş?) dosya
        raise ModelDosyasiHatasi(f"Model dosyası güvenli kipte okunamadı ({Path(yol).name}): "
                                 f"{type(e).__name__}") from e
    if not isinstance(ck, dict):
        raise ModelDosyasiHatasi(f"Model dosyası beklenen sözlük yapısında değil ({Path(yol).name}).")
    return ck


def kontrol_noktasi_dogrula(ck: dict, yol="model") -> dict:
    """Genel model kontrol noktasında gerekli anahtarları ve sınıf listesini/sırasını doğrular."""
    eksik = [k for k in GEREKLI_ANAHTARLAR if k not in ck]
    if eksik:
        raise ModelDosyasiHatasi(f"Model dosyasında eksik alan(lar): {', '.join(eksik)} ({Path(yol).name}).")
    siniflar = ck["siniflar"]
    if not isinstance(siniflar, (list, tuple)) or list(siniflar) != list(SINIFLAR):
        raise ModelDosyasiHatasi(f"Model dosyasının sınıf listesi/sırası uyumsuz: {siniflar} "
                                 f"(beklenen {list(SINIFLAR)}).")
    if not (isinstance(ck["boyut"], int) and 32 <= ck["boyut"] <= 1024):
        raise ModelDosyasiHatasi(f"Model dosyasında geçersiz giriş boyutu: {ck['boyut']!r}.")
    normalizasyon_dogrula(ck["ortalama"], ck["std"])
    return ck


def normalizasyon_dogrula(ortalama, std):
    """ortalama/std: 3 sonlu sayı, std > 0 olmalı (yoksa ModelDosyasiHatasi)."""
    try:
        o = np.asarray(ortalama, dtype=np.float64)
        s = np.asarray(std, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ModelDosyasiHatasi("Model dosyasında ortalama/std sayısal değil.") from e
    if o.shape != (3,) or s.shape != (3,):
        raise ModelDosyasiHatasi("Model dosyasında ortalama/std 3 kanallı değil.")
    if not (np.isfinite(o).all() and np.isfinite(s).all()):
        raise ModelDosyasiHatasi("Model dosyasında ortalama/std sonlu sayı değil (NaN/sonsuz).")
    if not (s > 0).all():
        raise ModelDosyasiHatasi("Model dosyasında std sıfır ya da negatif.")


def _model_yukle(model_yolu=None):
    """Modeli (dosya yolu + değişiklik zamanı) anahtarıyla bir kez yükler; dosya değişirse yeniden yükler."""
    yol = Path(model_yolu or os.environ.get("GENEL_MODEL_YOLU") or VARSAYILAN_MODEL).resolve()
    mtime = yol.stat().st_mtime_ns
    kayit = _onbellek.get(str(yol))
    if kayit is not None and kayit[0] == mtime:
        return kayit[1]
    gm = importlib.import_module("09_genelleme")
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = kontrol_noktasi_dogrula(kontrol_noktasi_oku(yol), yol)
    model = gm.model_kur(ck["omurga"], onegitimli=False)  # ağırlıklar dosyadan gelir, indirme yok
    try:
        model.load_state_dict(ck["model"])
    except RuntimeError as e:
        raise ModelDosyasiHatasi(f"Model ağırlıkları '{ck['omurga']}' mimarisiyle uyumsuz ({yol.name}).") from e
    model.to(cihaz).eval()
    _onbellek[str(yol)] = (mtime, (model, ck, cihaz, gm))
    return _onbellek[str(yol)][1]


def _rgb_oku(girdi) -> np.ndarray:
    if isinstance(girdi, np.ndarray):
        a = girdi
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        return np.ascontiguousarray(a[..., :3]).astype(np.uint8)
    im = girdi if isinstance(girdi, Image.Image) else Image.open(girdi)
    if im.mode in ("RGBA", "LA", "P"):  # şeffaf alan beyaz zemine (eğitimdeki kural)
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    return np.asarray(im.convert("RGB"))


def tahmin_et(goruntu: Union[str, Path, Image.Image, np.ndarray], model_yolu=None, zemin: str = "otomatik",
              on_islenmis: bool = False) -> dict:
    """Tek yaprak görüntüsü için sınıf ve olasılıklar.

    goruntu    : dosya yolu, PIL.Image ya da RGB numpy dizisi (H, W, 3, uint8)
    zemin      : "otomatik" (önerilen) | "beyaz" | "siyah" | "kart" | "genel"
    on_islenmis: True ise maskeleme atlanır (girdi zaten zemini siyah, yaprağa kırpılmış olmalı;
                 maske siyah olmayan piksellerden çıkarılır). Maskeniz elinizdeyse `islenmisten_tahmin`.
    Döner: {"sinif", "sinif_tr", "olasiliklar": {sinif: p}, "on_isleme": {...}}
    """
    rgb = _rgb_oku(goruntu)
    if on_islenmis:
        g = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        m = (g.max(axis=2) > 0).astype(np.uint8) * 255
        bilgi = {"zemin": "on_islenmis", "basarili": True, "alan_orani": float((m > 0).mean())}
    else:
        g, m, bilgi = genel_maske.on_isle(rgb, zemin)
    return islenmisten_tahmin(g, m, bilgi, model_yolu)


def islenmisten_tahmin(g: np.ndarray, m: np.ndarray, bilgi: dict = None, model_yolu=None) -> dict:
    """`genel_maske.on_isle` çıktısıyla (görüntü 256x256 zemini siyah, maske 0/255) tahmin.

    Eğitim/değerlendirmedeki yol (`09_genelleme.GenelVeri` + `veri.maske_uygula`) ile aynıdır: görüntü
    INTER_AREA, maske en yakın komşu ile model boyutuna getirilir, maske dışı sıfırlanır, normalize edilir.
    Uygulama bu fonksiyonu kullanır; böylece ekranda gösterilen ve şiddeti ölçülen kırpım modele girenle aynıdır."""
    model, ck, cihaz, gm = _model_yukle(model_yolu)
    b = ck["boyut"]
    g = cv2.resize(g, (b, b), interpolation=cv2.INTER_AREA)
    m = (cv2.resize(m, (b, b), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
    x = torch.from_numpy(g).permute(2, 0, 1).float() / 255.0 * torch.from_numpy(m)[None]
    if ck.get("isik"):
        x = gm.isik_normalize(x, torch.from_numpy(m)[None])
    x = (x - torch.tensor(ck["ortalama"]).view(3, 1, 1)) / torch.tensor(ck["std"]).view(3, 1, 1)
    with torch.no_grad():
        p = F.softmax(model(x[None].to(cihaz)).float(), 1)[0].cpu().numpy()
    siniflar = ck["siniflar"]
    k = int(p.argmax())
    return {"sinif": siniflar[k], "sinif_tr": SINIF_ADLARI_TR[siniflar[k]],
            "olasiliklar": {s: round(float(v), 4) for s, v in zip(siniflar, p)},
            "on_isleme": bilgi if bilgi is not None else {"zemin": "on_islenmis", "basarili": True,
                                                          "alan_orani": float((m > 0).mean())}}


if __name__ == "__main__":
    for yol in sys.argv[1:]:
        print(yol, json.dumps(tahmin_et(yol), ensure_ascii=False))
