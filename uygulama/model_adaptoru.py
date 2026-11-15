# -*- coding: utf-8 -*-
"""İnce adaptör: uygulama ile model/şiddet kodu arasındaki tek temas noktası.

Model seçimi (her çağrıda yeniden denenir, yeni model gelince kod değişmeden geçilir):
  1) src/tahmin.py içinde tahmin_et içe aktarılabiliyorsa o (orijinal fotoğrafı PIL olarak alır).
  2) Yoksa ciktilar/model_en_iyi.pt (FasulyeCNN) + bu uygulamanın kendi ön işlemesi.
Şiddet: src/siddet.py varsa onun lezyon_maskesi'si, yoksa src/06_siddet_tahmini.py (importlib).
Düzey adları her durumda: sağlıklı / az hastalıklı / çok hastalıklı.
"""
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

KOK = Path(__file__).resolve().parent.parent
SRC = KOK / "src"
for _p in (str(KOK), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ortak import GIRIS_BOYUT, SINIFLAR  # noqa: E402
from uygulama import yaprak_ayirma  # noqa: E402

ESKI_MODEL = KOK / "ciktilar" / "model_en_iyi.pt"
GENEL_MODEL = KOK / "ciktilar" / "genel_model" / "model.pt"
GENEL_AD = "EfficientNet-B0 genel model (ciktilar/genel_model/model.pt)"
T1, T2 = 0.005, 0.13                       # lezyon oranı eşikleri: sağlıklı < %0,5 <= az < %13 <= çok
DUZEYLER = ("sağlıklı", "az hastalıklı", "çok hastalıklı")
_ESKI_AD = {"hastalıklı": "çok hastalıklı"}


@dataclass
class Sonuc:
    sinif: str
    olasiliklar: dict
    duzey: str
    lezyon_orani: float
    orijinal: np.ndarray           # RGB, <=1024 px
    maske_yontemi: str
    maske_uyarisi: str | None
    yaprak: np.ndarray             # 256x256 kırpılmış, zemini siyah
    lezyon_katmani: np.ndarray     # yaprak üzerinde lezyonlar işaretli (256x256)
    model_adi: str
    uyarilar: list = field(default_factory=list)


def goruntu_ac(kaynak) -> Image.Image:
    """Yol/dosya nesnesi -> EXIF yönü düzeltilmiş RGB PIL görüntüsü (telefon fotoğrafları için)."""
    im = Image.open(kaynak)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        zemin = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(zemin, im)
    return im.convert("RGB")


# ---------------------------------------------------------------- şiddet
def _siddet_modulu():
    """(lezyon_maskesi fonksiyonu, kaynak adı)."""
    try:
        m = importlib.import_module("siddet")
        if hasattr(m, "lezyon_maskesi"):
            return m.lezyon_maskesi, "src/siddet.py"
    except Exception:
        pass
    spec = importlib.util.spec_from_file_location("_siddet_06", SRC / "06_siddet_tahmini.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.lezyon_maskesi, "src/06_siddet_tahmini.py"


def duzey_adi(oran: float) -> str:
    return DUZEYLER[0] if oran < T1 else (DUZEYLER[1] if oran < T2 else DUZEYLER[2])


def siddet_hesapla(yaprak256: np.ndarray, maske256: np.ndarray):
    """Döndürür: (lezyon oranı, düzey adı, lezyon ikili maskesi)."""
    fn, _ = _siddet_modulu()
    lez, yap = fn(yaprak256, maske256)
    oran = float(lez.sum() / max(yap.sum(), 1))
    return oran, duzey_adi(oran), lez


# ---------------------------------------------------------------- model
_ESKI_CACHE = {}


def _eski_model():
    if "m" not in _ESKI_CACHE:
        import torch
        from importlib import import_module
        m3 = import_module("03_model")
        ck = torch.load(ESKI_MODEL, map_location="cpu", weights_only=False)
        model = m3.FasulyeCNN(sinif_sayisi=ck.get("sinif_sayisi", 3))
        model.load_state_dict(ck["model"])
        model.eval()
        _ESKI_CACHE["m"] = (model, ck["ortalama"], ck["std"])
    return _ESKI_CACHE["m"]


def _eski_tahmin(yaprak256, maske256):
    import torch
    model, ort, std = _eski_model()
    g = cv2.resize(yaprak256, (GIRIS_BOYUT, GIRIS_BOYUT), interpolation=cv2.INTER_AREA)
    m = cv2.resize(maske256, (GIRIS_BOYUT, GIRIS_BOYUT), interpolation=cv2.INTER_NEAREST) > 127
    g = g * m[..., None]
    x = torch.from_numpy(g.transpose(2, 0, 1).copy()).float() / 255.0
    x = (x - torch.tensor(ort).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    with torch.no_grad():
        p = torch.softmax(model(x[None]), dim=1)[0].tolist()
    return {s: float(v) for s, v in zip(SINIFLAR, p)}


def _yeni_tahmin_fonksiyonu():
    try:
        mod = importlib.import_module("tahmin")
        importlib.reload(mod)
        return getattr(mod, "tahmin_et", None)
    except Exception:
        return None


def _olasiliklari_duzenle(cikti: dict) -> dict:
    ol = cikti.get("olasiliklar", cikti)
    sonuc = {s: 0.0 for s in SINIFLAR}
    for k, v in ol.items():
        ad = str(k)
        if ad not in sonuc:  # 'CBB', 'Sağlıklı' gibi adları eşle
            dusuk = ad.lower()
            ad = next((s for s in SINIFLAR if s in dusuk or
                       (s.startswith("adi") and "cbb" in dusuk) or
                       (s.startswith("hale") and "hale" in dusuk) or
                       (s == "saglikli" and ("sağlıklı" in dusuk or "saglikli" in dusuk))), ad)
        if ad in sonuc:
            sonuc[ad] = float(v)
    top = sum(sonuc.values())
    if top > 0:
        sonuc = {k: v / top for k, v in sonuc.items()}
    return sonuc


def hastalik_olasiligi(ol: dict) -> float:
    """P(hastalıklı) = P(CBB) + P(hale); P(sağlıklı) = 1 - bu."""
    return float(ol.get("adi_yaprak_yanikligi", 0.0) + ol.get("hale_yanikligi", 0.0))


def model_adi() -> str:
    if _yeni_tahmin_fonksiyonu() is not None and GENEL_MODEL.exists():
        return GENEL_AD
    return "FasulyeCNN (ciktilar/model_en_iyi.pt)"


def model_var() -> bool:
    return (_yeni_tahmin_fonksiyonu() is not None and GENEL_MODEL.exists()) or ESKI_MODEL.exists()


# ---------------------------------------------------------------- ana akış
def analiz_et(kaynak) -> Sonuc:
    pil = kaynak if isinstance(kaynak, Image.Image) else goruntu_ac(kaynak)
    rgb = yaprak_ayirma.kucult(np.asarray(pil))
    maske, yontem, uyari = yaprak_ayirma.yaprak_ayir(rgb)
    yaprak, m256 = yaprak_ayirma.kirp(rgb, maske)
    if yaprak is None:  # maske boş: tüm görüntü
        maske = np.ones(rgb.shape[:2], np.uint8)
        yaprak, m256 = yaprak_ayirma.kirp(rgb, maske)
        yontem, uyari = "ayırma başarısız (tüm görüntü)", "Yaprak bulunamadı; tüm fotoğraf kullanıldı."

    uyarilar = [uyari] if uyari else []
    ol, ad = None, None
    f = _yeni_tahmin_fonksiyonu() if GENEL_MODEL.exists() else None
    if f is not None:
        try:
            ol, ad = _olasiliklari_duzenle(f(pil)), GENEL_AD
        except Exception as e:  # yeni model hata verirse eskiye düş
            uyarilar.append(f"Yeni model çalışmadı ({type(e).__name__}); eski model kullanıldı.")
    if ol is None:
        if not ESKI_MODEL.exists():
            raise FileNotFoundError("Model dosyası bulunamadı (ciktilar/model_en_iyi.pt).")
        ol, ad = _eski_tahmin(yaprak, m256), "FasulyeCNN (ciktilar/model_en_iyi.pt)"
    sinif = max(ol, key=ol.get)

    oran, duzey, lez = siddet_hesapla(yaprak, m256)
    kat = yaprak.copy()
    kat[lez > 0] = (0.35 * kat[lez > 0] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    return Sonuc(sinif=sinif, olasiliklar=ol, duzey=duzey, lezyon_orani=oran, orijinal=rgb,
                 maske_yontemi=yontem, maske_uyarisi=uyari, yaprak=yaprak, lezyon_katmani=kat,
                 model_adi=ad, uyarilar=uyarilar)
