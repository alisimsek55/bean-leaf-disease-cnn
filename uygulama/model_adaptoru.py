# -*- coding: utf-8 -*-
"""İnce adaptör: uygulama ile model/şiddet kodu arasındaki tek temas noktası.

Tek ön işleme hattı: `src/genel_maske.on_isle` (genel model eğitimde/değerlendirmede bununla ölçüldü).
Aynı 256x256 görüntü + maske hem modele (`tahmin.islenmisten_tahmin`), hem şiddet ölçümüne, hem ekrana gider.
Model: ciktilar/genel_model/model.pt (src/tahmin.py) varsa o; yoksa ciktilar/model_en_iyi.pt (FasulyeCNN).
Modeller güvenli kipte (`torch.load(weights_only=True)`) bir kez yüklenir, (yol, mtime) ile önbellekte tutulur.
Şiddet: src/siddet.py varsa onun lezyon_maskesi'si, yoksa src/06_siddet_tahmini.py (importlib).
Düzey adları: sağlıklı / az hastalıklı / çok hastalıklı; yaprak ayrılamadıysa "hesaplanamadı",
değerlendirilebilir yaprak pikseli yoksa "belirsiz".
"""
import importlib
import importlib.util
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

KOK = Path(__file__).resolve().parent.parent
SRC = KOK / "src"
for _p in (str(KOK), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ortak import GIRIS_BOYUT, SINIFLAR  # noqa: E402
from uygulama import yaprak_ayirma  # noqa: E402
import genel_maske  # noqa: E402

try:  # bir kez içe aktarılır (yeniden yükleme modelin önbelleğini sıfırlardı)
    import tahmin as _tahmin  # noqa: E402
except Exception:  # torch/torchvision yoksa eski modele düşülür
    _tahmin = None

ESKI_MODEL = KOK / "ciktilar" / "model_en_iyi.pt"
GENEL_MODEL = KOK / "ciktilar" / "genel_model" / "model.pt"
GENEL_AD = "EfficientNet-B0 genel model (ciktilar/genel_model/model.pt)"
ESKI_AD = "FasulyeCNN (ciktilar/model_en_iyi.pt)"
T1, T2 = 0.005, 0.13                       # lezyon oranı eşikleri: sağlıklı < %0,5 <= az < %13 <= çok
DUZEYLER = ("sağlıklı", "az hastalıklı", "çok hastalıklı")
HESAPLANAMADI = "hesaplanamadı"            # yaprak zeminden ayrılamadı: şiddet ölçülmez
BELIRSIZ = "belirsiz"                      # yaprak maskesi aşınınca boşaldı (src/siddet.py ile aynı ad)

# ---------------------------------------------------------------- yükleme sınırları
MAKS_BAYT = 15 * 1024 * 1024               # .streamlit/config.toml: server.maxUploadSize = 15 (MB)
MAKS_PIKSEL = 40_000_000                   # 40 MP
MAKS_EN_BOY = 8.0                          # uzun kenar / kısa kenar
MIN_KENAR = 32
Image.MAX_IMAGE_PIXELS = MAKS_PIKSEL       # PIL'in kendi sıkıştırma bombası koruması da aynı sınırda


class GoruntuHatasi(ValueError):
    """Kullanıcıya olduğu gibi gösterilebilecek Türkçe görüntü hatası."""


@dataclass
class Sonuc:
    sinif: str
    olasiliklar: dict
    duzey: str                     # DUZEYLER'den biri, HESAPLANAMADI ya da BELIRSIZ
    lezyon_orani: float | None     # şiddet ölçülemediyse None
    orijinal: np.ndarray           # RGB, <=1024 px
    maske_yontemi: str
    maske_uyarisi: str | None
    yaprak: np.ndarray             # 256x256 kırpılmış, zemini siyah (modele giren görüntü)
    maske: np.ndarray              # 256x256 0/255 (modele giren maske)
    lezyon_katmani: np.ndarray     # yaprak üzerinde lezyonlar işaretli (256x256)
    model_adi: str
    ayirma_basarili: bool = True
    uyarilar: list = field(default_factory=list)

    @property
    def siddet_var(self) -> bool:
        return self.duzey in DUZEYLER


def _boyut_denetle(w: int, h: int):
    if w < MIN_KENAR or h < MIN_KENAR:
        raise GoruntuHatasi(f"Fotoğraf çok küçük ({w}x{h} piksel). En az {MIN_KENAR}x{MIN_KENAR} olmalı.")
    if w * h > MAKS_PIKSEL:
        raise GoruntuHatasi(f"Fotoğraf çok büyük ({w}x{h} = {w * h / 1e6:.0f} MP). "
                            f"En çok {MAKS_PIKSEL / 1e6:.0f} MP kabul edilir; fotoğrafı küçültüp tekrar deneyin.")
    if max(w, h) / min(w, h) > MAKS_EN_BOY:
        raise GoruntuHatasi(f"Fotoğrafın en-boy oranı çok uç ({w}x{h}). "
                            "Yaprağı ortalayan normal bir fotoğraf yükleyin.")


def _bayt_boyutu(kaynak):
    if isinstance(kaynak, (str, Path)):
        return Path(kaynak).stat().st_size
    if isinstance(getattr(kaynak, "size", None), int):  # Streamlit UploadedFile
        return kaynak.size
    if hasattr(kaynak, "seek") and hasattr(kaynak, "tell"):
        konum = kaynak.tell()
        kaynak.seek(0, io.SEEK_END)
        n = kaynak.tell()
        kaynak.seek(konum)
        return n
    return None


def goruntu_ac(kaynak) -> Image.Image:
    """Yol/dosya nesnesi -> EXIF yönü düzeltilmiş RGB PIL görüntüsü (telefon fotoğrafları için).

    Piksel verisi çözülmeden önce dosya boyutu (<= 15 MB), başlıktaki çözünürlük (<= 40 MP) ve
    en-boy oranı (<= 8:1) denetlenir; aşan dosyalar GoruntuHatasi ile reddedilir."""
    n = _bayt_boyutu(kaynak)
    if n is not None and n > MAKS_BAYT:
        raise GoruntuHatasi(f"Dosya çok büyük ({n / 1024 / 1024:.1f} MB). "
                            f"En çok {MAKS_BAYT // (1024 * 1024)} MB yükleyebilirsiniz.")
    try:
        im = Image.open(kaynak)  # yalnız başlık okunur
    except Image.DecompressionBombError as e:
        raise GoruntuHatasi("Fotoğraf çözünürlüğü çok büyük; küçültüp tekrar deneyin.") from e
    except (UnidentifiedImageError, OSError) as e:
        raise GoruntuHatasi("Dosya bir görüntü olarak okunamadı (JPG, PNG veya WEBP yükleyin).") from e
    _boyut_denetle(*im.size)
    try:
        im.load()
        im = ImageOps.exif_transpose(im)
    except Image.DecompressionBombError as e:
        raise GoruntuHatasi("Fotoğraf çözünürlüğü çok büyük; küçültüp tekrar deneyin.") from e
    except OSError as e:
        raise GoruntuHatasi("Görüntü dosyası bozuk ya da eksik; başka bir fotoğraf deneyin.") from e
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        zemin = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(zemin, im)
    return im.convert("RGB")


# ---------------------------------------------------------------- şiddet
_SIDDET = {}


def _siddet_modulu():
    """(modül, kaynak adı) — bir kez yüklenir."""
    if "m" not in _SIDDET:
        try:
            m = importlib.import_module("siddet")
            if not hasattr(m, "lezyon_maskesi"):
                raise ImportError
            _SIDDET["m"] = (m, "src/siddet.py")
        except Exception:
            spec = importlib.util.spec_from_file_location("_siddet_06", SRC / "06_siddet_tahmini.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _SIDDET["m"] = (mod, "src/06_siddet_tahmini.py")
    return _SIDDET["m"]


def duzey_adi(oran: float) -> str:
    return DUZEYLER[0] if oran < T1 else (DUZEYLER[1] if oran < T2 else DUZEYLER[2])


def siddet_hesapla(yaprak256: np.ndarray, maske256: np.ndarray):
    """Döndürür: (lezyon oranı veya None, düzey adı, lezyon ikili maskesi).

    Aşındırma sonrası değerlendirilebilir yaprak pikseli çok azsa düzey BELIRSIZ, oran None olur."""
    mod, _ = _siddet_modulu()
    lez, yap = mod.lezyon_maskesi(yaprak256, maske256)
    if int(yap.sum()) < getattr(mod, "MIN_YAPRAK_PIKSEL", 50):
        return None, getattr(mod, "BELIRSIZ", BELIRSIZ), np.zeros_like(lez)
    oran = float(lez.sum() / yap.sum())
    return oran, duzey_adi(oran), lez


# ---------------------------------------------------------------- model
_ESKI_CACHE = {}
_GENEL_HATA = {}   # (yol, mtime) -> hata metni: bozuk genel model her çağrıda yeniden denenmez
ESKI_ANAHTARLAR = ("model", "ortalama", "std", "gorev", "girdi", "sinif_sayisi")


def _normalizasyon_dogrula(ortalama, std):
    o = np.asarray(ortalama, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    if (o.shape != (3,) or s.shape != (3,) or not (np.isfinite(o).all() and np.isfinite(s).all())
            or not (s > 0).all()):
        raise ValueError("ortalama/std geçersiz (3 sonlu sayı ve std > 0 olmalı).")


def eski_kontrol_noktasi_dogrula(ck) -> dict:
    """Eski FasulyeCNN kontrol noktası: hastalık görevi, renkli girdi, 3 sınıf, geçerli normalizasyon.
    Siluet/kaynak kontrol modelleri (gorev/girdi farklı) aynı adla konsa da reddedilir."""
    if not isinstance(ck, dict):
        raise ValueError("Eski model dosyası sözlük yapısında değil.")
    eksik = [k for k in ESKI_ANAHTARLAR if k not in ck]
    if eksik:
        raise ValueError(f"Eski model dosyasında eksik alan(lar): {', '.join(eksik)}.")
    if ck["gorev"] != "hastalik" or ck["girdi"] != "renkli":
        raise ValueError(f"Eski model dosyası hastalık modeli değil (gorev={ck['gorev']!r}, "
                         f"girdi={ck['girdi']!r}).")
    if ck["sinif_sayisi"] != len(SINIFLAR):
        raise ValueError(f"Eski model dosyasının sınıf sayısı uyumsuz: {ck['sinif_sayisi']!r}.")
    try:
        _normalizasyon_dogrula(ck["ortalama"], ck["std"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Eski model: {e}") from e
    return ck


def _eski_model():
    import torch
    anahtar = (str(ESKI_MODEL.resolve()), ESKI_MODEL.stat().st_mtime_ns)
    if _ESKI_CACHE.get("anahtar") != anahtar:
        m3 = importlib.import_module("03_model")
        ck = eski_kontrol_noktasi_dogrula(torch.load(ESKI_MODEL, map_location="cpu", weights_only=True))
        model = m3.FasulyeCNN(sinif_sayisi=len(SINIFLAR))
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError as e:
            raise ValueError("Eski model ağırlıkları FasulyeCNN mimarisiyle uyumsuz.") from e
        model.eval()
        _ESKI_CACHE.update(anahtar=anahtar, m=(model, ck["ortalama"], ck["std"]))
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


def _genel_anahtar():
    return str(GENEL_MODEL.resolve()), GENEL_MODEL.stat().st_mtime_ns


def _genel_kullanilabilir() -> bool:
    """Genel model var ve bu dosya sürümü daha önce yüklenirken hata vermedi."""
    return _tahmin is not None and GENEL_MODEL.exists() and _genel_anahtar() not in _GENEL_HATA


def genel_model_hatasi():
    """Genel model dosyası bozuk/uyumsuz bulunduysa hata metni, yoksa None."""
    if _tahmin is None or not GENEL_MODEL.exists():
        return None
    return _GENEL_HATA.get(_genel_anahtar())


def _genel_tahmin(yaprak256, maske256, bilgi):
    return _tahmin.islenmisten_tahmin(yaprak256, maske256, bilgi, model_yolu=GENEL_MODEL)


def _genel_yukle() -> bool:
    """Genel modeli yüklemeyi dener; bozuk/uyumsuzsa hatayı kaydeder ve False döner."""
    if not _genel_kullanilabilir():
        return False
    try:
        _tahmin._model_yukle(GENEL_MODEL)
        return True
    except Exception as e:  # noqa: BLE001 - bozuk dosya, eksik anahtar, mimari uyumsuzluğu...
        _GENEL_HATA[_genel_anahtar()] = f"{type(e).__name__}: {e}"
        return False


def model_yukle():
    """Kullanılacak modeli önceden yükler (önbelleğe alır); model adını döndürür.

    Genel model yüklenemezse eski modele düşülür (`genel_model_hatasi()` nedeni verir);
    ikisi de kullanılamazsa hata fırlatılır."""
    if _genel_yukle():
        return GENEL_AD
    neden = genel_model_hatasi()
    ek = f"; genel model: {neden}" if neden else ""
    if ESKI_MODEL.exists():
        try:
            _eski_model()
            return ESKI_AD
        except Exception as e:
            raise RuntimeError(f"Hiçbir model yüklenemedi. Eski model: {e}{ek}") from e
    raise FileNotFoundError(f"Kullanılabilir model dosyası bulunamadı{ek}.")


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
    return GENEL_AD if _genel_kullanilabilir() else ESKI_AD


def model_var() -> bool:
    return _genel_kullanilabilir() or ESKI_MODEL.exists()


# ---------------------------------------------------------------- ana akış
AYIRMA_UYARISI = ("Yaprak zeminden ayrılamadı; model tüm fotoğrafı değerlendirdi ve şiddet hesaplanmadı. "
                  "Sonuç güvenilir olmayabilir. Yaprağı sade (beyaz/koyu) bir zemine koyup tekrar deneyin.")


def on_isle(rgb: np.ndarray):
    """Tek ön işleme hattı: (görüntü 256x256, maske 256x256 0/255, bilgi) — `genel_maske.on_isle`."""
    return genel_maske.on_isle(rgb, "otomatik")


def analiz_et(kaynak) -> Sonuc:
    if isinstance(kaynak, Image.Image):
        _boyut_denetle(*kaynak.size)
        pil = kaynak.convert("RGB")
    else:
        pil = goruntu_ac(kaynak)
    rgb = yaprak_ayirma.kucult(np.asarray(pil))
    yaprak, m256, bilgi = on_isle(rgb)
    basarili = bool(bilgi.get("basarili"))
    yontem = f"{bilgi.get('zemin')} zemin" if basarili else "ayırma başarısız (tüm görüntü)"
    uyari = None if basarili else AYIRMA_UYARISI
    if basarili and float(bilgi.get("alan_orani", 0)) > 0.75:
        uyari = "Yaprak görüntünün büyük kısmını kaplıyor; ayırma doğru olmayabilir."

    uyarilar = [uyari] if uyari else []
    ol, ad = None, None
    if _genel_yukle():
        try:
            ol, ad = _olasiliklari_duzenle(_genel_tahmin(yaprak, m256, bilgi)), GENEL_AD
        except Exception as e:  # yeni model hata verirse eskiye düş
            uyarilar.append(f"Yeni model çalışmadı ({type(e).__name__}: {e}); eski model kullanıldı.")
    elif genel_model_hatasi():
        uyarilar.append(f"Genel model yüklenemedi ({genel_model_hatasi()}); eski model kullanıldı.")
    if ol is None:
        if not ESKI_MODEL.exists():
            raise FileNotFoundError("Model dosyası bulunamadı (ciktilar/genel_model/model.pt).")
        ol, ad = _eski_tahmin(yaprak, m256), ESKI_AD
    sinif = max(ol, key=ol.get)

    kat = yaprak.copy()
    if basarili:
        oran, duzey, lez = siddet_hesapla(yaprak, m256)
        kat[lez > 0] = (0.35 * kat[lez > 0] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    else:  # tüm fotoğraf "yaprak" sayılırsa koyu masa/toprak lezyon görünür: ölçme
        oran, duzey = None, HESAPLANAMADI
    return Sonuc(sinif=sinif, olasiliklar=ol, duzey=duzey, lezyon_orani=oran, orijinal=rgb,
                 maske_yontemi=yontem, maske_uyarisi=uyari, yaprak=yaprak, maske=m256, lezyon_katmani=kat,
                 model_adi=ad, ayirma_basarili=basarili, uyarilar=uyarilar)
