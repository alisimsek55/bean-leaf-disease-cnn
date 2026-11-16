# -*- coding: utf-8 -*-
"""
12 - İKİNCİ GERÇEK CBB KAYNAĞI (F = Selçuk Üniversitesi "Bean Bacterial Blight Disease Dataset"):
kopyalama + SHA-256 metadata + 8 yönlü dHash kopya/artırma denetimi + ön işleme.

Kaynak: Kursun R., Gur A., Bastas K. K., Koklu M. (2024). "Detection of Bean Bacterial Blight Disease ..."
*Agronomy* 14(7):1495, doi:10.3390/agronomy14071495. Veri: https://www.muratkoklu.com/datasets/
(Bean_Bacterial_Blight_Disease_Dataset.zip). Lisans sayfada açık değil; atıf zorunlu. Görüntüler YENİDEN
DAĞITILMAZ: data/raw_selcuk/ ve data/islenmis_selcuk/ git dışındadır; depoda yalnız bu betik ve
data/selcuk_metadata.csv (dosya adı, çeşit, sınıf, SHA-256, boyut) vardır.

Girdi : --kaynak <indirilip açılmış klasör> (yalnız OKUNUR; içine yazılmaz, silinmez). Alt klasörler:
        Akbulut_Bacterial_Blight_Disease, Akbulut_Healthy, Ustun42_Bacterial_Blight_Disease, Ustun42_Healthy.
Adımlar:
  1. Kopyala: data/raw_selcuk/<cesit>_<cbb|saglikli>_<no>.jpg (no = orijinal adın parantez içindeki sayısı),
     SHA-256 + boyut -> data/selcuk_metadata.csv.
  2. Birebir kopya (aynı SHA-256) -> ilki tutulur, diğerleri atılır.
  3. 8 yönlü dHash (256 bit; 4 döndürme x ayna): i ile j "yakın kopya" <=> min_k Hamming(dHash(T_k(i)), dHash(j))
     <= ESIK. Döndürülmüş/çevrilmiş/yeniden boyutlandırılmış/hafif kırpılmış artırmalar ve aynı fotoğrafın
     farklı çözünürlükteki kopyaları böyle yakalanır. Bağlantılar union-find ile `grup` olur; bölme gruplarla yapılır.
     Farklı sınıfta yakın kopyası olan (etiket çelişkisi) gruplar atılır.
  4. A/B/C (data/raw), D (web), E (Bugwood kırpımları) ile çapraz yakın kopya: F görüntüsünün 8 yönünden biri
     herhangi bir dış görüntüye <= ESIK ise F'den ATILIR (özellikle C dışarıda kalsın diye). C görüntüleri burada
     YALNIZ dHash için okunur (etiket/model çıktısı kullanılmaz) — 08/11'deki kopya-denetimi istisnasıyla aynı.
  5. Ön işleme: çözünürlük birliği için uzun kenar 500 px'e küçültülür (veride 4608x2128'lik kareler de var),
     sonra `genel_maske.on_isle(otomatik)` — uygulamadaki `tahmin_et` ile AYNI yol. Gerekçe: RESULTS §14.
Çıktı: data/islenmis_selcuk/{goruntu,maske}/<sinif>/*.png, data/islenmis_selcuk/selcuk_veri.csv (bolme_*.csv ile
       aynı sütunlar + cesit, zemin, alan_orani), selcuk_dislanan.csv, selcuk_yakin_ciftler.csv, selcuk_ozeti.json,
       ciktilar/genel_model/selcuk_maske_kontrol.jpg (gözle kontrol).
Kullanım:
  python src/12_selcuk_veri.py --kaynak "<klasör>"     # tam akış
  python src/12_selcuk_veri.py --panel                  # ciktilar/sekiller/rapor/sekil_selcuk_ornekleri.png
  python src/12_selcuk_veri.py --taban ciktilar/genel_model/model_e2_onceki.pt   # F'yi görmemiş model, F'nin tamamı
"""
import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, KOK, TOHUM, VERI  # noqa: E402

_spec = importlib.util.spec_from_file_location("yakin_kopya", KOK / "scripts" / "yakin_kopya.py")
yk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yk)

HAM_F = VERI / "raw_selcuk"
CIKIS = VERI / "islenmis_selcuk"
METADATA = VERI / "selcuk_metadata.csv"
KAYNAK = "selcuk_bbb"
ESIK = 20  # 256 bit üzerinden Hamming (08/11 ile aynı)
UZUN_KENAR = 500   # = genel_maske.ILK_UZUN_KENAR (ortak hat; main'de denetlenir)
SERI_MANIFEST = VERI / "selcuk_seri_baglari.csv"  # gözle doğrulanmış ek seri bağları (dosya1, dosya2, not)
SERI_BLOK = 10     # aynı klasörde ardışık 10 numara = bir seri (çekim sırası; A'nın 10 karelik serileri gibi)
SERI_ESIK = 0.85   # blok sınırında ardışık iki karenin ImageNet gömmesi kosinüs benzerliği eşiği
ZAMAN_PENCERE_SN = 10  # aynı klasörde çekim zamanına göre ardışık iki kare arası <= 10 s ise aynı seri (EXIF)
ATIF = ("Kursun R., Gur A., Bastas K. K., Koklu M. (2024). Agronomy 14(7):1495, doi:10.3390/agronomy14071495 "
        "(Selçuk Üniversitesi, Bean Bacterial Blight Disease Dataset)")
KLASORLER = {
    "Akbulut_Bacterial_Blight_Disease": ("akbulut", "adi_yaprak_yanikligi"),
    "Akbulut_Healthy": ("akbulut", "saglikli"),
    "Ustun42_Bacterial_Blight_Disease": ("ustun42", "adi_yaprak_yanikligi"),
    "Ustun42_Healthy": ("ustun42", "saglikli"),
}
SINIF_KISA = {"adi_yaprak_yanikligi": "cbb", "saglikli": "saglikli"}
# 8 yön (dihedral grup D4): özdeşlik, 3 döndürme, 4 ayna
YONLER = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
          Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE,
          Image.Transpose.TRANSVERSE]


# ----------------------------------------------------------------------------- 1. kopyalama + metadata
def orijinal_no(ad: str) -> int:
    """'Akbulut_Healthy (12).jpg' -> 12."""
    m = re.search(r"\((\d+)\)", ad)
    if not m:
        raise ValueError(f"beklenmeyen dosya adı: {ad}")
    return int(m.group(1))


def hedef_ad(cesit: str, sinif: str, no: int) -> str:
    return f"{cesit}_{SINIF_KISA[sinif]}_{no:04d}.jpg"


def cekim_zamani(im: Image.Image):
    """EXIF DateTimeOriginal (yoksa DateTime) -> 'YYYY-MM-DD HH:MM:SS' ya da None."""
    from datetime import datetime
    e = im.getexif()
    z = e.get_ifd(0x8769).get(36867) or e.get(306)
    try:
        return datetime.strptime(str(z).strip(), "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S") if z else None
    except ValueError:
        return None


def sha256_dosya(yol) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def metadata_olustur(kaynak: Path, hedef: Path = HAM_F, kopyala: bool = True) -> pd.DataFrame:
    """Kaynak klasörü (yalnız okunur) -> hedef/<ad>.jpg kopyaları + metadata DataFrame.
    Hedefte aynı SHA-256'lı dosya varsa yeniden kopyalanmaz."""
    kayit = []
    for klasor, (cesit, sinif) in KLASORLER.items():
        d = Path(kaynak) / klasor
        if not d.is_dir():
            raise FileNotFoundError(f"beklenen alt klasör yok: {d}")
        for f in sorted(d.glob("*.jpg"), key=lambda p: orijinal_no(p.name)):
            sha = sha256_dosya(f)
            ad = hedef_ad(cesit, sinif, orijinal_no(f.name))
            if kopyala:
                hedef.mkdir(parents=True, exist_ok=True)
                h = hedef / ad
                if not (h.exists() and sha256_dosya(h) == sha):
                    shutil.copyfile(f, h)
            with Image.open(f) as im:
                w, hh = im.size
                yon, zaman = im.getexif().get(274, 1), cekim_zamani(im)
            kayit.append({"dosya": f"{hedef.name}/{ad}", "orijinal": f"{klasor}/{f.name}", "cesit": cesit,
                          "sinif": sinif, "sha256": sha, "genislik": w, "yukseklik": hh,
                          "bayt": f.stat().st_size, "exif_yon": int(yon), "cekim_zamani": zaman})
    return pd.DataFrame(kayit)


# ----------------------------------------------------------------------------- 2-3. dHash (8 yön) + gruplar
def dhash_bitleri(im: Image.Image, n: int = yk.N) -> np.ndarray:
    """yk.dhash ile aynı bit tanımı (bit y*n+x = piksel(y,x) > piksel(y,x+1)), bool dizi [n*n]."""
    g = np.asarray(im.convert("L").resize((n + 1, n), Image.LANCZOS), dtype=np.int16)
    return (g[:, :-1] > g[:, 1:]).reshape(-1)


def dhash8(im: Image.Image) -> np.ndarray:
    """8 yönün dHash bitleri [8, 256] (0 = özdeşlik)."""
    g = im.convert("L")
    return np.stack([dhash_bitleri(g if t is None else g.transpose(t)) for t in YONLER])


def bitler_int(b: np.ndarray) -> int:
    """bool bitler -> yk.dhash ile aynı tamsayı (bit i = b[i])."""
    return sum(1 << int(i) for i in np.flatnonzero(b))


def min8_uzaklik(H8a: np.ndarray, Hb: np.ndarray, blok: int = 2048) -> np.ndarray:
    """H8a [Na,8,256] bool, Hb [Nb,256] bool -> [Na,Nb] min_k Hamming(H8a[i,k], Hb[j]) (int16)."""
    B = (Hb.astype(np.float32) * 2 - 1)
    nbit = Hb.shape[1]
    out = np.empty((len(H8a), len(Hb)), np.int16)
    for s in range(0, len(H8a), blok):
        A = H8a[s:s + blok].astype(np.float32) * 2 - 1  # [b,8,256]
        d = (nbit - np.einsum("akn,mn->akm", A, B)) / 2
        out[s:s + blok] = np.rint(d.min(1)).astype(np.int16)
    return out


def ic_yakin_ciftler(H8: np.ndarray, esik: int = ESIK):
    """Küme içi 8 yönlü yakın kopya çiftleri: [(i, j, uzaklık)] (i < j). Simetrik değil (T_k(i) vs j) -> iki yön min."""
    D = min8_uzaklik(H8, H8[:, 0])
    D = np.minimum(D, D.T)
    i, j = np.nonzero(np.triu(D <= esik, 1))
    return [(int(a), int(b), int(D[a, b])) for a, b in zip(i, j)], D


def gruplar_kur(n: int, ciftler) -> list:
    return yk.grup_etiketleri(n, [(i, j) for i, j, *_ in ciftler])


def seri_baglantilari(klasor, no, benzerlik, blok: int = SERI_BLOK, esik: float = SERI_ESIK, pencere: int = 2):
    """Aynı klasörde (çeşit x sınıf) fotoğraflar çekim sırasıyla numaralı; ardışık kareler çoğu zaman aynı
    bitki/yaprağın biraz farklı açıdan çekimi (gözle doğrulandı). Bağlantılar:
      * aynı `blok`'luk numara aralığındaki ((no-1)//blok eşit) kareler birbirine (aynı seri sayılır);
      * numara farkı <= `pencere` olup blok sınırını aşan çiftler, görüntü benzerliği >= `esik` ise.
    benzerlik(i, j) -> float (ImageNet gömmesi kosinüs benzerliği; testte sahte fonksiyon).
    Döner [(i, j)] (indeksler girdi sırasına göre)."""
    katman = {}
    for i, (k, n) in enumerate(zip(klasor, no)):
        katman.setdefault(k, []).append((int(n), i))
    ciftler = []
    for ogeler in katman.values():
        ogeler.sort()
        for p in range(len(ogeler)):
            for q in range(p + 1, len(ogeler)):
                (na, a), (nb, b) = ogeler[p], ogeler[q]
                if nb - na > pencere:
                    break
                if (na - 1) // blok == (nb - 1) // blok or benzerlik(a, b) >= esik:
                    ciftler.append((a, b))
    return ciftler


def zaman_baglantilari(klasor, zaman, pencere: float = ZAMAN_PENCERE_SN):
    """Aynı klasörde çekim zamanına göre sıralı ardışık iki kare arası <= pencere saniye ise bağ [(i, j)].
    (A'daki seri çekim kuralının aynısı; F'de kareler ~5-8 s arayla çekilmiş, aynı yaprağın tekrar çekimi
    gözle 2-12 s arasında görüldü.) Zamanı olmayan kare bağlanmaz."""
    katman = {}
    for i, (k, z) in enumerate(zip(klasor, zaman)):
        if isinstance(z, str) and z:
            katman.setdefault(k, []).append((pd.Timestamp(z), i))
    out = []
    for ogeler in katman.values():
        ogeler.sort()
        for (t0, a), (t1, b) in zip(ogeler, ogeler[1:]):
            if (t1 - t0).total_seconds() <= pencere:
                out.append((a, b))
    return out


def gomme_agirlik_yolu(yol=None) -> Path:
    """Seri gruplaması için ImageNet EfficientNet-B0 ağırlık dosyası. Ağdan İNDİRİLMEZ: verilen yol ya da torch hub
    önbelleği (`<hub>/checkpoints/efficientnet_b0_rwightman-*.pth`) olmalı; yoksa anlaşılır hata (yan etkiden önce)."""
    if yol:
        y = Path(yol)
        if not y.is_file():
            raise FileNotFoundError(f"Gömme ağırlık dosyası yok: {y}")
        return y
    import torch
    import torchvision.models as tm
    ad = Path(tm.EfficientNet_B0_Weights.DEFAULT.url).name
    y = Path(torch.hub.get_dir()) / "checkpoints" / ad
    if not y.is_file():
        raise FileNotFoundError(
            f"EfficientNet-B0 ImageNet ağırlıkları yerel önbellekte yok ({y}). Bu betik ağdan indirmez: dosyayı "
            f"{tm.EfficientNet_B0_Weights.DEFAULT.url} adresinden elle indirip bu yola koyun ya da --gomme_agirlik verin.")
    return y


def gomme_girdisi(yol, kisa: int = 224) -> np.ndarray:
    """EXIF yönü uygulanmış görüntü, en-boy yönü korunarak (yatay 2:1 / dikey 1:2) kisa x 2*kisa'ya getirilir."""
    im = ImageOps.exif_transpose(Image.open(yol)).convert("RGB")
    w, h = im.size
    return np.array(im.resize((2 * kisa, kisa) if w >= h else (kisa, 2 * kisa)))


def gomme_hesapla(yollar, agirlik=None, bs: int = 32) -> np.ndarray:
    """ImageNet ön eğitimli EfficientNet-B0 (yalnız yerel ağırlık) ile L2-normalize gömme [N,1280].
    Yalnız F içi seri gruplaması için (benzer ardışık kareleri bağlamak); modelle/C ile ilgisi yok.
    Girdi EXIF yönü uygulanmış görüntüdür (dHash ve ön işleme ile aynı)."""
    import torch
    import torchvision.models as tm
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ag = tm.efficientnet_b0(weights=None)
    ag.load_state_dict(torch.load(gomme_agirlik_yolu(agirlik), map_location="cpu", weights_only=True))
    ag = ag.eval().to(cihaz)
    f = torch.nn.Sequential(ag.features, ag.avgpool, torch.nn.Flatten())
    ort = torch.tensor([0.485, 0.456, 0.406], device=cihaz).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=cihaz).view(1, 3, 1, 1)
    E = []
    with torch.no_grad():
        for y in yollar:  # yönler farklı olabildiği için tek tek (3648 görüntü, GPU'da ~1 dk)
            x = torch.from_numpy(gomme_girdisi(y)).permute(2, 0, 1)[None]
            x = (x.to(cihaz).float() / 255 - ort) / std
            E.append(torch.nn.functional.normalize(f(x), dim=1).cpu())
    return torch.cat(E).numpy()


def elle_seri_baglari(dosyalar, yol=SERI_MANIFEST):
    """Manifestteki (dosya1, dosya2) çiftlerini indeks çiftlerine çevirir. Atılmış (kopya) dosyalar atlanır;
    metadata'da hiç olmayan bir ad yazım hatasıdır -> ValueError."""
    if not Path(yol).exists():
        return []
    m = pd.read_csv(yol, comment="#")
    ix = {d: i for i, d in enumerate(dosyalar)}
    out = []
    for r in m.itertuples():
        a, b = f"raw_selcuk/{r.dosya1}", f"raw_selcuk/{r.dosya2}"
        for d in (a, b):
            if not (HAM_F / Path(d).name).exists() and d not in ix:
                raise ValueError(f"seri manifestinde bilinmeyen dosya: {d}")
        if a in ix and b in ix:
            out.append((ix[a], ix[b]))
    return out


# ----------------------------------------------------------------------------- bölme
def gruplu_bol(df: pd.DataFrame, oran: float = 0.3, tohum: int = TOHUM, katman=("sinif", "cesit")):
    """Grup bazlı, katmanlı (varsayılan sınıf x çeşit) bölme: her katmanda gruplar karıştırılır ve doğrulama
    payı oran*N'ye ulaşana kadar grup eklenir (web_bol ile aynı mantık). Bir grubun katmanı ilk satırından alınır
    (gruplar katman sınırını aşmamalı; aşarsa ValueError). Döner (eğitim, doğrulama)."""
    if df.groupby("grup")[list(katman)].nunique().max().max() > 1:
        raise ValueError("bir grup birden çok katmana yayılıyor")
    rng = np.random.default_rng(tohum)
    dog = []
    for _, g in df.groupby(list(katman), sort=True):
        gr = sorted(set(g.grup))
        rng.shuffle(gr)
        boy = g.grup.value_counts()
        hedef, n = oran * len(g), 0
        for x in gr:  # karışık sırada; grup eklemek hedefe yaklaştırıyorsa eklenir (büyük gruplar aşmasın)
            if abs(n + boy[x] - hedef) < abs(n - hedef):
                dog.append(x)
                n += int(boy[x])
    m = df.grup.isin(dog)
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)


# ----------------------------------------------------------------------------- 4. dış kaynak hashleri
def dis_hashler():
    """A/B/C (data/raw), D (işlenmemiş web), E (alınan kırpımlar, 11_bugwood ile aynı) -> (etiketler, H [N,256])."""
    et, H = [], []
    ham = pd.read_csv(VERI / "etiketler.csv")
    ham = ham[ham.kume == "raw"]
    kod = {"mendeley_ykvcrjffzd": "A", "mendeley_3km9d246z2": "B", "embrapa_pddb": "C"}
    for k, d in zip(ham.kaynak, ham.dosya):
        with Image.open(VERI / d) as im:
            H.append(dhash_bitleri(im))
        et.append((kod[k], d))
    wyol = VERI / "islenmis_web" / "web_veri.csv"
    if wyol.exists():
        for d in pd.read_csv(wyol).dosya:
            with Image.open(VERI / d) as im:
                H.append(dhash_bitleri(im))
            et.append(("D", d))
    ayk = VERI / "bugwood_ayiklama.csv"
    if ayk.exists() and (VERI / "raw_bugwood").exists():
        bw = importlib.import_module("11_bugwood_veri")
        ay = pd.read_csv(ayk, dtype={"numara": str, "parca": str, "kirpma": str})
        for r in ay[ay.karar == "al"].itertuples():
            yol = VERI / "raw_bugwood" / f"{r.numara}.jpg"
            if not yol.exists():
                continue
            rgb = np.asarray(Image.open(yol).convert("RGB"))
            x0, y0, x1, y1 = bw.kirpma_kutusu(r.kirpma, rgb.shape[1], rgb.shape[0])
            H.append(dhash_bitleri(Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1]))))
            et.append(("E", f"raw_bugwood/{r.numara}.jpg"))
    return et, np.stack(H)


# ----------------------------------------------------------------------------- 5. ön işleme
def kontrol_sayfasi(ciftler, yol, T=200, sut=6):
    """(ad, ham, işlenmiş) üçlüleri -> yan yana (ham | işlenmiş) gözle kontrol sayfası."""
    sat = (len(ciftler) + sut // 2 - 1) // (sut // 2)
    sayfa = Image.new("RGB", (sut * T, max(1, sat) * (T + 14)), "white")
    c = ImageDraw.Draw(sayfa)
    for j, (ad, k, g) in enumerate(ciftler):
        x, y = (j % (sut // 2)) * 2 * T, (j // (sut // 2)) * (T + 14)
        a = Image.fromarray(k)
        a.thumbnail((T, T))
        sayfa.paste(a, (x, y + 14 + (T - a.size[1]) // 2))
        sayfa.paste(Image.fromarray(g).resize((T, T)), (x + T, y + 14))
        c.text((x + 2, y + 1), ad, fill="black")
    Path(yol).parent.mkdir(parents=True, exist_ok=True)
    sayfa.save(yol, quality=85)


def _isle(yol):
    """Paralel işçi: ham dosya -> (görüntü, maske, bilgi, küçültülmüş ham). Ön işleme `genel_maske.ham_on_isle`:
    `tahmin_et` ve uygulama ile AYNI hat (EXIF + uzun kenar 500 + on_isle)."""
    import cv2
    cv2.setNumThreads(1)
    genel_maske = importlib.import_module("genel_maske")
    g, m, bilgi = genel_maske.ham_on_isle(yol, "otomatik")
    return g, m, bilgi, genel_maske.ilk_kucult(genel_maske.goruntu_oku(yol))


def sha_ayikla(meta: pd.DataFrame):
    """Birebir kopyalar (aynı SHA-256): grupta birden çok sınıf varsa (etiket çelişkisi) grubun TAMAMI atılır;
    tek sınıfsa ilki tutulur, diğerleri atılır. Döner (tutulan df, dışlanan kayıtlar)."""
    dislanan, at = [], set()
    for sha, g in meta.groupby("sha256", sort=False):
        if len(g) < 2:
            continue
        if g.sinif.nunique() > 1:
            for i in g.index:
                at.add(i)
                dislanan.append({"dosya": meta.dosya[i], "neden": "birebir_kopya_sinif_celiskili",
                                 "esli": ";".join(d for d in g.dosya if d != meta.dosya[i]), "uzaklik": 0})
        else:
            for i in g.index[1:]:
                at.add(i)
                dislanan.append({"dosya": meta.dosya[i], "neden": "birebir_kopya", "esli": g.dosya.iloc[0], "uzaklik": 0})
    return meta.drop(index=sorted(at)).reset_index(drop=True), dislanan


def kopya_ayikla(df: pd.DataFrame, ciftler):
    """8 yönlü dHash yakın-kopya çiftleri -> (atılacak {indeks: (neden, eşi, uzaklık)}, grup içi bağlantılar).
    Kural: farklı sınıf = etiket çelişkisi, ikisi de atılır; aynı sınıf farklı klasör (çeşit) = aynı fotoğrafın
    ikinci klasördeki kopyası, sonraki (indeksi büyük) atılır; aynı klasör = aynı gruba bağlanır."""
    at, bag = {}, []
    for i, j, d in ciftler:
        if df.sinif[i] != df.sinif[j]:
            at[i] = ("sinif_celiskili_yakin_kopya", df.dosya[j], d)
            at[j] = ("sinif_celiskili_yakin_kopya", df.dosya[i], d)
        elif df.cesit[i] != df.cesit[j]:
            at.setdefault(j, ("yakin_kopya_baska_cesit_klasoru", df.dosya[i], d))
        else:
            bag.append((i, j))
    return at, bag


def main(a):
    from concurrent.futures import ProcessPoolExecutor
    genel_maske = importlib.import_module("genel_maske")
    assert genel_maske.ILK_UZUN_KENAR == UZUN_KENAR
    agirlik = gomme_agirlik_yolu(a.gomme_agirlik)  # yan etkilerden ÖNCE: yerel ağırlık yoksa burada dur
    CIKIS.mkdir(parents=True, exist_ok=True)
    # 1. kopya + metadata
    meta = metadata_olustur(Path(a.kaynak))
    meta.to_csv(METADATA, index=False)
    boyutlar = {f"{s}|{c}": {k: int(v) for k, v in (g.genislik.astype(str) + "x" + g.yukseklik.astype(str)).value_counts().items()}
                for (s, c), g in meta.groupby(["sinif", "cesit"])}
    # 2. birebir kopya (SHA-256): ilki tutulur
    once = len(meta)
    df, dislanan = sha_ayikla(meta)
    klasor_adi = lambda d: Path(d).stem.rsplit("_", 1)[0]  # noqa: E731
    birebir_klasor = {k: int(v) for k, v in pd.Series([f"{klasor_adi(d['dosya'])} <- {klasor_adi(d['esli'].split(';')[0])}"
                                                        for d in dislanan]).value_counts().items()}
    n_birebir = len(dislanan)
    df["no"] = df.dosya.map(lambda d: int(Path(d).stem.rsplit("_", 1)[1]))
    df["klasor"] = df.cesit + "_" + df.sinif
    # 3. 8 yönlü dHash
    H8 = []
    for d in df.dosya:
        with Image.open(VERI / d) as im:
            H8.append(dhash8(ImageOps.exif_transpose(im)))
    H8 = np.stack(H8)
    ciftler, D = ic_yakin_ciftler(H8, ESIK)
    pd.DataFrame([(df.dosya[i], df.dosya[j], df.sinif[i] == df.sinif[j], df.cesit[i] == df.cesit[j], d,
                   int(np.count_nonzero(H8[i, 0] != H8[j, 0]))) for i, j, d in ciftler],
                 columns=["dosya1", "dosya2", "ayni_sinif", "ayni_cesit", "hamming_min8", "hamming_yonsuz"]
                 ).to_csv(CIKIS / "selcuk_yakin_ciftler.csv", index=False)
    en_yakin = (D.astype(np.int32) + np.eye(len(D), dtype=np.int32) * 999).min(1)
    at, bag = kopya_ayikla(df, ciftler)
    # seri (aynı çekim oturumu) bağlantıları
    E = gomme_hesapla([VERI / d for d in df.dosya], agirlik)
    np.save(CIKIS / "selcuk_gomme.npy", E)
    seri = seri_baglantilari(df.klasor.tolist(), df.no.tolist(), lambda i, j: float(E[i] @ E[j]))
    zb = zaman_baglantilari(df.klasor.tolist(), df.cekim_zamani.tolist())
    elle = elle_seri_baglari(df.dosya.tolist())
    gr = gruplar_kur(len(df), bag + seri + zb + elle)
    gr_eski_kural = gruplar_kur(len(df), bag + seri)
    gr_dhash = gruplar_kur(len(df), bag)
    df["grup"] = ["f" + g for g in gr]
    # 4. dış kaynaklarla çapraz yakın kopya (C yalnız dHash için okunur)
    et, Hd = dis_hashler()
    Dx = min8_uzaklik(H8, Hd)
    jmin = Dx.argmin(1)
    dmin = Dx[np.arange(len(df)), jmin]
    capraz = {int(i): (et[jmin[i]][0], et[jmin[i]][1], int(dmin[i])) for i in np.flatnonzero(dmin <= ESIK)}
    for i, (k, d, u) in capraz.items():
        at.setdefault(i, (f"yakin_kopya_{k}", d, u))
    for i, (n, e, u) in sorted(at.items()):
        dislanan.append({"dosya": df.dosya[i], "neden": n, "esli": e, "uzaklik": u})
    # 5. ön işleme (paralel; her görüntü bağımsız)
    kalan = [i for i in range(len(df)) if i not in at]
    with ProcessPoolExecutor(max_workers=a.isci) as ex:
        sonuc = list(ex.map(_isle, [VERI / df.dosya[i] for i in kalan], chunksize=8))
    kayit, kontrol = [], []
    kontrol_sec = set(np.random.default_rng(TOHUM).choice(kalan, size=min(48, len(kalan)), replace=False).tolist())
    for i, (g, m, bilgi, rgb) in zip(kalan, sonuc):
        r = df.loc[i]
        if not bilgi["basarili"]:
            dislanan.append({"dosya": r.dosya, "neden": "maske_basarisiz", "esli": "", "uzaklik": ""})
            continue
        ad = Path(r.dosya).stem + ".png"
        for alt, dizi in (("goruntu", g), ("maske", m)):
            (CIKIS / alt / r.sinif).mkdir(parents=True, exist_ok=True)
            Image.fromarray(dizi).save(CIKIS / alt / r.sinif / ad)
        if i in kontrol_sec:
            kontrol.append((f"{ad[:-4]} {bilgi['alan_orani']:.2f}", rgb, g))
        kayit.append({"islenmis": f"../islenmis_selcuk/goruntu/{r.sinif}/{ad}",
                      "maske": f"../islenmis_selcuk/maske/{r.sinif}/{ad}", "dosya": r.dosya, "sinif": r.sinif,
                      "kaynak": KAYNAK, "grup": r.grup, "cesit": r.cesit, "zemin": bilgi["zemin"],
                      "alan_orani": bilgi["alan_orani"]})
    son = pd.DataFrame(kayit)
    son.to_csv(CIKIS / "selcuk_veri.csv", index=False)
    pd.DataFrame(dislanan, columns=["dosya", "neden", "esli", "uzaklik"]).to_csv(CIKIS / "selcuk_dislanan.csv", index=False)
    boy = son.groupby("grup").size()
    ozet = {
        "goruntu": once, "boyutlar": boyutlar, "birebir_kopya": n_birebir, "birebir_klasor": birebir_klasor,
        "sha_benzersiz": int(len(df)), "esik_hamming_256": ESIK, "yakin_cift_8yon": len(ciftler),
        "yakin_cift_yalniz_yon_degisimiyle": int(sum(1 for i, j, _ in ciftler
                                                     if np.count_nonzero(H8[i, 0] != H8[j, 0]) > ESIK)),
        "dhash_grup_sayisi_atilanlar_dahil": int(len(set(gr_dhash))),
        "en_yakin_komsu_esik_alti_sayilari": {str(t): int((en_yakin <= t).sum()) for t in (10, 20, 30, 40, 50)},
        "seri_blok": SERI_BLOK, "seri_esik": SERI_ESIK, "seri_baglanti": len(seri), "zaman_baglanti": len(zb), "zaman_pencere_sn": ZAMAN_PENCERE_SN,
        "elle_seri_baglanti": len(elle), "grup_sayisi_yalniz_blok_gomme": int(len(set(gr_eski_kural))),
        "exif_yon_1_olmayan": int((df.exif_yon != 1).sum()), "cekim_zamani_eksik": int(df.cekim_zamani.isna().sum()),
        "grup_sayisi_dhash_seri_atilanlar_dahil": int(len(set(gr))),
        "atilan": {k: int(v) for k, v in pd.Series([d["neden"] for d in dislanan]).value_counts().items()},
        "capraz_yakin_kopya": {k: int(sum(1 for v in capraz.values() if v[0] == k)) for k in "ABCDE"},
        "capraz_en_kucuk_uzaklik": {k: (int(Dx[:, [n for n, e in enumerate(et) if e[0] == k]].min())
                                        if any(e[0] == k for e in et) else None) for k in "ABCDE"},
        "dis_goruntu_sayisi": {k: int(sum(1 for e in et if e[0] == k)) for k in "ABCDE"},
        "son_ornek": int(len(son)), "son_grup": int(son.grup.nunique()),
        "son_sinif_cesit": {f"{s}|{c}": int(n) for (s, c), n in son.groupby(["sinif", "cesit"]).size().items()},
        "son_grup_sinif_cesit": {f"{s}|{c}": int(n) for (s, c), n in son.groupby(["sinif", "cesit"]).grup.nunique().items()},
        "grup_boyu": {"medyan": float(boy.median()), "en_buyuk": int(boy.max())},
        "zemin": {k: int(v) for k, v in son.zemin.value_counts().items()},
        "alan_orani_ceyrekler": [round(float(q), 3) for q in son.alan_orani.quantile([0.1, 0.25, 0.5, 0.75, 0.9])],
    }
    (CIKIS / "selcuk_ozeti.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    kontrol_sayfasi(kontrol, CIKTI / "genel_model" / "selcuk_maske_kontrol.jpg")
    print(json.dumps(ozet, ensure_ascii=False, indent=2))


def taban(model_yolu, cikti):
    """Eğitim yok: verilen (F'yi hiç görmemiş) genel model ile F'nin tamamında doğruluk (çeşit x sınıf)."""
    import torch
    gm = importlib.import_module("09_genelleme")
    th = importlib.import_module("tahmin")
    from veri import etiketler, goruntuleri_yukle
    ck = th.kontrol_noktasi_dogrula(th.kontrol_noktasi_oku(model_yolu), model_yolu)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gm.model_kur(ck["omurga"], onegitimli=False)
    model.load_state_dict(ck["model"])
    model.to(cihaz)
    f = pd.read_csv(CIKIS / "selcuk_veri.csv")
    y = etiketler(f)
    P = gm.tahminler(model, goruntuleri_yukle(f, boyut=ck["boyut"]), ck.get("isik", False), cihaz)
    s = {"model": str(model_yolu), "F_tum": gm.metrik(f, y, P), "F_cesit": gm.cesit_bazinda(f, y, P)}
    Path(cikti).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(s["F_cesit"], ensure_ascii=False, indent=1))


def ornek_paneli():
    """F'den 8 kare (4 CBB, 4 sağlıklı; iki çeşitten eşit), üst: ham fotoğraf, alt: modele giren maskeli kırpım."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ortak import SEKIL
    df = pd.read_csv(CIKIS / "selcuk_veri.csv")
    rng = np.random.default_rng(TOHUM)
    sec = []
    for s in ("adi_yaprak_yanikligi", "saglikli"):
        for c in ("akbulut", "ustun42"):
            g = df[(df.sinif == s) & (df.cesit == c)]
            sec += [g.iloc[i] for i in rng.choice(len(g), 2, replace=False)]
    fig, ax = plt.subplots(2, 8, figsize=(19, 5.4))
    adlar = {"adi_yaprak_yanikligi": "CBB", "saglikli": "Sağlıklı"}
    cesit_ad = {"akbulut": "Akbulut", "ustun42": "Üstün42"}
    for j, r in enumerate(sec):
        ax[0, j].imshow(np.asarray(Image.open(VERI / r.dosya).convert("RGB")))
        ax[1, j].imshow(np.asarray(Image.open(VERI / "islenmis" / r.islenmis)))
        ax[0, j].set_title(f"{adlar[r.sinif]} — {cesit_ad[r.cesit]}", fontsize=9)
        ax[1, j].set_xlabel(Path(r.dosya).stem, fontsize=7)
        for k in (0, 1):
            ax[k, j].set_xticks([])
            ax[k, j].set_yticks([])
    ax[0, 0].set_ylabel("Ham fotoğraf", fontsize=8)
    ax[1, 0].set_ylabel("Maskeli kırpım\n(model girdisi)", fontsize=8)
    fig.suptitle("Kaynak F: Selçuk Üniversitesi Bean Bacterial Blight Disease Dataset (Türkiye; Akbulut ve Üstün42 "
                 "çeşitleri; Xanthomonas axonopodis pv. phaseoli)\nAtıf: " + ATIF + " — görüntüler yeniden dağıtılmaz",
                 fontsize=8)
    fig.tight_layout()
    (SEKIL / "rapor").mkdir(parents=True, exist_ok=True)
    fig.savefig(SEKIL / "rapor" / "sekil_selcuk_ornekleri.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kaynak", help="Bean_Bacterial_Blight_Disease_Dataset klasörü (yalnız okunur)")
    p.add_argument("--panel", action="store_true")
    p.add_argument("--isci", type=int, default=10, help="ön işleme için paralel süreç sayısı")
    p.add_argument("--gomme_agirlik", help="EfficientNet-B0 ImageNet ağırlık dosyası (yoksa torch hub önbelleği)")
    p.add_argument("--taban", help="F'yi görmemiş bir model.pt ile F'nin tamamında doğruluk (eğitim yok)")
    a = p.parse_args()
    if a.panel:
        ornek_paneli()
    elif a.taban:
        taban(a.taban, CIKTI / "genel_model" / "selcuk_taban.json")
    else:
        if not a.kaynak:
            p.error("--kaynak gerekli")
        main(a)
