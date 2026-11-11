"""Indirilen arsivleri data/raw/<sinif>/ altina duzenler; PIL ile dogrular, birebir kopyalari (sha256) ayiklar.
Arsiv uyelerinin yollarini hic kullanmaz (zip-slip yok); dosyalar kendi verdigimiz adlarla yazilir,
icerik baytlari degistirilmeden kopyalanir. Hicbir arsiv icerigi calistirilmaz.
Kullanim: python -I duzenle.py <proje_koku>
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict

from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000
GORUNTU_UZ = (".jpg", ".jpeg", ".png")

LIS_CCBY = "CC BY 4.0"
LIS_EMB = "CC BY-NC 4.0 (Embrapa)"
URL_YK = "https://data.mendeley.com/datasets/ykvcrjffzd/1"
URL_KM = "https://data.mendeley.com/datasets/3km9d246z2/3"
URL_EMB = "https://doi.org/10.48432/XA1OVL"

# 3km9d246z2 bean_leaves klasorleri -> hedef
KM_ESLEME = {
    "halo_blight": ("raw", "hale_yanikligi"),
    "healthy_leaves": ("raw", "saglikli"),
    "angular_leaf_spot": ("raw_ek", "kose_leke_als"),
    "cercospora_leaf_spot": ("raw_ek", "cercospora_yaprak_lekesi"),
    "fungi_pathogens": ("raw_ek", "fungal_patojenler"),
    "tobacco_caterpillar": ("raw_ek", "tutun_tirtili_zarari"),
    "yellow_mosaic": ("raw_ek", "sari_mozaik"),
}


def guvenli_ad(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def adaylar(kok):
    ind = os.path.join(kok, "data", "_indirilen")
    # 1) Mendeley ykvcrjffzd (Bangladesh fasulye + borulce): yalnizca Bean/Blight ve Bean/Fresh Leaf
    yk = os.path.join(ind, "mendeley_ykvcrjffzd", "Bean")
    for alt, sinif in (("Blight", "adi_yaprak_yanikligi"), ("Fresh Leaf", "saglikli")):
        d = os.path.join(yk, alt)
        for ad in sorted(os.listdir(d)):
            if ad.lower().endswith(GORUNTU_UZ):
                with open(os.path.join(d, ad), "rb") as f:
                    yield dict(veri=f.read(), ust="raw", sinif=sinif, kaynak="mendeley_ykvcrjffzd",
                               kaynak_yol=f"Bean/{alt}/{ad}", lisans=LIS_CCBY, url=URL_YK,
                               onek="bdcp", ad=ad)
    # 2) Mendeley 3km9d246z2 bean_leaves zip
    kmd = os.path.join(ind, "mendeley_3km9d246z2")
    for zad in sorted(os.listdir(kmd)):
        if not (zad.startswith("bean_leaves") and zad.endswith(".zip")):
            continue
        with zipfile.ZipFile(os.path.join(kmd, zad)) as z:
            for i in z.infolist():
                if i.is_dir() or not i.filename.lower().endswith(GORUNTU_UZ):
                    continue
                parca = i.filename.split("/")
                if len(parca) != 3:
                    print("beklenmeyen yol, atlandi:", i.filename)
                    continue
                ust, sinif = KM_ESLEME[parca[1]]
                yield dict(veri=z.read(i), ust=ust, sinif=sinif, kaynak="mendeley_3km9d246z2",
                           kaynak_yol=f"{zad}!{i.filename}", lisans=LIS_CCBY, url=URL_KM,
                           onek="bhd", ad=parca[2])
    # 3) Embrapa PDDB (Redape Dataverse)
    emd = os.path.join(ind, "embrapa_pddb")
    for zad, sinif, onek in (
        ("Feijao (Dry bean) - Crestamento Bacteriano (Common bacterial blight) - 1.zip", "adi_yaprak_yanikligi", "pddb_cbb"),
        ("Feijao (Dry bean) - Saudavel (Healthy) - 1.zip", "saglikli", "pddb_saglikli"),
    ):
        with zipfile.ZipFile(os.path.join(emd, zad)) as z:
            for i in z.infolist():
                if i.is_dir() or not i.filename.lower().endswith(GORUNTU_UZ):
                    continue
                yield dict(veri=z.read(i), ust="raw", sinif=sinif, kaynak="embrapa_pddb",
                           kaynak_yol=f"{zad}!{i.filename}", lisans=LIS_EMB, url=URL_EMB,
                           onek=onek, ad=os.path.basename(i.filename))


def dogrula(veri):
    try:
        with Image.open(io.BytesIO(veri)) as im:
            im.verify()
        with Image.open(io.BytesIO(veri)) as im:
            im.load()
            return dict(bicim=im.format, mod=im.mode, genislik=im.width, yukseklik=im.height)
    except Exception as e:  # noqa: BLE001
        return {"hata": f"{type(e).__name__}: {e}"}


def main():
    kok = sys.argv[1]
    data = os.path.join(kok, "data")
    for ust in ("raw", "raw_ek"):
        p = os.path.join(data, ust)
        if os.path.isdir(p):
            shutil.rmtree(p)
    kayit, bozuk, kopya = [], [], []
    hash_ilk = {}  # sha -> kayit
    hash_siniflar = defaultdict(set)
    tum = []
    for a in adaylar(kok):
        sha = hashlib.sha256(a["veri"]).hexdigest()
        a["sha256"] = sha
        tum.append({k: v for k, v in a.items() if k != "veri"})
        hash_siniflar[sha].add((a["ust"], a["sinif"]))
        bilgi = dogrula(a["veri"])
        if "hata" in bilgi:
            bozuk.append({**{k: v for k, v in a.items() if k != "veri"}, **bilgi})
            continue
        if sha in hash_ilk:
            kopya.append({"kaynak_yol": a["kaynak_yol"], "sinif": a["sinif"],
                          "ilk": hash_ilk[sha]["kaynak_yol"], "ilk_sinif": hash_ilk[sha]["sinif"]})
            continue
        uz = {"JPEG": ".jpg", "PNG": ".png"}.get(bilgi["bicim"], os.path.splitext(a["ad"])[1].lower())
        yeni_ad = f"{a['onek']}__{guvenli_ad(os.path.splitext(a['ad'])[0])}{uz}"
        hedef_kl = os.path.join(data, a["ust"], a["sinif"])
        os.makedirs(hedef_kl, exist_ok=True)
        n = 1
        temel = yeni_ad
        while os.path.exists(os.path.join(hedef_kl, yeni_ad)):
            n += 1
            yeni_ad = temel.replace(uz, f"_{n}{uz}")
        with open(os.path.join(hedef_kl, yeni_ad), "wb") as f:
            f.write(a["veri"])
        r = {"dosya": f"{a['ust']}/{a['sinif']}/{yeni_ad}", "sinif": a["sinif"], "kume": a["ust"],
             "kaynak": a["kaynak"], "kaynak_yol": a["kaynak_yol"], "kaynak_url": a["url"],
             "lisans": a["lisans"], "sha256": sha, **bilgi}
        hash_ilk[sha] = r
        kayit.append(r)

    # Siniflar arasi catisan kopyalar: hepsini cikar
    catisan = {s for s, kl in hash_siniflar.items() if len(kl) > 1}
    if catisan:
        yeni = []
        for r in kayit:
            if r["sha256"] in catisan:
                os.remove(os.path.join(data, r["dosya"]))
            else:
                yeni.append(r)
        kayit = yeni

    alanlar = ["dosya", "sinif", "kume", "kaynak", "kaynak_yol", "kaynak_url", "lisans", "sha256",
               "bicim", "mod", "genislik", "yukseklik"]
    with open(os.path.join(data, "etiketler.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar)
        w.writeheader()
        w.writerows(kayit)

    rapor = {
        "toplam_aday": len(tum),
        "bozuk": bozuk,
        "kopya_sayisi": len(kopya),
        "kopyalar": kopya,
        "siniflar_arasi_catisan_hash": len(catisan),
        "sinif_sayilari": Counter(f"{r['kume']}/{r['sinif']}" for r in kayit),
        "sinif_kaynak_sayilari": Counter(f"{r['kume']}/{r['sinif']} <- {r['kaynak']}" for r in kayit),
        "bicimler": Counter(f"{r['sinif']}:{r['bicim']}/{r['mod']}" for r in kayit),
        "cozunurluk": Counter(f"{r['sinif']} [{r['kaynak']}]: {r['genislik']}x{r['yukseklik']}" for r in kayit),
    }
    with open(os.path.join(data, "temizlik_raporu.json"), "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rapor.items() if k not in ("kopyalar", "cozunurluk")},
                     ensure_ascii=False, indent=1))
    top = Counter()
    for k, v in rapor["cozunurluk"].items():
        top[k] = v
    for k, v in top.most_common(40):
        print(f"{v:5d} {k}")


if __name__ == "__main__":
    main()
