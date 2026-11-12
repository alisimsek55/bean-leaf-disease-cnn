"""Veri kaynaklarini indirir. Her kaynak kendi klasorune: data/_indirilen/<kaynak>/
Kullanim: python -I indir.py <proje_koku> <kaynak>
kaynaklar: embrapa_pddb | mendeley_ykvcrjffzd | mendeley_3km9d246z2
"""
import hashlib
import json
import os
import sys
import time

import requests

UA = {"User-Agent": "research-dataset-client/1.0"}
S = requests.Session()
S.headers.update(UA)


class IndirmeHatasi(Exception):
    """Indirme/dogrulama basarisiz (boyut, sha256, ag hatasi)."""


YASAK_KARAKTER = set('<>:"|?*')


def guvenli_ad(ad):
    """Uzaktan gelen dosya adini dogrular; yalnizca duz bir dosya adi (basename) kabul eder.

    Bos ad, mutlak yol, surucu harfi (':'), her iki tur ayirici (slash ve ters slash),'.'/'..' bilesenleri,
    ayrilmis/kontrol karakterleri ve sondaki nokta/bosluk reddedilir (ValueError)."""
    if not isinstance(ad, str) or not ad.strip():
        raise ValueError(f"gecersiz dosya adi (bos): {ad!r}")
    if "/" in ad or "\\" in ad:
        raise ValueError(f"gecersiz dosya adi (yol ayirici): {ad!r}")
    if ad in (".", ".."):
        raise ValueError(f"gecersiz dosya adi (..): {ad!r}")
    if any(c in YASAK_KARAKTER or ord(c) < 32 for c in ad):
        raise ValueError(f"gecersiz dosya adi (yasak karakter): {ad!r}")
    if ad != ad.rstrip(". "):
        raise ValueError(f"gecersiz dosya adi (sonda nokta/bosluk): {ad!r}")
    return ad


def guvenli_yol(klasor, ad):
    """klasor/ad yolunu uretir; sonucun klasorun icinde kaldigini cozumlenmis yolla dogrular."""
    ad = guvenli_ad(ad)
    kok = os.path.realpath(klasor)
    yol = os.path.realpath(os.path.join(kok, ad))
    if os.path.commonpath([kok, yol]) != kok or yol == kok:
        raise ValueError(f"hedef yol klasor disina cikiyor: {ad!r}")
    return os.path.join(klasor, ad)


def sha256_dosya(yol):
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def sil(yol):
    try:
        os.remove(yol)
    except FileNotFoundError:
        pass


def getir(url, gecici):
    """url'yi gecici dosyaya yazar (HTTP istegi yapan tek yer; testlerde degistirilir)."""
    with S.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(gecici, "wb") as f:
            for parca in r.iter_content(1 << 20):
                f.write(parca)


def gecerli_mi(yol, beklenen_boyut, sha256):
    if beklenen_boyut and os.path.getsize(yol) != beklenen_boyut:
        return False
    if sha256 and sha256_dosya(yol).lower() != sha256.lower():
        return False
    return True


def indir(url, hedef, beklenen_boyut=None, sha256=None):
    if os.path.exists(hedef):
        # Ozet varsa mevcut dosya her zaman hash'lenir; boyut tek basina yeterli sayilmaz.
        if (sha256 or beklenen_boyut) and gecerli_mi(hedef, beklenen_boyut, sha256):
            print("zaten var:", hedef)
            return
        print("mevcut dosya gecersiz, yeniden indiriliyor:", hedef)
        sil(hedef)
    gecici = hedef + ".part"
    sil(gecici)
    try:
        getir(url, gecici)
        if beklenen_boyut and os.path.getsize(gecici) != beklenen_boyut:
            raise IndirmeHatasi(f"boyut uyusmuyor: {hedef}")
        if sha256 and sha256_dosya(gecici).lower() != sha256.lower():
            raise IndirmeHatasi(f"sha256 uyusmuyor: {hedef}")
        os.replace(gecici, hedef)
    except BaseException:
        sil(gecici)  # eksik/gecersiz parca birakma
        raise
    print("indirildi:", hedef, os.path.getsize(hedef))


def indir_tekrarli(url, hedef, beklenen_boyut=None, sha256=None, deneme=4, bekle=5):
    """indir'i yeniden dener; son denemeden sonra IndirmeHatasi firlatir (sessizce gecmez)."""
    son = None
    for i in range(deneme):
        try:
            indir(url, hedef, beklenen_boyut, sha256)
            return
        except Exception as e:  # noqa: BLE001
            son = e
            print("hata, tekrar:", e)
            if i < deneme - 1:
                time.sleep(bekle * (i + 1))
    raise IndirmeHatasi(f"{deneme} denemeden sonra indirilemedi: {hedef}: {son}")


def embrapa(kok):
    d = os.path.join(kok, "data", "_indirilen", "embrapa_pddb")
    os.makedirs(d, exist_ok=True)
    # Redape Dataverse, doi:10.48432/XA1OVL; dosya kimlikleri API'den alindi
    dosyalar = {
        5972: "Feijao (Dry bean) - Crestamento Bacteriano (Common bacterial blight) - 1.zip",
        6015: "Feijao (Dry bean) - Saudavel (Healthy) - 1.zip",
    }
    meta = S.get("https://www.redape.dados.embrapa.br/api/datasets/:persistentId/",
                 params={"persistentId": "doi:10.48432/XA1OVL"}, timeout=120).json()
    with open(os.path.join(d, "_dataverse_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    boyutlar = {x["dataFile"]["id"]: x["dataFile"] for x in meta["data"]["latestVersion"]["files"]}
    for fid, ad in dosyalar.items():
        df = boyutlar[fid]
        assert df["filename"] == ad, df["filename"]
        indir(f"https://www.redape.dados.embrapa.br/api/access/datafile/{fid}", guvenli_yol(d, ad),
              df["filesize"])


def mendeley_meta(kimlik, d):
    meta = S.get(f"https://data.mendeley.com/public-api/datasets/{kimlik}", timeout=120).json()
    with open(os.path.join(d, "_mendeley_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def ykvcrjffzd(kok):
    d = os.path.join(kok, "data", "_indirilen", "mendeley_ykvcrjffzd")
    os.makedirs(d, exist_ok=True)
    meta = mendeley_meta("ykvcrjffzd", d)
    klasorler = S.get("https://data.mendeley.com/public-api/datasets/ykvcrjffzd/folders/1", timeout=120).json()
    with open(os.path.join(d, "_mendeley_folders.json"), "w", encoding="utf-8") as f:
        json.dump(klasorler, f, ensure_ascii=False, indent=1)
    ad = {k["id"]: k for k in klasorler}
    bean = [k["id"] for k in klasorler if k["name"] == "Bean"][0]
    istenen = {k["id"]: k["name"] for k in klasorler if k.get("parent_id") == bean and k["name"] in ("Blight", "Fresh Leaf")}
    print("klasorler:", istenen)
    basarisiz = []
    for fl in meta["files"]:
        if fl["folder_id"] not in istenen:
            continue
        hk = os.path.join(d, "Bean", istenen[fl["folder_id"]])
        os.makedirs(hk, exist_ok=True)
        hedef = guvenli_yol(hk, fl["filename"])
        try:
            indir_tekrarli(fl["content_details"]["download_url"], hedef, fl["size"],
                           fl["content_details"].get("sha256_hash"))
        except IndirmeHatasi as e:
            basarisiz.append(str(e))
        time.sleep(0.3)
    if basarisiz:
        raise SystemExit(f"{len(basarisiz)} dosya indirilemedi: " + "; ".join(basarisiz))


def km9(kok):
    d = os.path.join(kok, "data", "_indirilen", "mendeley_3km9d246z2")
    os.makedirs(d, exist_ok=True)
    meta = mendeley_meta("3km9d246z2", d)
    for fl in meta["files"]:
        if fl["filename"].startswith("bean_leaves"):
            indir(fl["content_details"]["download_url"], guvenli_yol(d, fl["filename"]), fl["size"],
                  fl["content_details"].get("sha256_hash"))


if __name__ == "__main__":
    kok, kaynak = sys.argv[1], sys.argv[2]
    {"embrapa_pddb": embrapa, "mendeley_ykvcrjffzd": ykvcrjffzd, "mendeley_3km9d246z2": km9}[kaynak](kok)
