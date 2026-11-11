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


def indir(url, hedef, beklenen_boyut=None, sha256=None):
    if os.path.exists(hedef) and beklenen_boyut and os.path.getsize(hedef) == beklenen_boyut:
        print("zaten var:", hedef)
        return
    gecici = hedef + ".part"
    with S.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        h = hashlib.sha256()
        with open(gecici, "wb") as f:
            for parca in r.iter_content(1 << 20):
                f.write(parca)
                h.update(parca)
    if beklenen_boyut and os.path.getsize(gecici) != beklenen_boyut:
        raise SystemExit(f"boyut uyusmuyor: {hedef}")
    if sha256 and h.hexdigest() != sha256:
        raise SystemExit(f"sha256 uyusmuyor: {hedef}")
    os.replace(gecici, hedef)
    print("indirildi:", hedef, os.path.getsize(hedef))


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
        indir(f"https://www.redape.dados.embrapa.br/api/access/datafile/{fid}", os.path.join(d, ad),
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
    for fl in meta["files"]:
        if fl["folder_id"] not in istenen:
            continue
        hk = os.path.join(d, "Bean", istenen[fl["folder_id"]])
        os.makedirs(hk, exist_ok=True)
        hedef = os.path.join(hk, os.path.basename(fl["filename"]))
        if os.path.exists(hedef) and os.path.getsize(hedef) == fl["size"]:
            continue
        for deneme in range(4):
            try:
                indir(fl["content_details"]["download_url"], hedef, fl["size"], fl["content_details"].get("sha256_hash"))
                break
            except Exception as e:  # noqa: BLE001
                print("hata, tekrar:", e)
                time.sleep(5 * (deneme + 1))
        time.sleep(0.3)


def km9(kok):
    d = os.path.join(kok, "data", "_indirilen", "mendeley_3km9d246z2")
    os.makedirs(d, exist_ok=True)
    meta = mendeley_meta("3km9d246z2", d)
    for fl in meta["files"]:
        if fl["filename"].startswith("bean_leaves"):
            indir(fl["content_details"]["download_url"], os.path.join(d, fl["filename"]), fl["size"],
                  fl["content_details"].get("sha256_hash"))


if __name__ == "__main__":
    kok, kaynak = sys.argv[1], sys.argv[2]
    {"embrapa_pddb": embrapa, "mendeley_ykvcrjffzd": ykvcrjffzd, "mendeley_3km9d246z2": km9}[kaynak](kok)
