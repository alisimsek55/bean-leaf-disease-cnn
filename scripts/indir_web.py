"""Ek (dorduncu) kaynak: PlantWild web goruntuleri -> data/raw_web/
Kaynak: Wei T. ve ark. "Benchmarking In-the-wild Multimodal Plant Disease Recognition and A Versatile
Baseline" (ACM MM 2024), PlantWild veri seti, https://huggingface.co/datasets/uqtwei2/PlantWild
Lisans: CC BY-NC-ND 4.0 (ticari olmayan; turetilmis goruntuler dagitilmaz, yalniz yerel egitimde kullanilir).

Arsivin tamami (2.7 GB + 1.6 GB) indirilmez: HTTP Range ile yalniz ilgili klasorlerin uyeleri cekilir.
Her uye zipfile tarafindan CRC32 ile dogrulanir; SHA-256 kaydedilir. Arsiv ici yollar dosya yazarken
KULLANILMAZ: hedef ad `pw<surum>__<sira>.<uzanti>` olarak uretilir ve guvenli_yol ile klasor icinde tutulur.
Kullanim: python -I scripts/indir_web.py <proje_koku>
"""
import csv
import hashlib
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indir import guvenli_yol  # noqa: E402
from uzak_zip_listele import UzakDosya  # noqa: E402

HF = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/"
# (arsiv, arsiv ici klasor, hedef sinif, onek)
SECIM = [
    ("plantwild.zip", "plantwild/images/bean halo blight/", "hale_yanikligi", "pw1"),
    ("plantwild_v2.zip", "plantwild_v2/bean halo blight/", "hale_yanikligi", "pw2"),
    ("plantwild.zip", "plantwild/images/bean leaf/", "saglikli", "pw1"),
]
IZINLI_UZANTI = {".jpg", ".jpeg", ".png"}
MAKS_UYE = 30 * 1024 * 1024  # bir goruntu icin ust sinir (zip bombasina karsi)


def main():
    kok = sys.argv[1]
    hedef_kok = os.path.join(kok, "data", "raw_web")
    kayit = []
    arsivler = {}
    for arsiv, klasor, sinif, onek in SECIM:
        if arsiv not in arsivler:
            arsivler[arsiv] = zipfile.ZipFile(io.BufferedReader(UzakDosya(HF + arsiv), buffer_size=1 << 20))
        z = arsivler[arsiv]
        uyeler = sorted((i for i in z.infolist() if i.filename.startswith(klasor) and not i.is_dir()),
                        key=lambda i: i.filename)
        hk = os.path.join(hedef_kok, sinif)
        os.makedirs(hk, exist_ok=True)
        for n, i in enumerate(uyeler):
            uz = os.path.splitext(i.filename)[1].lower()
            if uz not in IZINLI_UZANTI or i.file_size > MAKS_UYE:
                print("atlandi:", i.filename)
                continue
            ad = f"{onek}__{sinif[:4]}_{n:04d}{uz}"
            yol = guvenli_yol(hk, ad)
            veri = z.read(i)  # CRC32 dogrulamasi zipfile icinde yapilir (bozuksa BadZipFile)
            with open(yol, "wb") as f:
                f.write(veri)
            kayit.append({"dosya": f"raw_web/{sinif}/{ad}", "sinif": sinif, "kaynak": "plantwild_web",
                          "kaynak_yol": f"{arsiv}:{i.filename}", "kaynak_url": "https://huggingface.co/datasets/uqtwei2/PlantWild",
                          "lisans": "CC BY-NC-ND 4.0", "sha256": hashlib.sha256(veri).hexdigest(),
                          "crc32": f"{i.CRC:08x}"})
        print(f"{arsiv}:{klasor} -> {sinif}: {len(uyeler)} dosya")
    with open(os.path.join(hedef_kok, "etiketler_web.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kayit[0]))
        w.writeheader()
        w.writerows(kayit)
    print("toplam:", len(kayit))


if __name__ == "__main__":
    main()
