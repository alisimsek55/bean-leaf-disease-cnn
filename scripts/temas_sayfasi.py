"""Her siniftan rastgele goruntulerle numarali bir temas sayfasi (contact sheet) uretir; gorsel etiket kontrolu icin.
Kullanim: python -I temas_sayfasi.py <proje_koku> <cikti_klasoru> [n=8] [tohum=42] [kaynak_filtresi]
"""
import csv
import os
import random
import sys
from collections import defaultdict

from PIL import Image, ImageDraw


def main():
    kok, cikti = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tohum = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    filtre = sys.argv[5] if len(sys.argv) > 5 else None
    data = os.path.join(kok, "data")
    os.makedirs(cikti, exist_ok=True)
    siniflar = defaultdict(list)
    with open(os.path.join(data, "etiketler.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if filtre and r["kaynak"] != filtre:
                continue
            siniflar[f"{r['kume']}_{r['sinif']}"].append(r["dosya"])
    T = 420
    for s, dosyalar in sorted(siniflar.items()):
        secim = random.Random(tohum).sample(dosyalar, min(n, len(dosyalar)))
        sut = 4
        sat = (len(secim) + sut - 1) // sut
        sayfa = Image.new("RGB", (sut * T, sat * (T + 20)), "white")
        c = ImageDraw.Draw(sayfa)
        for i, d in enumerate(secim):
            with Image.open(os.path.join(data, d)) as im:
                im = im.convert("RGB")
                im.thumbnail((T, T))
            x, y = (i % sut) * T, (i // sut) * (T + 20)
            sayfa.paste(im, (x, y + 20))
            c.text((x + 4, y + 4), f"{i + 1}: {os.path.basename(d)[:55]}", fill="black")
        ad = os.path.join(cikti, f"{s}{'_' + filtre if filtre else ''}.jpg")
        sayfa.save(ad, quality=88)
        print(ad)
        for i, d in enumerate(secim):
            print(f"   {i + 1}: {d}")


if __name__ == "__main__":
    main()
