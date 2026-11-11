"""data/etiketler.csv'deki goruntuler icin 256-bit dHash ile yakin-kopya taramasi (dosya silmez).
Yakin kopyalari (ayni yapragin seri cekimleri, yeniden kodlanmis kopyalar) baglantili bilesenlere ayirir ve
etiketler.csv'ye `grup` sutunu ekler. Egitim/dogrulama/test ayriminda ayni gruptaki goruntuler AYNI bolume
konmalidir (sizinti onlemi).
Kullanim: python -I yakin_kopya.py <proje_koku> [esik=20]
"""
import csv
import os
import sys
from collections import Counter

from PIL import Image

N = 16  # 16x16 = 256 bit


def dhash(yol):
    with Image.open(yol) as im:
        g = im.convert("L").resize((N + 1, N), Image.LANCZOS)
    p = g.tobytes()
    bit = 0
    for y in range(N):
        for x in range(N):
            if p[y * (N + 1) + x] > p[y * (N + 1) + x + 1]:
                bit |= 1 << (y * N + x)
    return bit


def main():
    kok = sys.argv[1]
    esik = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    data = os.path.join(kok, "data")
    yol_csv = os.path.join(data, "etiketler.csv")
    with open(yol_csv, encoding="utf-8") as f:
        okuyucu = csv.DictReader(f)
        alanlar = [a for a in okuyucu.fieldnames if a != "grup"]
        satirlar = list(okuyucu)
    h = [dhash(os.path.join(data, r["dosya"])) for r in satirlar]
    ebeveyn = list(range(len(h)))

    def bul(i):
        while ebeveyn[i] != i:
            ebeveyn[i] = ebeveyn[ebeveyn[i]]
            i = ebeveyn[i]
        return i

    ciftler = []
    for i in range(len(h)):
        hi = h[i]
        for j in range(i + 1, len(h)):
            d = (hi ^ h[j]).bit_count()
            if d <= esik:
                ayni = satirlar[i]["sinif"] == satirlar[j]["sinif"]
                ciftler.append((satirlar[i]["dosya"], satirlar[j]["dosya"], ayni, d))
                ebeveyn[bul(i)] = bul(j)
    kokler = {}
    for i, r in enumerate(satirlar):
        k = bul(i)
        kokler.setdefault(k, len(kokler))
        r["grup"] = f"g{kokler[k]:05d}"
    with open(yol_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar + ["grup"])
        w.writeheader()
        w.writerows(satirlar)
    with open(os.path.join(data, "yakin_kopyalar.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dosya1", "dosya2", "ayni_sinif", "hamming_256"])
        w.writerows(ciftler)
    c = Counter(("ayni_sinif" if a else "farkli_sinif") for _, _, a, _ in ciftler)
    print(f"esik={esik}/256  ciftler:", dict(c))
    boy = Counter(Counter(r["grup"] for r in satirlar).values())
    print("grup sayisi:", len(kokler), " grup boyu dagilimi:", sorted(boy.items())[:15], "en buyuk:", max(boy))
    for a, b, s, d in [x for x in ciftler if not x[2]][:20]:
        print("FARKLI", d, a, b)


if __name__ == "__main__":
    main()
