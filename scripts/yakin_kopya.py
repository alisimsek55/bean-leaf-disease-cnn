"""data/etiketler.csv'deki goruntuler icin yakin-kopya ve seri cekim gruplamasi (dosya silmez).
Iki tur baglanti ayni union-find icinde birlestirilir ve etiketler.csv'ye `grup` sutunu yazilir:
  1. 256-bit dHash, Hamming uzakligi <= esik (yeniden kodlanmis kopyalar, neredeyse ayni kareler).
  2. Elle dogrulanmis ayni-yaprak ciftleri (ELLE_BIRLESTIR).
  3. Seri cekim (burst): dosya adinda kamera zaman damgasi `IMG_<YYYYMMDD>_<HHMMSS>` varsa, ayni
     (kaynak, sinif) icinde zaman damgalari arasinda en fazla `pencere` saniye olan kareler ayni gruba
     baglanir (zincirleme). Ayni yapragin ard arda cekilen kareleri dHash esigini asabildigi icin
     yalnizca dHash ile ayri gruplara dusuyor ve egitim/dogrulama arasinda sizinti olusturuyordu.
     Zaman damgasi okunamayan dosyalarda (ornegin `Good_Leaf_671.png`, `DSC_0629.jpg`) yalnizca dHash
     kullanilir: bu adlardaki sayilar veri seti/kamera sira numarasidir, cekim zamani degildir.
Egitim/dogrulama/test ayriminda ayni gruptaki goruntuler AYNI bolume konmalidir (sizinti onlemi).
Kullanim: python -I yakin_kopya.py <proje_koku> [esik=20] [pencere_sn=2]
"""
import csv
import os
import re
import sys
from collections import Counter
from datetime import datetime

N = 16  # 16x16 = 256 bit
ESIK = 20  # dHash Hamming esigi (256 bit uzerinden)
PENCERE_SN = 2  # seri cekim penceresi (saniye)

# Kaynak A (Mendeley ykvcrjffzd) dosya adlari: bdcp__IMG_20240105_235312_641.jpg
# (tarih _ saat-dakika-saniye _ milisaniye). Milisaniye alani (tam 3 basamak) varsa okunur ve seri
# penceresi tam zaman farkiyla karsilastirilir; yoksa 0 ms sayilir.
# Elle dogrulanmis ayni-yaprak ciftleri (dHash/seri ile yakalanmayan): Embrapa'daki iki saglikli kare
# ayni yapragin iki cekimidir (bkz. docs/DATASET.tr.md).
ELLE_BIRLESTIR = [("raw/saglikli/pddb_saglikli__1.jpg", "raw/saglikli/pddb_saglikli__2.jpg")]

SERI_DESENI = re.compile(r"(?<![0-9A-Za-z])IMG_(\d{8})_(\d{6})(?:_(\d{3})(?!\d))?(?!\d)")


def cekim_zamani(dosya):
    """Dosya adindaki `IMG_<YYYYMMDD>_<HHMMSS>[_<ms>]` damgasini datetime olarak dondurur (ms yoksa 0);
    yoksa / gecersizse None."""
    m = SERI_DESENI.search(os.path.basename(str(dosya).replace("\\", "/")))
    if not m:
        return None
    try:
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return t.replace(microsecond=int(m.group(3) or 0) * 1000)


def saniye_anahtari(dosya):
    """Cekim saniyesi anahtari ('YYYYMMDD_HHMMSS') ya da None."""
    t = cekim_zamani(dosya)
    return t.strftime("%Y%m%d_%H%M%S") if t else None


def seri_baglantilari(satirlar, pencere=PENCERE_SN, katman=("kaynak", "sinif")):
    """Seri cekim baglantilari: (i, j) indeks ciftleri.
    Her katmanda (varsayilan: ayni kaynak + ayni sinif) zaman damgasina gore siralanir; ardisik iki kare
    arasindaki tam fark (milisaniye dahil) <= `pencere` saniye ise baglanir. Zincirleme oldugu icin 0-1-2-3 sn'lik bir seri tek
    bileseni olusturur. Zaman damgasi olmayan satirlar baglanmaz."""
    katmanlar = {}
    for i, r in enumerate(satirlar):
        t = cekim_zamani(r["dosya"])
        if t is not None:
            katmanlar.setdefault(tuple(r.get(k, "") for k in katman), []).append((t, i))
    ciftler = []
    for ogeler in katmanlar.values():
        ogeler.sort()
        for (t0, i0), (t1, i1) in zip(ogeler, ogeler[1:]):
            if (t1 - t0).total_seconds() <= pencere:
                ciftler.append((i0, i1))
    return ciftler


def elle_baglantilar(satirlar, ciftler=ELLE_BIRLESTIR):
    """ELLE_BIRLESTIR listesindeki dosya ciftleri icin (i, j) baglantilari (ikisi de varsa)."""
    indeks = {r["dosya"]: i for i, r in enumerate(satirlar)}
    return [(indeks[a], indeks[b]) for a, b in ciftler if a in indeks and b in indeks]


def dhash_baglantilari(hashler, esik=ESIK):
    """dHash baglantilari: Hamming uzakligi <= esik olan (i, j, uzaklik) ucluleri (i < j)."""
    ciftler = []
    for i in range(len(hashler)):
        hi = hashler[i]
        for j in range(i + 1, len(hashler)):
            d = (hi ^ hashler[j]).bit_count()
            if d <= esik:
                ciftler.append((i, j, d))
    return ciftler


def grup_etiketleri(n, baglantilar):
    """Union-find ile baglantili bilesenleri bulur; ilk gorulme sirasina gore 'g00000' etiketleri dondurur."""
    ebeveyn = list(range(n))

    def bul(i):
        while ebeveyn[i] != i:
            ebeveyn[i] = ebeveyn[ebeveyn[i]]
            i = ebeveyn[i]
        return i

    for i, j in baglantilar:
        a, b = bul(i), bul(j)
        if a != b:
            ebeveyn[a] = b
    kokler = {}
    etiketler = []
    for i in range(n):
        k = bul(i)
        kokler.setdefault(k, len(kokler))
        etiketler.append(f"g{kokler[k]:05d}")
    return etiketler


def dhash(yol):
    from PIL import Image  # yalnizca goruntu okurken gerekli; saf fonksiyonlar PIL'siz test edilebilir

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
    esik = int(sys.argv[2]) if len(sys.argv) > 2 else ESIK
    pencere = float(sys.argv[3]) if len(sys.argv) > 3 else PENCERE_SN
    data = os.path.join(kok, "data")
    yol_csv = os.path.join(data, "etiketler.csv")
    with open(yol_csv, encoding="utf-8") as f:
        okuyucu = csv.DictReader(f)
        alanlar = [a for a in okuyucu.fieldnames if a != "grup"]
        satirlar = list(okuyucu)
    h = [dhash(os.path.join(data, r["dosya"])) for r in satirlar]
    dh = dhash_baglantilari(h, esik)
    seri = seri_baglantilari(satirlar, pencere)
    elle = elle_baglantilar(satirlar)
    yalniz_dhash = grup_etiketleri(len(satirlar), [(i, j) for i, j, _ in dh] + elle)
    etiketler = grup_etiketleri(len(satirlar), [(i, j) for i, j, _ in dh] + elle + seri)
    for r, g in zip(satirlar, etiketler):
        r["grup"] = g
    ciftler = [(satirlar[i]["dosya"], satirlar[j]["dosya"], satirlar[i]["sinif"] == satirlar[j]["sinif"], d)
               for i, j, d in dh]
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
    damgali = sum(cekim_zamani(r["dosya"]) is not None for r in satirlar)
    print(f"seri cekim: pencere={pencere} sn, zaman damgali goruntu={damgali}, seri baglantisi={len(seri)}")
    print("grup sayisi: yalniz dHash", len(set(yalniz_dhash)), "-> dHash + seri", len(set(etiketler)))
    boy = Counter(Counter(r["grup"] for r in satirlar).values())
    print(" grup boyu dagilimi:", sorted(boy.items())[:15], "en buyuk:", max(boy))
    for a, b, s, d in [x for x in ciftler if not x[2]][:20]:
        print("FARKLI", d, a, b)


if __name__ == "__main__":
    main()
