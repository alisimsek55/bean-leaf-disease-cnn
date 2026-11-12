"""Egitim/dogrulama bolmesinde capraz sizinti denetimi (goruntu gerekmez; yalnizca CSV okur).
data/bolme_egitim.csv ve data/bolme_dogrulama.csv icin sunlari raporlar:
  * iki tarafta birden bulunan `grup` kimlikleri,
  * iki tarafta birden bulunan dosya yollari,
  * iki tarafta birden bulunan seri cekim anahtarlari:
      - cekim saniyesi (dosya adindaki IMG_<YYYYMMDD>_<HHMMSS>, ayni kaynak icinde),
      - pencereli seri (yakin_kopya.py ile ayni tanim: ayni kaynak + sinif icinde ardisik kareler
        arasinda <= pencere saniye; zincirleme).
Kaynak bazinda sayilar ve etkilenen dogrulama satiri sayisi/yuzdesi yazilir. Herhangi bir sizinti
bulunursa cikis kodu 1'dir.
Kullanim: python -I bolme_denetimi.py <proje_koku> [pencere_sn=2]
"""
import csv
import importlib.util
import os
import sys
from collections import Counter


def _yakin_kopya():
    # `python -I` betik klasorunu sys.path'e eklemedigi icin dosya yolundan yuklenir.
    yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yakin_kopya.py")
    spec = importlib.util.spec_from_file_location("yakin_kopya", yol)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


yk = _yakin_kopya()


def oku(yol):
    with open(yol, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def denetle(egitim, dogrulama, pencere=yk.PENCERE_SN):
    """Iki satir listesi (dict) icin sizinti raporu (dict) dondurur."""
    tum = [dict(r, _bolum="egitim") for r in egitim] + [dict(r, _bolum="dogrulama") for r in dogrulama]
    n_dog = len(dogrulama)

    ortak_grup = sorted({r["grup"] for r in egitim} & {r["grup"] for r in dogrulama})
    ortak_dosya = sorted({r["dosya"] for r in egitim} & {r["dosya"] for r in dogrulama})

    # Cekim saniyesi (kaynak, anahtar)
    def sn(r):
        a = yk.saniye_anahtari(r["dosya"])
        return (r.get("kaynak", ""), a) if a else None

    eg_sn = {sn(r) for r in egitim} - {None}
    dog_sn = {sn(r) for r in dogrulama} - {None}
    ortak_sn = eg_sn & dog_sn

    # Pencereli seri: yalnizca seri baglantilariyla bilesenler (iki bolum birlikte)
    seri_grup = yk.grup_etiketleri(len(tum), yk.seri_baglantilari(tum, pencere))
    taraflar = {}
    for r, g in zip(tum, seri_grup):
        taraflar.setdefault(g, set()).add(r["_bolum"])
    capraz_seri = {g for g, t in taraflar.items() if len(t) == 2}

    kaynaklar = sorted({r.get("kaynak", "") for r in tum})
    kaynak_bazinda = {}
    for k in kaynaklar:
        dog_k = [(r, g) for r, g in zip(tum, seri_grup) if r["_bolum"] == "dogrulama" and r.get("kaynak", "") == k]
        kaynak_bazinda[k] = {
            "dogrulama_satiri": len(dog_k),
            "ortak_grup": len({r["grup"] for r, _ in dog_k} & set(ortak_grup)),
            "ortak_saniye": sum(1 for kk, _ in ortak_sn if kk == k),
            "capraz_seri": len({g for _, g in dog_k} & capraz_seri),
            "dogrulama_saniye_etkilenen": sum(1 for r, _ in dog_k if sn(r) in ortak_sn),
            "dogrulama_seri_etkilenen": sum(1 for _, g in dog_k if g in capraz_seri),
        }

    dog_grup = set(ortak_grup)
    dog_dosya = set(ortak_dosya)
    etkilenen = sum(
        1 for r, g in zip(tum, seri_grup)
        if r["_bolum"] == "dogrulama"
        and (r["grup"] in dog_grup or r["dosya"] in dog_dosya or sn(r) in ortak_sn or g in capraz_seri)
    )
    return {
        "pencere_sn": pencere,
        "egitim_satiri": len(egitim),
        "dogrulama_satiri": n_dog,
        "ortak_grup": len(ortak_grup),
        "ortak_dosya": len(ortak_dosya),
        "ortak_saniye": len(ortak_sn),
        "ortak_dakika": len({(k, a[:13]) for k, a in eg_sn} & {(k, a[:13]) for k, a in dog_sn}),
        "capraz_seri": len(capraz_seri),
        "dogrulama_saniye_etkilenen": sum(v["dogrulama_saniye_etkilenen"] for v in kaynak_bazinda.values()),
        "dogrulama_seri_etkilenen": sum(v["dogrulama_seri_etkilenen"] for v in kaynak_bazinda.values()),
        "dogrulama_etkilenen": etkilenen,
        "kaynak_bazinda": kaynak_bazinda,
        "ornek_ortak_grup": ortak_grup[:5],
        "ornek_ortak_dosya": ortak_dosya[:5],
        "ornek_ortak_saniye": sorted(a for _, a in ortak_sn)[:5],
    }


def sizinti_var(rapor):
    return any(rapor[k] for k in ("ortak_grup", "ortak_dosya", "ortak_saniye", "capraz_seri"))


def yuzde(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def yazdir(rapor):
    n = rapor["dogrulama_satiri"]
    print(f"egitim={rapor['egitim_satiri']}  dogrulama={n}  seri penceresi={rapor['pencere_sn']} sn")
    print(f"ortak grup kimligi        : {rapor['ortak_grup']}")
    print(f"ortak dosya yolu          : {rapor['ortak_dosya']}")
    print(f"ortak cekim saniyesi      : {rapor['ortak_saniye']}  (ortak cekim dakikasi: {rapor['ortak_dakika']})")
    print(f"iki tarafa bolunmus seri  : {rapor['capraz_seri']}")
    print(f"etkilenen dogrulama satiri: saniye {rapor['dogrulama_saniye_etkilenen']} "
          f"({yuzde(rapor['dogrulama_saniye_etkilenen'], n)}), seri {rapor['dogrulama_seri_etkilenen']} "
          f"({yuzde(rapor['dogrulama_seri_etkilenen'], n)}), herhangi {rapor['dogrulama_etkilenen']} "
          f"({yuzde(rapor['dogrulama_etkilenen'], n)})")
    print("kaynak bazinda:")
    for k, v in rapor["kaynak_bazinda"].items():
        m = v["dogrulama_satiri"]
        print(f"  {k}: dogrulama={m}  ortak_grup={v['ortak_grup']}  ortak_saniye={v['ortak_saniye']}  "
              f"capraz_seri={v['capraz_seri']}  etkilenen(saniye)={v['dogrulama_saniye_etkilenen']} "
              f"({yuzde(v['dogrulama_saniye_etkilenen'], m)})  etkilenen(seri)={v['dogrulama_seri_etkilenen']} "
              f"({yuzde(v['dogrulama_seri_etkilenen'], m)})")
    for ad in ("ornek_ortak_grup", "ornek_ortak_dosya", "ornek_ortak_saniye"):
        if rapor[ad]:
            print(f"{ad}: {rapor[ad]}")
    print("SONUC:", "SIZINTI VAR" if sizinti_var(rapor) else "sizinti bulunmadi")


def main():
    kok = sys.argv[1] if len(sys.argv) > 1 else "."
    pencere = float(sys.argv[2]) if len(sys.argv) > 2 else yk.PENCERE_SN
    data = os.path.join(kok, "data")
    rapor = denetle(oku(os.path.join(data, "bolme_egitim.csv")), oku(os.path.join(data, "bolme_dogrulama.csv")),
                    pencere)
    yazdir(rapor)
    return 1 if sizinti_var(rapor) else 0


if __name__ == "__main__":
    sys.exit(main())
