# -*- coding: utf-8 -*-
"""
11 - EK CBB KAYNAĞI (E = Bugwood / IPM Images, konu 11171): ayıklama + kopya denetimi + ön işleme.

Girdi : data/raw_bugwood/<numara>.jpg (scripts/indir_bugwood.py), data/bugwood_metadata.csv,
        data/bugwood_ayiklama.csv (elle, görüntülere tek tek bakılarak hazırlandı):
          numara, parca, karar (al|cikar), neden, kirpma ("x0,y0,x1,y1" orijinal piksel; boş = tüm görüntü),
          seri (elle verilen seri/yaprak grubu; aynı yaprağın/sahnenin kareleri aynı seri)
Adımlar:
  1. Bugwood görüntülerinin sağ altındaki numara etiketi (siyah kutu, beyaz yazı) bir kaynak imzasıdır:
     her görüntünün alt %7'si kesilir (kırpma verilmişse kırpma bu bölgeye taşmayacak şekilde kısaltılır).
  2. 256-bit dHash (scripts/yakin_kopya.py) — alınan görüntülerden A/B/C/D'nin herhangi birine Hamming <= 20 olan
     ATILIR (C sızıntısı olmasın). E içi yakın kopyalar ve elle verilen `seri` aynı `grup`a birleştirilir.
  3. `genel_maske.on_isle(otomatik)` ile maskeleme; başarısız olan dışlanır.
Çıktı: data/islenmis_bugwood/{goruntu,maske}/adi_yaprak_yanikligi/*.png, data/islenmis_bugwood/bugwood_veri.csv
       (bolme_*.csv ile aynı sütunlar + zemin, alan_orani, numara, atif), bugwood_ozeti.json,
       ciktilar/genel_model/bugwood_maske_kontrol.jpg (gözle kontrol için).
       --panel: ciktilar/sekiller/rapor/sekil_bugwood_ornekleri.png (8 örnek, ham kırpım + maskeli, atıflı).
Kullanım: python src/11_bugwood_veri.py [--panel]
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, KOK, VERI  # noqa: E402
import genel_maske  # noqa: E402

_spec = importlib.util.spec_from_file_location("yakin_kopya", KOK / "scripts" / "yakin_kopya.py")
yk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yk)

HAM_E = VERI / "raw_bugwood"
CIKIS = VERI / "islenmis_bugwood"
SINIF = "adi_yaprak_yanikligi"
KAYNAK = "bugwood_ipm"
ESIK = 20
ALT_KESIM = 0.07  # sağ alttaki numara etiketini atmak için alttan kesilen oran


def kirpma_kutusu(kirpma, W, H, alt_kesim=ALT_KESIM):
    """'x0,y0,x1,y1' (ya da boş) -> görüntü içinde, etiket bölgesini dışlayan (x0, y0, x1, y1)."""
    ymax = int(round(H * (1 - alt_kesim)))
    if isinstance(kirpma, str) and kirpma.strip():
        x0, y0, x1, y1 = [int(float(v)) for v in kirpma.split(",")]
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    x0, x1 = max(0, min(x0, W)), max(0, min(x1, W))
    y0, y1 = max(0, min(y0, ymax)), max(0, min(y1, ymax))
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise ValueError(f"kırpma çok küçük: {kirpma}")
    return x0, y0, x1, y1


def dhash_dizi(rgb):
    """yk.dhash ile aynı (yol yerine dizi alır; bit y*N+x = piksel(y,x) > piksel(y,x+1)).
    Not (2026-11-16): ilk sürüm bitleri ters sırada (ilk piksel en yüksek bit) topluyordu; A/B/C/D ile
    karşılaştırma bu yüzden geçersizdi. Düzeltilmiş tanımla yeniden ölçüldü: E'nin A/B/C/D'ye en küçük
    uzaklığı 78/79/85/88 (> 20; 8 yönlü dahil) — sonuç (yakın kopya 0) değişmedi."""
    g = Image.fromarray(rgb).convert("L").resize((yk.N + 1, yk.N), Image.LANCZOS)
    p = g.tobytes()
    bit = 0
    for y in range(yk.N):
        for x in range(yk.N):
            if p[y * (yk.N + 1) + x] > p[y * (yk.N + 1) + x + 1]:
                bit |= 1 << (y * yk.N + x)
    return bit


def main():
    ay = pd.read_csv(VERI / "bugwood_ayiklama.csv", dtype={"numara": str, "parca": str, "kirpma": str, "seri": str})
    meta = pd.read_csv(VERI / "bugwood_metadata.csv", dtype={"numara": str}).set_index("numara")
    al = ay[ay.karar == "al"].reset_index(drop=True)
    kirp, hashler = [], []
    for r in al.itertuples():
        rgb = np.asarray(Image.open(HAM_E / f"{r.numara}.jpg").convert("RGB"))
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = kirpma_kutusu(r.kirpma, W, H)
        k = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        kirp.append(k)
        hashler.append(dhash_dizi(k))
    # A/B/C (raw) ve D (web) ile yakın kopya
    ham = pd.read_csv(VERI / "etiketler.csv")
    ham = ham[ham.kume == "raw"]
    diger = [(k, VERI / d) for k, d in zip(ham.kaynak, ham.dosya)]
    wyol = VERI / "islenmis_web" / "web_veri.csv"
    if wyol.exists():
        diger += [("plantwild_web", VERI / d) for d in pd.read_csv(wyol).dosya]
    dh = [(k, yk.dhash(p)) for k, p in diger]
    yakin = {}
    for i, a in enumerate(hashler):
        for k, b in dh:
            d = (a ^ b).bit_count()
            if d <= ESIK and (i not in yakin or d < yakin[i][1]):
                yakin[i] = (k, d)
    # E içi gruplar: dHash + elle seri
    bag = [(i, j) for i, j, _ in yk.dhash_baglantilari(hashler, ESIK)]
    seri = al.seri.fillna("").tolist()
    for i in range(len(al)):
        for j in range(i + 1, len(al)):
            if seri[i] and seri[i] == seri[j]:
                bag.append((i, j))
    gruplar = yk.grup_etiketleri(len(al), bag)
    kayit, dislanan, kontrol = [], [], []
    for i, r in al.iterrows():
        if i in yakin:
            dislanan.append({"numara": r.numara, "parca": r.parca, "neden": f"yakin_kopya_{yakin[i][0]}"})
            continue
        g, m, bilgi = genel_maske.on_isle(kirp[i], "otomatik")
        if not bilgi["basarili"]:
            dislanan.append({"numara": r.numara, "parca": r.parca, "neden": "maske_basarisiz"})
            continue
        ad = f"bw__{r.numara}{'_' + r.parca if isinstance(r.parca, str) and r.parca else ''}.png"
        for alt, dizi in (("goruntu", g), ("maske", m)):
            (CIKIS / alt / SINIF).mkdir(parents=True, exist_ok=True)
            Image.fromarray(dizi).save(CIKIS / alt / SINIF / ad)
        kontrol.append((ad, kirp[i], g))
        kayit.append({"islenmis": f"../islenmis_bugwood/goruntu/{SINIF}/{ad}",
                      "maske": f"../islenmis_bugwood/maske/{SINIF}/{ad}", "dosya": f"raw_bugwood/{r.numara}.jpg",
                      "sinif": SINIF, "kaynak": KAYNAK, "grup": "e" + gruplar[i], "zemin": bilgi["zemin"],
                      "alan_orani": bilgi["alan_orani"], "numara": r.numara,
                      "atif": meta.atif.get(r.numara, "Bugwood.org")})
    df = pd.DataFrame(kayit)
    CIKIS.mkdir(parents=True, exist_ok=True)
    df.to_csv(CIKIS / "bugwood_veri.csv", index=False)
    pd.DataFrame(dislanan, columns=["numara", "parca", "neden"]).to_csv(CIKIS / "bugwood_dislanan.csv", index=False)
    ozet = {"ayiklama_satiri": int(len(ay)), "goruntu_sayisi": int(ay.numara.nunique()),
            "alinan_goruntu": int(al.numara.nunique()), "alinan_ornek": int(len(al)),
            "cikarilan_goruntu": int(ay[ay.karar == "cikar"].numara.nunique()),
            "cikarma_nedenleri": {k: int(v) for k, v in ay[ay.karar == "cikar"].neden.str.split(":").str[0].value_counts().items()},
            "abc_d_yakin_kopya": len(yakin), "maske_basarisiz": int(sum(d["neden"] == "maske_basarisiz" for d in dislanan)),
            "son_ornek": int(len(df)), "grup_sayisi": int(df.grup.nunique()) if len(df) else 0,
            "zemin": {k: int(v) for k, v in df.zemin.value_counts().items()} if len(df) else {}}
    (CIKIS / "bugwood_ozeti.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    # gözle kontrol sayfası: (kırpılmış ham | maskelenmiş) çiftleri
    T, sut = 200, 6
    sat = (len(kontrol) + sut // 2 - 1) // (sut // 2)
    sayfa = Image.new("RGB", (sut * T, max(1, sat) * (T + 14)), "white")
    c = ImageDraw.Draw(sayfa)
    for j, (ad, k, g) in enumerate(kontrol):
        x, y = (j % (sut // 2)) * 2 * T, (j // (sut // 2)) * (T + 14)
        a = Image.fromarray(k)
        a.thumbnail((T, T))
        sayfa.paste(a, (x, y + 14))
        sayfa.paste(Image.fromarray(g).resize((T, T)), (x + T, y + 14))
        c.text((x + 2, y + 1), ad, fill="black")
    (CIKTI / "genel_model").mkdir(parents=True, exist_ok=True)
    sayfa.save(CIKTI / "genel_model" / "bugwood_maske_kontrol.jpg", quality=85)
    print(json.dumps(ozet, ensure_ascii=False, indent=2))


def ornek_paneli(n=8):
    """E'den 8 örnek: üstte kırpılmış ham kare, altta modele giren maskeli görüntü; altında atıf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ortak import SEKIL
    df = pd.read_csv(CIKIS / "bugwood_veri.csv", dtype={"numara": str})
    ay = pd.read_csv(VERI / "bugwood_ayiklama.csv", dtype={"numara": str, "parca": str, "kirpma": str})
    ay = ay[ay.karar == "al"].fillna("")
    # her gruptan en az bir örnek, sonra sırayla
    sec = df.groupby("grup").head(1)
    sec = pd.concat([sec, df.drop(sec.index)]).head(n)
    fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 5.6))
    for j, r in enumerate(sec.itertuples()):
        parca = Path(r.islenmis).stem.split("__")[1][len(r.numara):].lstrip("_")
        a = ay[(ay.numara == r.numara) & (ay.parca == parca)].iloc[0]
        rgb = np.asarray(Image.open(HAM_E / f"{r.numara}.jpg").convert("RGB"))
        x0, y0, x1, y1 = kirpma_kutusu(a.kirpma, rgb.shape[1], rgb.shape[0])
        ax[0, j].imshow(rgb[y0:y1, x0:x1])
        ax[1, j].imshow(np.asarray(Image.open(VERI / "islenmis" / r.islenmis)))
        ax[0, j].set_title(f"{r.numara}{'_' + parca if parca else ''}", fontsize=8)
        ax[1, j].set_xlabel(r.atif.replace(", ", ",\n", 1), fontsize=6)
        for k in (0, 1):
            ax[k, j].set_xticks([])
            ax[k, j].set_yticks([])
    ax[0, 0].set_ylabel("Kırpılmış ham", fontsize=8)
    ax[1, 0].set_ylabel("Maskeli (model girdisi)", fontsize=8)
    fig.suptitle("Kaynak E: Bugwood / IPM Images, adi yaprak yanıklığı (Xanthomonas axonopodis pv. phaseoli) — "
                 "CC BY 3.0 US / CC BY-NC 3.0 US; atıf: Fotoğrafçı, Kurum, Bugwood.org", fontsize=8)
    fig.tight_layout()
    (SEKIL / "rapor").mkdir(parents=True, exist_ok=True)
    fig.savefig(SEKIL / "rapor" / "sekil_bugwood_ornekleri.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--panel" in sys.argv:
        ornek_paneli()
    else:
        main()
