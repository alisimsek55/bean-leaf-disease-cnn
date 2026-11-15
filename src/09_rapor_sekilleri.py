# -*- coding: utf-8 -*-
"""
09 - RAPOR ŞEKİLLERİ (Word'e konacak fotoğraf panelleri). Tohum 42; seçim tekrarlanabilir.

Çıktılar (ciktilar/sekiller/rapor/):
  sekil1_veri_seti.png    16 ham fotoğraf (4x4), sınıf ve kaynak karışık/dengeli
  sekil2_duzenleme.png    12 ön işlenmiş görüntü (8 hastalıklı + 4 sağlıklı)
  sekil2b_on_isleme.png   ham -> maske -> işlenmiş dönüşümü (3 örnek)
  sekil3_siddet.png       her şiddet düzeyinden 4 yaprak, lezyon oranı (%) ve lezyon katmanı
  sekil4_ozellik.png      ilk evrişim katmanından 8 öznitelik haritası + Grad-CAM
Maske kalitesi bozuk (dışlanan) görüntüler seçilmez. Etiketler Türkçedir.
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, ISLENMIS, SINIFLAR, TOHUM, VERI, kaynak_kisa  # noqa: E402
from siddet import DUZEYLER, lezyon_maskesi  # noqa: E402
from veri import YaprakVeri, goruntuleri_yukle  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
CIKIS = CIKTI / "sekiller" / "rapor"
DPI = 150
GENIS = 1600 / DPI  # inç
AD = {"adi_yaprak_yanikligi": "Adi yanıklık", "hale_yanikligi": "Hale yanıklığı", "saglikli": "Sağlıklı"}


def tablo():
    rap = pd.read_csv(ISLENMIS / "on_isleme_raporu.csv")
    rap = rap[~rap["dislandi"]].reset_index(drop=True)
    sid = pd.read_csv(CIKTI / "siddet_tahmini.csv")[["dosya", "lezyon_orani", "siddet_duzeyi"]]
    rap = rap.merge(sid, on="dosya")
    rap["k"] = rap["kaynak"].map(kaynak_kisa)
    return rap


def dengeli_sec(df, sinif_sayilari, rng):
    """Her sınıftan istenen sayıda örnek; sınıf içinde kaynaklar sırayla (round robin) dağıtılır."""
    sec = []
    for s, n in sinif_sayilari.items():
        g = df[df.sinif == s]
        kaynaklar = sorted(g.k.unique())
        havuz = {k: list(rng.permutation(g[g.k == k].index)) for k in kaynaklar}
        i, say = 0, 0
        while say < n and any(havuz.values()):
            k = kaynaklar[i % len(kaynaklar)]
            i += 1
            if havuz[k]:
                sec.append(havuz[k].pop())
                say += 1
    return sec


def ac(yol):
    return Image.open(yol).convert("RGB")


def ham_kucuk(yol, boy=420):
    im = ac(yol)
    im.thumbnail((boy, boy))
    return np.asarray(im)


def izgara(n_sat, n_sut, baslik, h_kare=2.9):
    fig, eks = plt.subplots(n_sat, n_sut, figsize=(GENIS, h_kare * n_sat + 0.8), squeeze=False)
    fig.suptitle(baslik, fontsize=14, fontweight="bold", y=0.995)
    for a in eks.ravel():
        a.axis("off")
    return fig, eks


def kucult(yol, limit=1.4e6):
    """PNG 1,4 MB'ı aşarsa 256 renge indirger (fotoğraf paneli için gözle fark edilmez)."""
    if yol.stat().st_size > limit:
        Image.open(yol).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)             .save(yol, optimize=True)


def kaydet(fig, ad):
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.8, w_pad=0.4)
    yol = CIKIS / ad
    fig.savefig(yol, dpi=DPI)
    plt.close(fig)
    kucult(yol)
    print(ad, f"{yol.stat().st_size / 1e6:.2f} MB")


def sekil1(rap, rng):
    sec = dengeli_sec(rap, {"adi_yaprak_yanikligi": 5, "hale_yanikligi": 5, "saglikli": 6}, rng)
    sec = [sec[j] for j in rng.permutation(len(sec))]
    fig, eks = izgara(4, 4, "Şekil 1. Veri seti oluşturma (ham fotoğraflar)")
    for a, i in zip(eks.ravel(), sec):
        r = rap.loc[i]
        a.imshow(ham_kucuk(VERI / r.dosya))
        a.set_title(f"{AD[r.sinif]} – {r.k}", fontsize=12, pad=4)
    kaydet(fig, "sekil1_veri_seti.png")


def sekil2(rap, rng):
    sec_h = dengeli_sec(rap, {"adi_yaprak_yanikligi": 4, "hale_yanikligi": 4}, rng)
    sec_s = dengeli_sec(rap, {"saglikli": 4}, rng)
    sec_h = [sec_h[j] for j in rng.permutation(len(sec_h))]
    fig, eks = izgara(3, 4, "Şekil 2. Veri seti düzenleme (ön işlenmiş görüntüler)  –  üst iki satır: hastalıklı, alt satır: sağlıklı",
                      h_kare=3.0)
    for a, i in zip(eks.ravel(), sec_h + sec_s):
        r = rap.loc[i]
        a.imshow(ac(ISLENMIS / r.islenmis))
        a.set_title(f"{AD[r.sinif]} – {r.k}", fontsize=12, pad=4)
    kaydet(fig, "sekil2_duzenleme.png")

    # ham -> maske -> işlenmiş (CBB, hale, sağlıklı birer örnek)
    cbb = next(i for i in sec_h if rap.loc[i, "sinif"] == "adi_yaprak_yanikligi")
    hale = next(i for i in sec_h if rap.loc[i, "sinif"] == "hale_yanikligi")
    fig, eks = izgara(3, 3, "Şekil 2b. Ön işleme adımları: ham fotoğraf → yaprak maskesi → ön işlenmiş görüntü", h_kare=3.3)
    for sat, i in enumerate([cbb, hale, sec_s[0]]):
        r = rap.loc[i]
        eks[sat, 0].imshow(ham_kucuk(VERI / r.dosya))
        eks[sat, 1].imshow(np.asarray(Image.open(ISLENMIS / r.maske).convert("L")), cmap="gray")
        eks[sat, 2].imshow(ac(ISLENMIS / r.islenmis))
        eks[sat, 0].set_title(f"Ham – {AD[r.sinif]} ({r.k})", fontsize=11, pad=4)
        eks[sat, 1].set_title("Yaprak maskesi", fontsize=11, pad=4)
        eks[sat, 2].set_title("Ön işlenmiş (256×256)", fontsize=11, pad=4)
    kaydet(fig, "sekil2b_on_isleme.png")


def sekil3(rap, rng):
    fig, eks = izgara(3, 4, "Şekil 3. Sınıflandırma: sağlıklı / az hastalıklı / çok hastalıklı  (lezyon pikselleri mor)",
                      h_kare=3.3)
    for sat, d in enumerate(DUZEYLER):
        g = rap[rap.siddet_duzeyi == d]
        if d == "sağlıklı":
            sec = dengeli_sec(g[g.sinif == "saglikli"], {"saglikli": 4}, rng)
        else:  # hastalıklı düzeylerde CBB ve hale karışık
            sec = dengeli_sec(g, {"adi_yaprak_yanikligi": 2, "hale_yanikligi": 2}, rng)
        sec = [sec[j] for j in rng.permutation(len(sec))]
        for a, i in zip(eks[sat], sec):
            r = rap.loc[i]
            gr = np.asarray(ac(ISLENMIS / r.islenmis))
            m = np.asarray(Image.open(ISLENMIS / r.maske).convert("L"))
            lez, _ = lezyon_maskesi(gr, m)
            ust = gr.copy()
            ust[lez > 0] = (0.35 * ust[lez > 0] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
            a.imshow(ust)
            a.set_title(f"{d}\nlezyon oranı: %{100 * r.lezyon_orani:.1f}  ({AD[r.sinif]})", fontsize=9, pad=4)
    kaydet(fig, "sekil3_siddet.png")


def sekil4(rap, rng):
    mod = importlib.import_module("03_model")
    ck = torch.load(CIKTI / "model_en_iyi.pt", map_location="cpu", weights_only=False)
    model = mod.FasulyeCNN(sinif_sayisi=ck["sinif_sayisi"])
    model.load_state_dict(ck["model"])
    model.eval()
    # Örnek: modelin doğru ve emin (>%90) sınıflandırdığı, orta şiddetli adi yanıklık yaprağı
    aday = rap[(rap.sinif == "adi_yaprak_yanikligi") & rap.lezyon_orani.between(0.12, 0.30)]
    aday = aday.loc[[aday.index[j] for j in rng.permutation(len(aday))[:30]]]
    X = goruntuleri_yukle(aday, ck["girdi"])
    ds = YaprakVeri(X, np.zeros(len(X), int), ck["ortalama"], ck["std"], egitim=False)
    sec = None
    for j in range(len(ds)):
        with torch.no_grad():
            p = F.softmax(model(ds[j][0][None]), 1)[0]
        if int(p.argmax()) == SINIFLAR.index("adi_yaprak_yanikligi") and float(p.max()) > 0.9:
            sec = j
            break
    if sec is None:
        raise SystemExit("uygun örnek bulunamadı")
    r = aday.iloc[sec]
    x = ds[sec][0][None].clone().requires_grad_(True)
    blok = model.ozellik[0]  # ilk evrişim bloğu: conv 4x4 + BN + ReLU
    with torch.no_grad():
        ilk = blok.relu(blok.bn(blok.conv(x)))[0].numpy()
    yap = X[sec, 3].numpy() > 127
    # yaprak içinde en yüksek varyanslı 8 kanal (deterministik)
    kanallar = np.sort(np.argsort(-ilk[:, yap].var(1))[:8])
    akt, grad = {}, {}
    k = model.son_evrisim.relu
    h1 = k.register_forward_hook(lambda m, i, o: akt.__setitem__("a", o))
    h2 = k.register_full_backward_hook(lambda m, gi, go: grad.__setitem__("g", go[0]))
    c = model(x)
    sinif = int(c.argmax(1))
    c[0, sinif].backward()
    h1.remove()
    h2.remove()
    w = grad["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * akt["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()
    rgb = X[sec, :3].numpy().transpose(1, 2, 0)
    cmap = plt.get_cmap("jet")(cam)[..., :3]
    ust = (0.5 * rgb / 255 + 0.5 * cmap) * yap[..., None] + (rgb / 255) * (~yap[..., None])
    fig = plt.figure(figsize=(GENIS, 4.6))
    fig.suptitle("Şekil 4. Özellik çıkarma: ilk evrişim katmanı öznitelik haritaları ve Grad-CAM", fontsize=14,
                 fontweight="bold", y=0.985)
    gs = fig.add_gridspec(2, 6, width_ratios=[1.3, 1, 1, 1, 1, 1.3], wspace=0.06, hspace=0.2,
                          left=0.01, right=0.99, top=0.83, bottom=0.02)
    a = fig.add_subplot(gs[:, 0])
    a.imshow(rgb.astype(np.uint8))
    a.axis("off")
    a.set_title(f"Girdi (128×128)\n{AD[r.sinif]}", fontsize=11)
    for n, ch in enumerate(kanallar):
        a = fig.add_subplot(gs[n // 4, 1 + n % 4])
        a.imshow(ilk[ch] * yap, cmap="viridis")
        a.axis("off")
        a.set_title(f"Öznitelik haritası {n + 1}", fontsize=9.5)
    a = fig.add_subplot(gs[:, 5])
    a.imshow(ust)
    a.axis("off")
    a.set_title(f"Grad-CAM\ntahmin: {AD[SINIFLAR[sinif]]}\n(kırmızı = yüksek ilgi)", fontsize=10.5)
    yol = CIKIS / "sekil4_ozellik.png"
    fig.savefig(yol, dpi=DPI)
    plt.close(fig)
    kucult(yol)
    print("sekil4_ozellik.png", f"{yol.stat().st_size / 1e6:.2f} MB", "örnek:", r.dosya, "kanallar", kanallar.tolist())


def main():
    CIKIS.mkdir(parents=True, exist_ok=True)
    rap = tablo()
    for f in (sekil1, sekil2, sekil3, sekil4):
        f(rap, np.random.default_rng(TOHUM))  # her panel kendi tohumuyla: biri değişse diğerleri değişmez


if __name__ == "__main__":
    main()
