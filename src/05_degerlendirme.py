# -*- coding: utf-8 -*-
"""
05 - DEĞERLENDİRME: doğrulama kümesinde metrikler, kaynak bazında analiz, kontrol deneyleri,
Grad-CAM ve tüm şekiller. Tüm sayılar ciktilar/metrikler.json dosyasına yazılır.

Önce 04 (ana + kontroller) ve 06 çalıştırılmış olmalı.
Kullanım: python src/05_degerlendirme.py
"""
import importlib
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).parent))
from ortak import (CIKTI, GIRIS_BOYUT, ISLENMIS, KISA_AD, SEKIL, SINIFLAR, SINIF_ADLARI_TR, TOHUM, VERI,  # noqa: E402
                   kaynak_kisa)
from veri import YaprakVeri, bolme_oku, etiketler, goruntuleri_yukle  # noqa: E402

model_mod = importlib.import_module("03_model")
siddet = importlib.import_module("06_siddet_tahmini")
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_yukle(yol):
    ck = torch.load(yol, map_location=cihaz, weights_only=False)
    m = model_mod.FasulyeCNN(sinif_sayisi=ck["sinif_sayisi"]).to(cihaz)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def tahmin_et(model, ck, df):
    X = goruntuleri_yukle(df, ck["girdi"])
    y = etiketler(df, ck["gorev"])
    ds = YaprakVeri(X, y, ck["ortalama"], ck["std"], egitim=False)
    olas = []
    with torch.no_grad():
        for i in range(0, len(ds), 128):
            x = torch.stack([ds[j][0] for j in range(i, min(i + 128, len(ds)))]).to(cihaz)
            olas.append(F.softmax(model(x), 1).cpu())
    return X, y, torch.cat(olas).numpy()


def temel_metrikler(y, p, adlar):
    pr, re, f1, des = precision_recall_fscore_support(y, p, labels=range(len(adlar)), zero_division=0)
    return {
        "dogruluk": float(accuracy_score(y, p)),
        "dengeli_dogruluk": float(balanced_accuracy_score(y, p)),
        "makro_f1": float(f1_score(y, p, average="macro")),
        "sinif_bazinda": {a: {"kesinlik": float(pr[i]), "duyarlilik": float(re[i]), "f1": float(f1[i]),
                              "destek": int(des[i])} for i, a in enumerate(adlar)},
        "karisiklik_matrisi": {"satir=gercek, sutun=tahmin, sira": adlar,
                               "matris": confusion_matrix(y, p, labels=range(len(adlar))).tolist()},
    }


def kaynak_bazinda(df, y, p):
    """Her (sınıf, kaynak) alt kümesi için doğruluk ve tahmin dağılımı."""
    d = df.assign(y=y, p=p)
    sonuc = {}
    for (s, k), g in d.groupby(["sinif", "kaynak"]):
        sonuc[f"{s} | kaynak {kaynak_kisa(k)}"] = {
            "n": int(len(g)), "dogruluk": float((g.y == g.p).mean()),
            "tahmin_dagilimi": {SINIFLAR[i]: int((g.p == i).sum()) for i in range(3)}}
    for k, g in d.groupby("kaynak"):
        sonuc[f"TUM | kaynak {kaynak_kisa(k)}"] = {"n": int(len(g)), "dogruluk": float((g.y == g.p).mean())}
    return sonuc


# ---------------------------------------------------------------- Grad-CAM
class GradCAM:
    """Son evrişim bloğunun (8x8x256) aktivasyon ve gradyanlarıyla sınıf aktivasyon haritası."""

    def __init__(self, model):
        self.model, self.akt, self.grad = model, None, None
        katman = model.son_evrisim.relu
        katman.register_forward_hook(lambda m, i, o: setattr(self, "akt", o))
        katman.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0]))

    def __call__(self, x, sinif=None):
        self.model.zero_grad()
        c = self.model(x)
        if sinif is None:
            sinif = c.argmax(1)
        c.gather(1, sinif.view(-1, 1)).sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.akt).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
        return cam.detach().cpu().numpy(), sinif.cpu().numpy()


def gradcam_analizi(model, ck, df, X, p):
    """Tüm doğrulama görüntülerinde CAM ağırlığının yaprak içi / yaprak kenarı / lezyon payı."""
    gc = GradCAM(model)
    ds = YaprakVeri(X, np.zeros(len(X), int), ck["ortalama"], ck["std"], egitim=False)
    camler = []
    for i in range(0, len(ds), 64):
        x = torch.stack([ds[j][0] for j in range(i, min(i + 64, len(ds)))]).to(cihaz)
        x.requires_grad_(True)
        cam, _ = gc(x, torch.as_tensor(p[i:i + len(x)], device=cihaz))
        camler.append(cam)
    camler = np.concatenate(camler)
    kayit = []
    for i in range(len(X)):
        rgb = X[i, :3].numpy().transpose(1, 2, 0).copy()
        m = X[i, 3].numpy()
        yap = (m > 127).astype(np.uint8)
        ic = cv2.erode(yap, np.ones((9, 9), np.uint8))   # kenar şeridi = yaprak - 4 px aşındırılmış yaprak
        serit = yap - ic
        lez, _ = siddet.lezyon_maskesi(rgb, m)
        c = camler[i]
        top = c.sum() + 1e-8
        alan = yap.size
        kayit.append({
            "cam_yaprak_ici": c[yap > 0].sum() / top, "alan_yaprak": yap.sum() / alan,
            "cam_kenar_seridi": c[serit > 0].sum() / top, "alan_kenar_seridi": serit.sum() / alan,
            "cam_lezyon": c[lez > 0].sum() / top, "alan_lezyon": lez.sum() / alan,
        })
    k = pd.DataFrame(kayit)
    k["sinif"] = df["sinif"].values
    ozet = {}
    for s, g in k.groupby("sinif"):
        ozet[s] = {
            "cam_payi_yaprak_ici": float(g.cam_yaprak_ici.mean()), "alan_payi_yaprak": float(g.alan_yaprak.mean()),
            "cam_payi_kenar_seridi": float(g.cam_kenar_seridi.mean()),
            "alan_payi_kenar_seridi": float(g.alan_kenar_seridi.mean()),
            "cam_payi_lezyon": float(g.cam_lezyon.mean()), "alan_payi_lezyon": float(g.alan_lezyon.mean()),
            # zenginleşme > 1: CAM o bölgeye alanının oranından fazla ağırlık veriyor
            "kenar_zenginlesme": float((g.cam_kenar_seridi / g.alan_kenar_seridi.clip(1e-6)).median()),
            "lezyon_zenginlesme_medyan": float((g.cam_lezyon / g.alan_lezyon.clip(1e-6))[g.alan_lezyon > 0.01].median()),
        }
    return camler, ozet


# ---------------------------------------------------------------- şekiller
def sekil_sinif_ornekleri(df):
    n = 6
    fig, eks = plt.subplots(3, n, figsize=(2.2 * n, 2.6 * 3))
    for i, s in enumerate(SINIFLAR):
        d = df[df.sinif == s]
        # kaynak çeşitliliği için her kaynaktan örnek al
        # her kaynaktan sırayla birer örnek (kaynak çeşitliliği görünsün)
        gruplar = [g.sample(frac=1, random_state=TOHUM) for _, g in d.groupby("kaynak")]
        sira = [g.iloc[k] for k in range(n) for g in gruplar if k < len(g)]
        sec = pd.DataFrame(sira[:n])
        for j in range(n):
            ax = eks[i, j]
            ax.axis("off")
            if j < len(sec):
                r = sec.iloc[j]
                ax.imshow(Image.open(ISLENMIS / r.islenmis))
                ax.set_title(f"{SINIF_ADLARI_TR[s]}\nkaynak {kaynak_kisa(r.kaynak)}", fontsize=8)
    fig.suptitle("Sınıf örnekleri (ön işleme sonrası, 256×256, arka plan siyah)", fontsize=12)
    fig.tight_layout()
    fig.savefig(SEKIL / "sinif_ornekleri.png", dpi=130)
    plt.close(fig)


def sekil_on_isleme(rap):
    ornekler = [("adi_yaprak_yanikligi", "mendeley_ykvcrjffzd"), ("saglikli", "mendeley_ykvcrjffzd"),
                ("hale_yanikligi", "mendeley_3km9d246z2"), ("saglikli", "mendeley_3km9d246z2"),
                ("adi_yaprak_yanikligi", "embrapa_pddb")]
    fig, eks = plt.subplots(len(ornekler), 3, figsize=(9, 3 * len(ornekler)))
    for i, (s, k) in enumerate(ornekler):
        r = rap[(rap.sinif == s) & (rap.kaynak == k)].sample(1, random_state=TOHUM + 3).iloc[0]
        ham = Image.open(VERI / r.dosya)
        if ham.mode != "RGB":
            ham = ham.convert("RGBA")
            z = Image.new("RGBA", ham.size, (255, 255, 255, 255))
            ham = Image.alpha_composite(z, ham).convert("RGB")
        ham.thumbnail((512, 512))
        islenmis = np.asarray(Image.open(ISLENMIS / r.islenmis))
        mask = np.asarray(Image.open(ISLENMIS / r.maske))
        for j, (img, bas) in enumerate([(ham, "1) Ham görüntü"), (mask, "2) Yaprak maskesi"),
                                        (islenmis, "3) Zemin siyah + kırpılmış")]):
            ax = eks[i, j]
            ax.imshow(img, cmap="gray" if j == 1 else None)
            ax.axis("off")
            ax.set_title(f"{bas}" + (f"\n{SINIF_ADLARI_TR[s]}, kaynak {kaynak_kisa(k)}" if j == 0 else ""),
                         fontsize=9)
    fig.suptitle("Ön işleme öncesi / sonrası (her kaynaktan bir örnek)", fontsize=12)
    fig.tight_layout()
    fig.savefig(SEKIL / "on_isleme_oncesi_sonrasi.png", dpi=120)
    plt.close(fig)


def sekil_egriler(gecmis, en_iyi, ad, baslik):
    g = pd.DataFrame(gecmis)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(g.epoch, g.egitim_kayip, label="Eğitim")
    a1.plot(g.epoch, g.dogrulama_kayip, label="Doğrulama")
    a2.plot(g.epoch, 100 * g.egitim_dogruluk, label="Eğitim")
    a2.plot(g.epoch, 100 * g.dogrulama_dogruluk, label="Doğrulama")
    for a, y in [(a1, "Kayıp (ağırlıklı çapraz entropi)"), (a2, "Doğruluk (%)")]:
        a.axvline(en_iyi, color="gray", ls="--", lw=1, label=f"En iyi epoch ({en_iyi})")
        a.set_xlabel("Epoch")
        a.set_ylabel(y)
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(baslik)
    fig.tight_layout()
    fig.savefig(SEKIL / ad, dpi=130)
    plt.close(fig)


def sekil_karisiklik(cm, adlar, ad, baslik):
    cm = np.array(cm)
    yuzde = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(yuzde, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(adlar)):
        for j in range(len(adlar)):
            ax.text(j, i, f"{cm[i, j]}\n(%{100 * yuzde[i, j]:.1f})", ha="center", va="center",
                    color="white" if yuzde[i, j] > 0.5 else "black", fontsize=11)
    ax.set_xticks(range(len(adlar)), adlar)
    ax.set_yticks(range(len(adlar)), adlar)
    ax.set_xlabel("Tahmin edilen sınıf")
    ax.set_ylabel("Gerçek sınıf")
    ax.set_title(baslik, fontsize=10)
    fig.tight_layout()
    fig.savefig(SEKIL / ad, dpi=130)
    plt.close(fig)


def sekil_gradcam(df, X, y, p, olas, camler):
    rng = np.random.default_rng(TOHUM)
    secim = []
    for c in range(3):  # her sınıftan 3 doğru + 1 yanlış (varsa)
        dogru = np.where((y == c) & (p == c))[0]
        yanlis = np.where((y == c) & (p != c))[0]
        secim += list(rng.choice(dogru, min(3, len(dogru)), replace=False))
        if len(yanlis):
            secim += list(rng.choice(yanlis, 1))
    n = len(secim)
    sut = 4
    satir = int(np.ceil(n / sut))
    fig, eks = plt.subplots(2 * satir, sut, figsize=(2.6 * sut, 2.7 * 2 * satir))
    for a in eks.ravel():
        a.axis("off")
    for k, i in enumerate(secim):
        rr, cc = 2 * (k // sut), k % sut
        rgb = X[i, :3].numpy().transpose(1, 2, 0)
        eks[rr, cc].imshow(rgb)
        dur = "DOĞRU" if y[i] == p[i] else "YANLIŞ"
        eks[rr, cc].set_title(f"Gerçek: {KISA_AD[SINIFLAR[y[i]]]} | kaynak {kaynak_kisa(df.kaynak.iloc[i])}\n"
                              f"Tahmin: {KISA_AD[SINIFLAR[p[i]]]} (%{100 * olas[i, p[i]]:.0f}) {dur}", fontsize=7.5)
        eks[rr + 1, cc].imshow(rgb)
        eks[rr + 1, cc].imshow(camler[i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        eks[rr + 1, cc].set_title("Grad-CAM (tahmin edilen sınıf)", fontsize=7.5)
    fig.suptitle("Grad-CAM: modelin karar verirken baktığı bölgeler (kırmızı = yüksek)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(SEKIL / "gradcam_ornekleri.png", dpi=130)
    plt.close(fig)


def sekil_dogru_yanlis(df, X, y, p, olas):
    rng = np.random.default_rng(TOHUM + 1)
    dogru = [i for c in range(3) for i in rng.choice(np.where((y == c) & (p == c))[0], 2, replace=False)]
    yanlis_tum = np.where(y != p)[0]
    yanlis = list(rng.choice(yanlis_tum, min(6, len(yanlis_tum)), replace=False))
    fig, eks = plt.subplots(2, 6, figsize=(14, 6.8))
    for a in eks.ravel():
        a.axis("off")
    for satir, liste, bas in [(0, dogru, "Doğru"), (1, yanlis, "Yanlış")]:
        for j, i in enumerate(liste):
            ax = eks[satir, j]
            ax.imshow(X[i, :3].numpy().transpose(1, 2, 0))
            ax.set_title(f"{bas} | kaynak {kaynak_kisa(df.kaynak.iloc[i])}\nGerçek: {KISA_AD[SINIFLAR[y[i]]]}\n"
                         f"Tahmin: {KISA_AD[SINIFLAR[p[i]]]} (%{100 * olas[i, p[i]]:.0f})",
                         fontsize=8, color="green" if satir == 0 else "red")
    fig.suptitle("Doğrulama kümesinden doğru (üst) ve yanlış (alt) sınıflandırılan örnekler", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3)
    fig.savefig(SEKIL / "dogru_yanlis_ornekleri.png", dpi=130)
    plt.close(fig)


def sekil_kontrol(m_ana, m_siluet, cogunluk, kaynak_m, embrapa=None):
    adlar = ["Ana model\n(renkli yaprak)", "Kontrol: yalnız\nsiluet", "Çoğunluk sınıfı\n(hep 'Sağlıklı')",
             "Kaynak dışı test:\nEmbrapa CBB (n=25)\n(C hiç görülmeden)"]
    emb = np.nan
    if embrapa:
        s = embrapa["sinif_bazinda"]["adi_yaprak_yanikligi"]
        emb = s["dogru"] / s["n"]
    dog = [m_ana["dogruluk"], m_siluet["dogruluk"] if m_siluet else np.nan, cogunluk["dogruluk"], emb]
    f1 = [m_ana["makro_f1"], m_siluet["makro_f1"] if m_siluet else np.nan, cogunluk["makro_f1"], np.nan]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    b1 = ax.bar(x - 0.18, np.array(dog) * 100, 0.36, label="Doğruluk", color="#3b78b5")
    b2 = ax.bar(x + 0.18, np.array(f1) * 100, 0.36, label="Makro F1", color="#e69138")
    for b in list(b1) + list(b2):
        if not np.isnan(b.get_height()):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, adlar)
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    t = "Kontrol deneyleri (ilk üçü doğrulama kümesi, 3 sınıf; sonuncu: yalnız doğruluk)"
    if kaynak_m:
        t += f"\nEk kontrol: sağlıklı yapraklarda kaynak (A/B) tahmini doğruluğu = %{100 * kaynak_m['dogruluk']:.1f}"
    ax.set_title(t, fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(SEKIL / "kontrol_deneyleri.png", dpi=130)
    plt.close(fig)


def main():
    SEKIL.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    dog = bolme_oku("dogrulama")
    rap = pd.read_csv(ISLENMIS / "on_isleme_raporu.csv")
    rap = rap[~rap["dislandi"]]
    adlar3 = [KISA_AD[s] for s in SINIFLAR]

    # --- Ana model ---
    model, ck = model_yukle(CIKTI / "model_en_iyi.pt")
    X, y, olas = tahmin_et(model, ck, dog)
    p = olas.argmax(1)
    ana = temel_metrikler(y, p, SINIFLAR)
    ana["kaynak_bazinda"] = kaynak_bazinda(dog, y, p)
    cm = np.array(ana["karisiklik_matrisi"]["matris"])
    ana["cbb_hale_karisikligi"] = {
        "cbb_iken_hale_denen": int(cm[0, 1]), "hale_iken_cbb_denen": int(cm[1, 0]),
        "cbb_toplam": int(cm[0].sum()), "hale_toplam": int(cm[1].sum()),
        "hastalikli_iken_saglikli_denen": int(cm[0, 2] + cm[1, 2]),
        "saglikli_iken_hastalikli_denen": int(cm[2, 0] + cm[2, 1]),
    }
    camler, cam_ozet = gradcam_analizi(model, ck, dog, X, p)
    ana["gradcam_nicel"] = cam_ozet
    egz = json.loads((CIKTI / "egitim_ozeti.json").read_text(encoding="utf-8"))
    pd.DataFrame({"dosya": dog.dosya, "sinif": dog.sinif, "kaynak": dog.kaynak, "tahmin": [SINIFLAR[i] for i in p],
                  **{f"olasilik_{s}": olas[:, i].round(4) for i, s in enumerate(SINIFLAR)}}).to_csv(
        CIKTI / "dogrulama_tahminleri.csv", index=False)

    # --- Basit taban çizgisi: hep çoğunluk sınıfı ---
    cogunluk = {"dogruluk": float((y == 2).mean()), "makro_f1": float(f1_score(y, np.full_like(y, 2), average="macro"))}

    # --- Kontrol 1: yalnız siluetle eğitilen aynı model ---
    siluet = None
    if (CIKTI / "kontrol_siluet" / "model_en_iyi.pt").exists():
        ms, cks = model_yukle(CIKTI / "kontrol_siluet" / "model_en_iyi.pt")
        _, ys, os_ = tahmin_et(ms, cks, dog)
        ps = os_.argmax(1)
        siluet = temel_metrikler(ys, ps, SINIFLAR)
        siluet["kaynak_bazinda"] = kaynak_bazinda(dog, ys, ps)
        es = json.loads((CIKTI / "kontrol_siluet" / "egitim_ozeti.json").read_text(encoding="utf-8"))
        siluet.update({"egitim_suresi_sn": es["egitim_suresi_sn"], "calisan_epoch": es["calisan_epoch"],
                       "en_iyi_epoch": es["en_iyi_epoch"]})
        sekil_karisiklik(siluet["karisiklik_matrisi"]["matris"], adlar3, "karisiklik_matrisi_siluet_kontrol.png",
                         f"Kontrol: yalnız siluet — doğruluk %{100 * siluet['dogruluk']:.1f}, "
                         f"makro F1 %{100 * siluet['makro_f1']:.1f}")
        sekil_egriler(es["gecmis"], es["en_iyi_epoch"], "egitim_egrileri_siluet_kontrol.png",
                      "Kontrol deneyi (yalnız siluet) eğitim eğrileri")

    # --- Kontrol 2: sağlıklı yapraklarda kaynak (A siyah kumaş / B beyaz zemin) tahmini ---
    kaynak_m = None
    if (CIKTI / "kontrol_kaynak" / "model_en_iyi.pt").exists():
        mk, ckk = model_yukle(CIKTI / "kontrol_kaynak" / "model_en_iyi.pt")
        dk = dog[(dog.sinif == "saglikli") & dog.kaynak.isin(["mendeley_ykvcrjffzd", "mendeley_3km9d246z2"])]
        dk = dk.reset_index(drop=True)
        _, yk, ok = tahmin_et(mk, ckk, dk)
        kaynak_m = temel_metrikler(yk, ok.argmax(1), ["A_siyah_kumas", "B_beyaz_zemin"])
        ek = json.loads((CIKTI / "kontrol_kaynak" / "egitim_ozeti.json").read_text(encoding="utf-8"))
        kaynak_m.update({"egitim_ornek": ek["egitim_ornek"], "calisan_epoch": ek["calisan_epoch"]})

    # --- Kontrol 3: Embrapa (C) hiç görülmeden eğitilen model, C'nin TÜM görüntülerinde (kaynak dışı test) ---
    embrapa = None
    if (CIKTI / "kontrol_embrapa_haric" / "model_en_iyi.pt").exists():
        me, cke = model_yukle(CIKTI / "kontrol_embrapa_haric" / "model_en_iyi.pt")
        tum = pd.concat([bolme_oku("egitim"), bolme_oku("dogrulama")])
        dc = tum[tum.kaynak == "embrapa_pddb"].reset_index(drop=True)
        _, yc, oc = tahmin_et(me, cke, dc)
        pc = oc.argmax(1)
        embrapa = {"n": int(len(yc)), "dogruluk": float((yc == pc).mean()),
                   "sinif_bazinda": {SINIFLAR[c]: {"n": int((yc == c).sum()),
                                                   "dogru": int(((yc == c) & (pc == c)).sum()),
                                                   "tahmin_dagilimi": {SINIFLAR[j]: int(((yc == c) & (pc == j)).sum())
                                                                       for j in range(3)}}
                                     for c in sorted(set(yc.tolist()))}}
        # Aynı modelin A+B doğrulama kümesindeki başarımı (karşılaştırma için)
        dab = dog[dog.kaynak != "embrapa_pddb"].reset_index(drop=True)
        _, yab, oab = tahmin_et(me, cke, dab)
        embrapa["ayni_model_AB_dogrulama"] = {"dogruluk": float((yab == oab.argmax(1)).mean()),
                                              "makro_f1": float(f1_score(yab, oab.argmax(1), average="macro"))}

    # --- Şekiller ---
    sekil_sinif_ornekleri(rap)
    sekil_on_isleme(rap)
    sekil_egriler(egz["gecmis"], egz["en_iyi_epoch"], "egitim_egrileri.png",
                  "FasulyeCNN eğitim / doğrulama kayıp ve doğruluk eğrileri")
    sekil_karisiklik(cm, adlar3, "karisiklik_matrisi.png",
                     f"FasulyeCNN doğrulama karışıklık matrisi (n={len(y)})\n"
                     f"doğruluk %{100 * ana['dogruluk']:.1f}, makro F1 %{100 * ana['makro_f1']:.1f}")
    sekil_gradcam(dog, X, y, p, olas, camler)
    sekil_dogru_yanlis(dog, X, y, p, olas)
    sekil_kontrol(ana, siluet, cogunluk, kaynak_m, embrapa)

    # --- metrikler.json ---
    bolme = json.loads((CIKTI / "bolme_ozeti.json").read_text(encoding="utf-8"))
    sid = json.loads((CIKTI / "siddet_ozeti.json").read_text(encoding="utf-8")) if (CIKTI / "siddet_ozeti.json").exists() else None
    ek_modeller = {}
    for yol in CIKTI.glob("ek_*/metrikler_ek.json"):
        ek_modeller[yol.parent.name] = json.loads(yol.read_text(encoding="utf-8"))
    metr = {
        "ana_model": {
            "ad": "FasulyeCNN (sıfırdan, 16 katman)", "giris_boyutu": GIRIS_BOYUT,
            "parametre_sayisi": egz["parametre_sayisi"], "egitim_suresi_sn": egz["egitim_suresi_sn"],
            "calisan_epoch": egz["calisan_epoch"], "en_iyi_epoch": egz["en_iyi_epoch"],
            "egitim_ayarlari": egz["ayarlar"], "kayip_agirliklari": egz["kayip_agirliklari"],
            "egitim_sinif_sayilari": egz["egitim_sinif_sayilari"], "cihaz": egz["cihaz"],
            "en_iyi_epoch_egitim_dogrulugu": egz["gecmis"][egz["en_iyi_epoch"] - 1]["egitim_dogruluk"],
            "dogrulama": ana,
        },
        "taban_cizgisi_cogunluk_sinifi": cogunluk,
        "kontrol_siluet": siluet,
        "kontrol_kaynak_tahmini_saglikli": kaynak_m,
        "kontrol_embrapa_kaynak_disi_test": embrapa,
        "veri_bolme": bolme,
        "siddet_tahmini": sid,
        "ek_modeller": ek_modeller or None,
    }
    (CIKTI / "metrikler.json").write_text(json.dumps(metr, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dogruluk": ana["dogruluk"], "makro_f1": ana["makro_f1"],
                      "siluet": siluet and {"dogruluk": siluet["dogruluk"], "makro_f1": siluet["makro_f1"]},
                      "kaynak": kaynak_m and kaynak_m["dogruluk"], "cogunluk": cogunluk}, ensure_ascii=False, indent=1))
    print(json.dumps(ana["kaynak_bazinda"], ensure_ascii=False, indent=1))
    print(json.dumps(cam_ozet, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
