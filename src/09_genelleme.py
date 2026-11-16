# -*- coding: utf-8 -*-
"""
09 - KAYNAK DIŞI GENELLEME deneyleri (transfer öğrenmesi + kaynak imzasını kırmaya yönelik eğitim).

Dürüstlük kuralı: Kaynak C (Embrapa) bu betikte eğitimde, doğrulamada, erken durdurmada ve yöntem
seçiminde KULLANILMAZ. C yalnızca `--c_testi` bayrağıyla, en sonda, önceden seçilmiş yapılandırmalar
için okunur ve her okuma `ciktilar/genel_model/c_bakis_kaydi.jsonl` dosyasına kaydedilir.
Erken durdurma yok: sabit epoch sayısı + kosinüs öğrenme oranı, SON epoch'un ağırlıkları kullanılır.

Vekil (C'siz) kaynak-dışı protokoller (`--protokol`):
  tam            : A+B eğitim (+ `--web` ise D eğitim) -> A+B doğrulama (+ D doğrulama). `--web` yoksa
                   D (PlantWild web) hiç görülmez ve tamamı kaynak-dışı vekil test olarak ölçülür
                   (hale-D: hiç görülmemiş kaynaktan hastalık; C'deki CBB sorununun aynası).
  haric_saglikli_A: sağlıklı-A eğitimden çıkarılır; doğrulamadaki sağlıklı-A'nın kaç tanesinin "sağlıklı"
                   denildiği ölçülür. Kısayol öğrenen model "A -> CBB" der.
  haric_saglikli_B: aynısı sağlıklı-B için (kısayol: "B -> hale").
Seçim ölçütü (önceden belirlendi, bkz. RESULTS §12): vekil skor = ortalama(sağlıklı-A tutma, sağlıklı-B
tutma, hale-D duyarlılığı); A+B doğrulama makro F1 >= 0.95 koşuluyla.

Kullanım örnekleri:
  python src/09_genelleme.py --ad b0_temel --omurga efficientnet_b0 --artirma temel
  python src/09_genelleme.py --ad b0_guclu --artirma guclu --protokol haric_saglikli_A
  python src/09_genelleme.py --ad son --artirma guclu --web --kaydet --c_testi
"""
import argparse
import importlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, SINIFLAR, TOHUM, VERI, kaynak_kisa, tohum_ayarla  # noqa: E402
from veri import bolme_oku, etiketler, goruntuleri_yukle, maske_uygula  # noqa: E402

C_KAYNAK = "embrapa_pddb"
WEB_KAYNAK = "plantwild_web"
BUGWOOD_KAYNAK = "bugwood_ipm"
SELCUK_KAYNAK = "selcuk_bbb"
IMAGENET_ORT, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
KLASOR = CIKTI / "genel_model"


def kaynak_kodu(k):
    return {"mendeley_ykvcrjffzd": "A", "mendeley_3km9d246z2": "B", C_KAYNAK: "C", WEB_KAYNAK: "D",
            BUGWOOD_KAYNAK: "E", SELCUK_KAYNAK: "F"}[k]


# ----------------------------------------------------------------------------- girdi normalizasyonu
def isik_normalize(rgb01, m, hedef=0.45):
    """Maske içindeki ortalama parlaklığı sabit bir değere ölçekler (aydınlatma/pozlama farkını azaltır).
    rgb01 [B,3,H,W] veya [3,H,W] float 0-1, m aynı boyutlu (1 kanal) 0/1."""
    tek = rgb01.dim() == 3
    if tek:
        rgb01, m = rgb01[None], m[None]
    lum = (0.299 * rgb01[:, 0] + 0.587 * rgb01[:, 1] + 0.114 * rgb01[:, 2])[:, None]
    ort = (lum * m).sum((1, 2, 3), keepdim=True) / m.sum((1, 2, 3), keepdim=True).clamp(min=1)
    out = (rgb01 * (hedef / ort.clamp(min=0.02))).clamp(0, 1) * m
    return out[0] if tek else out


# ----------------------------------------------------------------------------- veri artırma
class GucluRenk:
    """Kaynak imzasını (kamera rengi, beyaz dengesi, pozlama, keskinlik, sıkıştırma) bozmaya yönelik
    rastgele foto-metrik dönüşümler. uint8 [3,H,W] alır/döndürür. Ton (hue) sınırlı tutulur: hastalık
    rengi (sarı hale, kahverengi nekroz) bilgi taşır."""

    def __init__(self):
        self.jit = v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.04)
        self.blur = v2.GaussianBlur(5, sigma=(0.1, 1.8))
        self.jpeg = v2.JPEG((25, 90))

    def __call__(self, x):
        r = torch.rand(8)
        x = self.jit(x)
        f = x.float() / 255.0
        if r[0] < 0.8:  # gamma
            f = f.clamp(1e-4, 1) ** float(math.exp(np.random.uniform(-0.4, 0.4)))
        if r[1] < 0.8:  # beyaz dengesi / kamera rengi: kanal kazançları
            f = f * torch.empty(3, 1, 1).uniform_(0.8, 1.2)
        if r[2] < 0.15:  # gri
            f = (0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]).expand(3, -1, -1).clone()
        if r[3] < 0.3:  # çözünürlük kaybı
            H = f.shape[-1]
            s = int(H * np.random.uniform(0.3, 0.7))
            f = F.interpolate(F.interpolate(f[None], size=s, mode="bilinear", antialias=True), size=H,
                              mode="bilinear")[0]
        if r[4] < 0.3:  # sensör gürültüsü
            f = f + torch.randn_like(f) * np.random.uniform(0.005, 0.03)
        x = (f.clamp(0, 1) * 255).round().to(torch.uint8)
        if r[5] < 0.3:
            x = self.blur(x)
        if r[6] < 0.3:
            x = self.jpeg(x)
        return x


class GenelVeri(Dataset):
    """X uint8 [N,4,H,W] (RGB+maske). Eğitimde geometri + renk artırma; çıkış ImageNet normalize."""

    def __init__(self, X, y, k, egitim, artirma="temel", isik=False, boyut=224):
        self.X, self.y, self.k = X, torch.as_tensor(y, dtype=torch.long), torch.as_tensor(k, dtype=torch.long)
        self.egitim, self.isik = egitim, isik
        self.ort = torch.tensor(IMAGENET_ORT).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        olcek = (0.75, 1.0) if artirma == "temel" else (0.5, 1.0)
        self.geo = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(180, fill=0),
                               v2.RandomResizedCrop(boyut, scale=olcek, ratio=(0.85, 1.15), antialias=True)])
        self.renk = (v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)
                     if artirma == "temel" else GucluRenk())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.egitim:
            x = self.geo(x)
            rgb, m = self.renk(x[:3]), x[3:4]
        else:
            rgb, m = x[:3], x[3:4]
        rgb = maske_uygula(rgb, m).float() / 255.0
        if self.isik:
            rgb = isik_normalize(rgb, (m > 127).float())
        return (rgb - self.ort) / self.std, self.y[i], self.k[i]


# ----------------------------------------------------------------------------- model
class GradTers(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l):
        ctx.l = l
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.l * g, None


class GenelModel(nn.Module):
    """Ön eğitimli omurga + 3 sınıflı baş (+ isteğe bağlı kaynak başı, gradyan tersine çevirme ile)."""

    def __init__(self, omurga="efficientnet_b0", sinif=3, kaynak_sayisi=3, onegitimli=True):
        super().__init__()
        import torchvision.models as tm
        self.omurga_adi = omurga
        if omurga.startswith("efficientnet"):
            ag = getattr(tm, omurga)(weights="DEFAULT" if onegitimli else None)
            boyut = ag.classifier[1].in_features
            self.govde = nn.Sequential(ag.features, ag.avgpool, nn.Flatten())
        elif omurga.startswith("convnext"):
            ag = getattr(tm, omurga)(weights="DEFAULT" if onegitimli else None)
            boyut = ag.classifier[2].in_features
            self.govde = nn.Sequential(ag.features, ag.avgpool, ag.classifier[0], ag.classifier[1])
        elif omurga == "fasulye_cnn":
            # 03_model.FasulyeCNN ile aynı 16 katman (ön eğitim YOK); baş = son Dropout + Linear(128, 3)
            fc = importlib.import_module("03_model").FasulyeCNN(sinif_sayisi=sinif)
            boyut = 128
            self.govde = nn.Sequential(fc.ozellik, fc.son_evrisim, fc.gap, *list(fc.siniflayici)[:3])
            self.bas = nn.Sequential(*list(fc.siniflayici)[3:])
            self.kaynak_bas = nn.Sequential(nn.Linear(boyut, 256), nn.ReLU(), nn.Linear(256, kaynak_sayisi))
            return
        elif omurga.startswith("resnet"):
            ag = getattr(tm, omurga)(weights="DEFAULT" if onegitimli else None)
            boyut = ag.fc.in_features
            ag.fc = nn.Identity()
            self.govde = ag
        else:
            raise ValueError(omurga)
        self.bas = nn.Sequential(nn.Dropout(0.3), nn.Linear(boyut, sinif))
        self.kaynak_bas = nn.Sequential(nn.Linear(boyut, 256), nn.ReLU(), nn.Linear(256, kaynak_sayisi))

    def forward(self, x, grl=0.0):
        z = self.govde(x)
        c = self.bas(z)
        if grl > 0:
            return c, self.kaynak_bas(GradTers.apply(z, grl))
        return c


def model_kur(omurga, onegitimli=True):
    return GenelModel(omurga, onegitimli=onegitimli)


# ----------------------------------------------------------------------------- yardımcılar
def web_oku():
    yol = VERI / "islenmis_web" / "web_veri.csv"
    return pd.read_csv(yol) if yol.exists() else None


def bugwood_oku():
    yol = VERI / "islenmis_bugwood" / "bugwood_veri.csv"
    return pd.read_csv(yol, dtype={"numara": str}) if yol.exists() else None


def selcuk_oku():
    yol = VERI / "islenmis_selcuk" / "selcuk_veri.csv"
    return pd.read_csv(yol) if yol.exists() else None


def selcuk_bol(f, tohum=TOHUM):
    """F (Selçuk) gruplu, sınıf x çeşit katmanlı %70/%30 (12_selcuk_veri.gruplu_bol). Sabit tohum (eğitim tohumundan
    bağımsız): her eğitim tohumunda aynı bölme."""
    return importlib.import_module("12_selcuk_veri").gruplu_bol(f, 0.3, tohum)


def grup_bol_tam(df, oran, tohum=TOHUM):
    """Az sayıda büyük grup için (E): toplamı oran*N'ye en yakın grup altkümelerinden biri (tohumla seçilir)
    ikinci parçaya gider. Döner (birinci, ikinci). Grup sayısı <= 16 varsayılır (tüm altkümeler denenir)."""
    gr = sorted(set(df.grup))
    if len(gr) > 16:
        raise ValueError("grup_bol_tam en çok 16 grup içindir")
    n = df.grup.value_counts().reindex(gr).to_numpy()
    hedef = oran * len(df)
    altk = [[g for j, g in enumerate(gr) if (b >> j) & 1] for b in range(1, 2 ** len(gr) - 1)]
    fark = [abs(sum(n[j] for j in range(len(gr)) if (b >> j) & 1) - hedef) for b in range(1, 2 ** len(gr) - 1)]
    en_iyi = [a for a, f in zip(altk, fark) if f == min(fark)]
    sec = en_iyi[np.random.default_rng(tohum).integers(len(en_iyi))]
    m = df.grup.isin(sec)
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)


def bugwood_yari(e, tohum=TOHUM):
    """E (Bugwood CBB) gruplu olarak ikiye: (eğitim yarısı, vekil test yarısı). Bölme her zaman sabit tohumla
    (eğitim tohumundan bağımsız) yapılır; test yarısı vekil varyantta eğitimde ve doğrulamada hiç yoktur."""
    return grup_bol_tam(e, 0.5, tohum)


def c_kaydi_yaz(kayit, yol=None):
    """C (Embrapa) bakış kaydına bir satır ekler (zaman otomatik). C okunmadan önce 'basladi', sonuçla 'tamamlandi'."""
    yol = yol or KLASOR / "c_bakis_kaydi.jsonl"
    with open(yol, "a", encoding="utf-8") as f:
        f.write(json.dumps({"zaman": datetime.now().isoformat(timespec="seconds"), **kayit}, ensure_ascii=False) + "\n")


def web_bol(web, tohum=TOHUM, oran=0.3):
    """D (web) için grup bazlı, sınıf katmanlı %70/%30 bölme (A/B bölmesiyle aynı mantık)."""
    rng = np.random.default_rng(tohum)
    dog = []
    for s, g in web.groupby("sinif"):
        gr = sorted(set(g.grup))
        rng.shuffle(gr)
        hedef, n = oran * len(g), 0
        for x in gr:
            if n >= hedef:
                break
            dog.append(x)
            n += int((g.grup == x).sum())
    m = web.grup.isin(dog)
    egt, dgr = web[~m].reset_index(drop=True), web[m].reset_index(drop=True)
    # Algoritma değişmedi; yalnız büyük gruplar bir tarafı boş bırakmasın diye denetlenir (sınıf bazında).
    for s, g in web.groupby("sinif"):
        n_dog = int(m[g.index].sum())
        if n_dog == 0 or n_dog == len(g):
            raise ValueError(f"web_bol: '{s}' sınıfında {len(g) - n_dog} eğitim / {n_dog} doğrulama örneği çıktı; "
                             f"büyük gruplar bir tarafı boş bırakıyor (grup sayısı: {g.grup.nunique()}, oran={oran}). "
                             "Grupları küçültün/oranı ya da tohumu değiştirin.")
    return egt, dgr


def cesit_bazinda(df, y, P):
    """F için (sınıf|çeşit) bazında doğru/n ve tahmin dağılımı."""
    p = P.argmax(1)
    d = df.assign(y=y, p=p)
    return {f"{s}|{c}": {"n": int(len(g)), "dogru": int((g.y == g.p).sum()), "dogruluk": float((g.y == g.p).mean()),
                         "tahmin": [int((g.p == i).sum()) for i in range(3)]}
            for (s, c), g in d.groupby(["sinif", "cesit"])}


def tahminler(model, X, isik, cihaz, bs=64, tta=False):
    ds = GenelVeri(X, np.zeros(len(X), int), np.zeros(len(X), int), egitim=False, isik=isik)
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(cihaz.type, enabled=cihaz.type == "cuda"):
        for i in range(0, len(ds), bs):
            x = torch.stack([ds[j][0] for j in range(i, min(i + bs, len(ds)))]).to(cihaz)
            p = F.softmax(model(x).float(), 1)
            if tta:
                p = p + F.softmax(model(torch.flip(x, [3])).float(), 1) + F.softmax(model(torch.flip(x, [2])).float(), 1)
                p = p / 3
            out.append(p.cpu())
    return torch.cat(out).numpy()


def metrik(df, y, P):
    p = P.argmax(1)
    d = df.assign(y=y, p=p)
    kb = {}
    for (s, k), g in d.groupby(["sinif", "kaynak"]):
        kb[f"{s}|{kaynak_kodu(k)}"] = {"n": int(len(g)), "dogru": int((g.y == g.p).sum()),
                                       "dogruluk": float((g.y == g.p).mean()),
                                       "tahmin": [int((g.p == i).sum()) for i in range(3)]}
    sinif_var = sorted(set(y.tolist()))
    return {"n": int(len(y)), "dogruluk": float((p == y).mean()),
            "makro_f1": float(f1_score(y, p, labels=sinif_var, average="macro")),
            "karisiklik": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(), "kaynak_bazinda": kb}


# ----------------------------------------------------------------------------- ana akış
def calistir(a):
    tohum_ayarla(a.tohum)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kl = KLASOR / "deneyler" / f"{a.ad}_{a.protokol}_t{a.tohum}"
    kl.mkdir(parents=True, exist_ok=True)
    log = open(kl / "log.txt", "w", encoding="utf-8")

    def yaz(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    egt, dog = bolme_oku("egitim"), bolme_oku("dogrulama")
    egt, dog = egt[egt.kaynak != C_KAYNAK], dog[dog.kaynak != C_KAYNAK]  # C hiçbir zaman burada yok
    egt, dog = egt.assign(cesit=""), dog.assign(cesit="")
    web = web_oku()
    web_egt = web_dog = None
    if web is not None:
        web_egt, web_dog = web_bol(web)
    if a.web:
        egt = pd.concat([egt, web_egt.reindex(columns=egt.columns, fill_value="")], ignore_index=True)
        dog = pd.concat([dog, web_dog.reindex(columns=dog.columns, fill_value="")], ignore_index=True)
    e_test = None
    if a.bugwood != "yok":
        bw = bugwood_oku()
        assert bw is not None, "önce src/11_bugwood_veri.py çalıştırılmalı"
        if a.bugwood == "yari":  # vekil: E'nin yarısı eğitimde, diğer yarısı tamamen dışarıda (kaynak-dışı CBB testi)
            e_egt, e_test = bugwood_yari(bw)
            egt = pd.concat([egt, e_egt.reindex(columns=egt.columns, fill_value="")], ignore_index=True)
        else:  # tam: E de %70/%30 gruplu bölünür
            e_egt, e_dog = grup_bol_tam(bw, 0.3)
            egt = pd.concat([egt, e_egt.reindex(columns=egt.columns, fill_value="")], ignore_index=True)
            dog = pd.concat([dog, e_dog.reindex(columns=dog.columns, fill_value="")], ignore_index=True)
    f_test = None
    if a.selcuk != "yok":  # EK 3: F (Selçuk) %70/%30 gruplu; haric_<cesit>: o çeşit eğitim/doğrulamadan tamamen çıkar
        fs = selcuk_oku()
        assert fs is not None, "önce src/12_selcuk_veri.py çalıştırılmalı"
        f_egt, f_dog = selcuk_bol(fs)
        if a.selcuk.startswith("haric_"):
            cs = a.selcuk[len("haric_"):]
            assert cs in set(fs.cesit), cs
            f_test = fs[fs.cesit == cs].reset_index(drop=True)
            f_egt, f_dog = f_egt[f_egt.cesit != cs], f_dog[f_dog.cesit != cs]
        egt = pd.concat([egt, f_egt.reindex(columns=egt.columns)], ignore_index=True)
        dog = pd.concat([dog, f_dog.reindex(columns=dog.columns)], ignore_index=True)
    tutulan = None
    if a.protokol.startswith("haric_saglikli_"):
        kay = {"A": "mendeley_ykvcrjffzd", "B": "mendeley_3km9d246z2"}[a.protokol[-1]]
        sec = lambda d: (d.sinif == "saglikli") & (d.kaynak == kay)  # noqa: E731
        tutulan = dog[sec(dog)].reset_index(drop=True)
        egt, dog = egt[~sec(egt)], dog[~sec(dog)]
    egt, dog = egt.reset_index(drop=True), dog.reset_index(drop=True)
    assert not (egt.kaynak == C_KAYNAK).any() and not (dog.kaynak == C_KAYNAK).any()

    kaynak_idx = {k: i for i, k in enumerate(["mendeley_ykvcrjffzd", "mendeley_3km9d246z2", WEB_KAYNAK])}
    kaynak_idx[BUGWOOD_KAYNAK] = 2  # kaynak başı yalnız GRL'de (sağlıklılarda) kullanılır; E'de sağlıklı yok
    kaynak_idx[SELCUK_KAYNAK] = 2  # GRL bu çalışmada kullanılmıyor; yalnız indeks geçerli olsun diye
    Xe = goruntuleri_yukle(egt, boyut=a.boyut)
    ye, ke = etiketler(egt), egt.kaynak.map(kaynak_idx).to_numpy()
    de = GenelVeri(Xe, ye, ke, egitim=True, artirma=a.artirma, isik=a.isik, boyut=a.boyut)
    sayi = np.bincount(ye, minlength=3)
    if a.denge == "hucre":
        # her sınıf eşit toplam olasılık; sınıf içinde her kaynak eşit pay
        hucre = egt.groupby(["sinif", "kaynak"]).size()
        ks = egt.groupby("sinif").kaynak.nunique()
        w = np.array([1.0 / (3 * ks[s] * hucre[(s, k)]) for s, k in zip(egt.sinif, egt.kaynak)])
        ornekleyici = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True,
                                            generator=torch.Generator().manual_seed(a.tohum))
        yuk = DataLoader(de, batch_size=a.batch, sampler=ornekleyici, num_workers=0, drop_last=True)
        kayip_w = None
    else:
        yuk = DataLoader(de, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True,
                         generator=torch.Generator().manual_seed(a.tohum))
        kayip_w = torch.tensor(len(ye) / (3 * np.maximum(sayi, 1)), dtype=torch.float32, device=cihaz)
    yaz(f"{a} | eğitim {len(egt)} {sayi.tolist()} | doğrulama {len(dog)}")
    yaz("eğitim hücreleri: " + str(egt.groupby(["sinif", "kaynak"]).size().to_dict()))

    model = model_kur(a.omurga).to(cihaz)
    kayip_f = nn.CrossEntropyLoss(weight=kayip_w, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    adim = a.epoch * len(yuk)
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=adim, pct_start=0.1)
    olcek = torch.amp.GradScaler(enabled=cihaz.type == "cuda")
    t0 = time.time()
    for ep in range(1, a.epoch + 1):
        model.train()
        tk, tn = 0.0, 0
        for x, y, k in yuk:
            x, y, k = x.to(cihaz), y.to(cihaz), k.to(cihaz)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(cihaz.type, enabled=cihaz.type == "cuda"):
                if a.grl > 0:
                    c, kc = model(x, grl=a.grl)
                    kay = kayip_f(c.float(), y)
                    s = y == SINIFLAR.index("saglikli")  # kaynak karşıtlığı yalnız sağlıklılarda (sınıf-koşullu)
                    if s.sum() > 1:
                        kay = kay + F.cross_entropy(kc[s].float(), k[s])
                else:
                    kay = kayip_f(model(x).float(), y)
            olcek.scale(kay).backward()
            olcek.step(opt)
            olcek.update()
            plan.step()
            tk += kay.item() * len(y)
            tn += len(y)
        yaz(f"epoch {ep:2d} kayıp {tk / tn:.4f} | {time.time() - t0:.0f} sn")

    sonuc = {"ayarlar": vars(a), "egitim_n": int(len(egt)), "egitim_sinif": sayi.tolist(),
             "sure_sn": round(time.time() - t0, 1)}
    Xd = goruntuleri_yukle(dog, boyut=a.boyut)
    Pd = tahminler(model, Xd, a.isik, cihaz)
    yd = etiketler(dog)
    sonuc["dogrulama"] = metrik(dog, yd, Pd)
    ab = dog.kaynak.isin(["mendeley_ykvcrjffzd", "mendeley_3km9d246z2"]).to_numpy()
    sonuc["dogrulama_AB"] = metrik(dog[ab], yd[ab], Pd[ab])
    fm = (dog.kaynak == SELCUK_KAYNAK).to_numpy()
    if fm.any():
        sonuc["dogrulama_F"] = metrik(dog[fm], yd[fm], Pd[fm])
        sonuc["dogrulama_F_cesit"] = cesit_bazinda(dog[fm], yd[fm], Pd[fm])
    pd.DataFrame(Pd, columns=SINIFLAR).assign(dosya=dog.dosya, sinif=dog.sinif, kaynak=dog.kaynak).to_csv(
        kl / "dogrulama_olasiliklar.csv", index=False)
    if tutulan is not None:
        P = tahminler(model, goruntuleri_yukle(tutulan, boyut=a.boyut), a.isik, cihaz)
        sonuc["tutulan_saglikli"] = metrik(tutulan, etiketler(tutulan), P)
        sonuc["vekil_tutma"] = sonuc["tutulan_saglikli"]["dogruluk"]
    if e_test is not None:
        P = tahminler(model, goruntuleri_yukle(e_test, boyut=a.boyut), a.isik, cihaz)
        sonuc["bugwood_vekil_test"] = metrik(e_test, etiketler(e_test), P)
        sonuc["vekil_E_CBB"] = sonuc["bugwood_vekil_test"]["dogruluk"]
        sonuc["vekil_E_hastalikli"] = float((P.argmax(1) != SINIFLAR.index("saglikli")).mean())
        pd.DataFrame(P, columns=SINIFLAR).assign(dosya=e_test.dosya, grup=e_test.grup).to_csv(
            kl / "bugwood_vekil_olasiliklar.csv", index=False)
    if f_test is not None:
        P = tahminler(model, goruntuleri_yukle(f_test, boyut=a.boyut), a.isik, cihaz)
        yf = etiketler(f_test)
        sonuc["selcuk_cesit_disi"] = metrik(f_test, yf, P)
        cbb = yf == SINIFLAR.index("adi_yaprak_yanikligi")
        sonuc["vekil_F_CBB"] = float((P.argmax(1)[cbb] == yf[cbb]).mean())
        sonuc["vekil_F_CBB_hastalikli"] = float((P.argmax(1)[cbb] != SINIFLAR.index("saglikli")).mean())
        sonuc["vekil_F_saglikli"] = float((P.argmax(1)[~cbb] == yf[~cbb]).mean())
        pd.DataFrame(P, columns=SINIFLAR).assign(dosya=f_test.dosya, sinif=f_test.sinif, cesit=f_test.cesit).to_csv(
            kl / "selcuk_cesit_disi_olasiliklar.csv", index=False)
    if web is not None and not a.web:
        P = tahminler(model, goruntuleri_yukle(web, boyut=a.boyut), a.isik, cihaz)
        sonuc["web_kaynak_disi"] = metrik(web, etiketler(web), P)
        pd.DataFrame(P, columns=SINIFLAR).assign(dosya=web.dosya, sinif=web.sinif).to_csv(
            kl / "web_olasiliklar.csv", index=False)
    yaz("A+B doğrulama: " + json.dumps({k: sonuc["dogrulama_AB"][k] for k in ("n", "dogruluk", "makro_f1")}))
    if "dogrulama_F" in sonuc:
        yaz("F doğrulama: " + json.dumps({k: sonuc["dogrulama_F"][k] for k in ("n", "dogruluk", "makro_f1")}))
    for anahtar in ("tutulan_saglikli", "web_kaynak_disi", "bugwood_vekil_test", "selcuk_cesit_disi"):
        if anahtar in sonuc:
            yaz(f"{anahtar}: " + json.dumps(sonuc[anahtar]["kaynak_bazinda"], ensure_ascii=False))

    if a.kaydet or a.c_testi:
        torch.save({"model": model.state_dict(), "omurga": a.omurga, "boyut": a.boyut, "isik": a.isik,
                    "ortalama": IMAGENET_ORT, "std": IMAGENET_STD, "siniflar": SINIFLAR,
                    "ayarlar": vars(a)}, kl / "model.pt")
    if a.c_testi:
        c_kaydi_yaz({"durum": "basladi", "deney": kl.name, "model": str(kl / "model.pt")})  # okumadan ÖNCE deneme kaydı
        tum = pd.concat([bolme_oku("egitim"), bolme_oku("dogrulama")], ignore_index=True)
        dc = tum[tum.kaynak == C_KAYNAK].reset_index(drop=True)
        Pc = tahminler(model, goruntuleri_yukle(dc, boyut=a.boyut), a.isik, cihaz)
        sonuc["C_test"] = metrik(dc, etiketler(dc), Pc)
        pd.DataFrame(Pc, columns=SINIFLAR).assign(dosya=dc.dosya, sinif=dc.sinif).to_csv(
            kl / "C_olasiliklar.csv", index=False)
        with open(KLASOR / "c_bakis_kaydi.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"zaman": datetime.now().isoformat(timespec="seconds"), "durum": "tamamlandi",
                                "deney": kl.name,
                                "C_CBB_dogru": sonuc["C_test"]["kaynak_bazinda"].get("adi_yaprak_yanikligi|C", {}).get("dogru"),
                                "C_toplam_dogru": int((Pc.argmax(1) == etiketler(dc)).sum())}, ensure_ascii=False) + "\n")
        yaz("C TESTİ: " + json.dumps(sonuc["C_test"]["kaynak_bazinda"], ensure_ascii=False))
    (kl / "sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    log.close()
    return sonuc


def arg_ayristirici():
    p = argparse.ArgumentParser()
    p.add_argument("--ad", required=True)
    p.add_argument("--protokol", default="tam", choices=["tam", "haric_saglikli_A", "haric_saglikli_B"])
    p.add_argument("--omurga", default="efficientnet_b0")
    p.add_argument("--artirma", default="temel", choices=["temel", "guclu"])
    p.add_argument("--isik", action="store_true", help="maske içi parlaklık normalizasyonu (girişte)")
    p.add_argument("--denge", default="kayip", choices=["kayip", "hucre"])
    p.add_argument("--grl", type=float, default=0.0, help="sağlıklılarda kaynak karşıtı baş ağırlığı (0=kapalı)")
    p.add_argument("--web", action="store_true", help="D (PlantWild web) eğitim bölmesini eğitime kat")
    p.add_argument("--bugwood", default="yok", choices=["yok", "yari", "tam"],
                   help="E (Bugwood CBB): yari = yarısı eğitimde, yarısı vekil kaynak-dışı test; tam = %%70/%%30")
    p.add_argument("--selcuk", default="yok", choices=["yok", "tam", "haric_akbulut", "haric_ustun42"],
                   help="F (Selçuk CBB/sağlıklı): tam = %%70/%%30 gruplu; haric_<cesit> = o çeşit tamamen dışarıda "
                        "(vekil 'aynı kaynak, farklı çeşit' testi)")
    p.add_argument("--boyut", type=int, default=224)
    p.add_argument("--epoch", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tohum", type=int, default=TOHUM)
    p.add_argument("--kaydet", action="store_true")
    p.add_argument("--c_testi", action="store_true", help="SON TEST: C'yi oku (kaydı tutulur)")
    return p


if __name__ == "__main__":
    calistir(arg_ayristirici().parse_args())
