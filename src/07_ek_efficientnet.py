# -*- coding: utf-8 -*-
"""
07 - İSTEĞE BAĞLI EK: transfer öğrenmesi karşılaştırması (EfficientNet-B0, ImageNet ağırlıkları).

ANA TESLİMAT DEĞİLDİR; yalnız derste gösterilen ikinci yaklaşımla karşılaştırma içindir.
Ağırlıklar torchvision'ın resmi adresinden (download.pytorch.org) iner.
Aynı bölme, aynı ön işlenmiş görüntüler, aynı veri artırma ve sınıf ağırlıklı kayıp; giriş
224x224 ve ImageNet normalizasyonu. Tüm ağ ince ayarlanır (AdamW lr 3e-4), en fazla 40 epoch,
erken durdurma sabrı 8.
Çıktı: ciktilar/ek_efficientnet_b0/{model_en_iyi.pt, egitim_logu.txt, metrikler_ek.json}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, SINIFLAR, TOHUM, tohum_ayarla, kaynak_kisa  # noqa: E402
import veri  # noqa: E402

BOYUT = 224


def main():
    tohum_ayarla()
    veri.GIRIS_BOYUT = BOYUT  # veri artırmadaki kırpma boyutu
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kl = CIKTI / "ek_efficientnet_b0"
    kl.mkdir(parents=True, exist_ok=True)
    log = open(kl / "egitim_logu.txt", "w", encoding="utf-8")

    def yaz(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    egt, dog = veri.bolme_oku("egitim"), veri.bolme_oku("dogrulama")
    Xe, Xd = veri.goruntuleri_yukle(egt, boyut=BOYUT), veri.goruntuleri_yukle(dog, boyut=BOYUT)
    ye, yd = veri.etiketler(egt), veri.etiketler(dog)
    ort, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    de = veri.YaprakVeri(Xe, ye, ort, std, egitim=True)
    de.geo.transforms[-1] = veri.v2.RandomResizedCrop(BOYUT, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True)
    dd = veri.YaprakVeri(Xd, yd, ort, std, egitim=False)
    ge = DataLoader(de, batch_size=32, shuffle=True, num_workers=0, drop_last=True,
                    generator=torch.Generator().manual_seed(TOHUM))
    gd = DataLoader(dd, batch_size=64, shuffle=False, num_workers=0)

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model = model.to(cihaz)
    n_param = sum(p.numel() for p in model.parameters())
    say = np.bincount(ye, minlength=3)
    kayip_f = nn.CrossEntropyLoss(weight=torch.tensor(len(ye) / (3 * say), dtype=torch.float32, device=cihaz))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    en_iyi, en_iyi_ep, sabir, gecmis = 1e9, 0, 0, []
    t0 = time.time()
    for ep in range(1, 41):
        model.train()
        for x, y in ge:
            x, y = x.to(cihaz), y.to(cihaz)
            opt.zero_grad()
            k = kayip_f(model(x), y)
            k.backward()
            opt.step()
        model.eval()
        dk, olas = 0.0, []
        with torch.no_grad():
            for x, y in gd:
                c = model(x.to(cihaz))
                dk += kayip_f(c, y.to(cihaz)).item() * len(y)
                olas.append(F.softmax(c, 1).cpu())
        dk /= len(yd)
        p = torch.cat(olas).argmax(1).numpy()
        acc, mf1 = float((p == yd).mean()), float(f1_score(yd, p, average="macro"))
        gecmis.append({"epoch": ep, "dogrulama_kayip": dk, "dogrulama_dogruluk": acc, "dogrulama_makro_f1": mf1})
        if dk < en_iyi - 1e-4:
            en_iyi, en_iyi_ep, sabir = dk, ep, 0
            torch.save(model.state_dict(), kl / "model_en_iyi.pt")
            en_iyi_p = p.copy()
        else:
            sabir += 1
        yaz(f"Epoch {ep:2d} | doğrulama kayıp {dk:.4f} doğruluk {acc:.4f} makroF1 {mf1:.4f} | {time.time() - t0:.0f} sn")
        if sabir >= 8:
            yaz("Erken durdurma.")
            break
    p = en_iyi_p
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    pr, re, f1, _ = precision_recall_fscore_support(yd, p, labels=[0, 1, 2], zero_division=0)
    kb = {}
    d = dog.assign(y=yd, p=p)
    for (s, k), g in d.groupby(["sinif", "kaynak"]):
        kb[f"{s} | kaynak {kaynak_kisa(k)}"] = {"n": int(len(g)), "dogruluk": float((g.y == g.p).mean())}
    sonuc = {"model": "EfficientNet-B0 (ImageNet ön eğitimli, tüm ağ ince ayar)", "giris": BOYUT,
             "parametre_sayisi": n_param, "egitim_suresi_sn": round(time.time() - t0, 1),
             "calisan_epoch": len(gecmis), "en_iyi_epoch": en_iyi_ep,
             "dogruluk": float((p == yd).mean()), "makro_f1": float(f1_score(yd, p, average="macro")),
             "sinif_bazinda": {s: {"kesinlik": float(pr[i]), "duyarlilik": float(re[i]), "f1": float(f1[i])}
                               for i, s in enumerate(SINIFLAR)},
             "karisiklik_matrisi": confusion_matrix(yd, p, labels=[0, 1, 2]).tolist(),
             "kaynak_bazinda": kb, "gecmis": gecmis}
    (kl / "metrikler_ek.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    yaz(f"Bitti: doğruluk {sonuc['dogruluk']:.4f}, makro F1 {sonuc['makro_f1']:.4f}")


if __name__ == "__main__":
    main()
