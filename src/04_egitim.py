# -*- coding: utf-8 -*-
"""
04 - EĞİTİM: FasulyeCNN'i sıfırdan eğitir.

Ayarlar (gerekçeler SONUCLAR.md'de):
  * Kayıp: sınıf ağırlıklı çapraz entropi, w_c = N / (K * n_c)  (dengesizlik için)
  * Optimizer: AdamW, öğrenme oranı 1e-3, ağırlık azaltma 1e-4, batch 32
  * Öğrenme oranı planı: ReduceLROnPlateau (doğrulama kaybı 5 epoch iyileşmezse x0.5)
  * En fazla 100 epoch; erken durdurma: doğrulama kaybı 15 epoch iyileşmezse dur
  * En iyi model = en düşük doğrulama kaybı
  * Veri artırma: yatay/dikey çevirme, 0-360° döndürme, rastgele kırp-boyutlandır (%75-100),
    hafif parlaklık/kontrast/doygunluk/ton değişimi; artırma sonrası arka plan tekrar siyaha boyanır.

Kullanım:
  python src/04_egitim.py                                  # ana model (renkli, 3 sınıf)
  python src/04_egitim.py --girdi siluet --ad siluet       # kontrol: yalnız yaprak siluetiyle
  python src/04_egitim.py --gorev kaynak --ad kaynak        # kontrol: sağlıklılarda kaynak (A/B) tahmini
  python src/04_egitim.py --haric_kaynak embrapa_pddb --ad embrapa_haric   # kontrol: Embrapa (C) kaynak dışı test
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, TOHUM, tohum_ayarla  # noqa: E402
from veri import YaprakVeri, bolme_oku, etiketler, goruntuleri_yukle, kanal_istatistigi  # noqa: E402

model_mod = importlib.import_module("03_model")


def calistir(args):
    tohum_ayarla(args.tohum)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    klasor = CIKTI / ("" if args.ad == "ana" else f"kontrol_{args.ad}")
    klasor.mkdir(parents=True, exist_ok=True)
    log = open(klasor / "egitim_logu.txt", "w", encoding="utf-8")

    def yaz(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    egt, dog = bolme_oku("egitim"), bolme_oku("dogrulama")
    if args.gorev == "kaynak":
        # Kontrol deneyi: yalnız sağlıklı sınıfın A ve B kaynaklı görüntüleri; hedef = kaynak
        egt = egt[(egt.sinif == "saglikli") & egt.kaynak.isin(["mendeley_ykvcrjffzd", "mendeley_3km9d246z2"])]
        dog = dog[(dog.sinif == "saglikli") & dog.kaynak.isin(["mendeley_ykvcrjffzd", "mendeley_3km9d246z2"])]
    if args.haric_kaynak:
        # Kontrol deneyi: bir kaynağı eğitimden ve doğrulamadan tamamen çıkar (kaynak dışı test için)
        egt = egt[egt.kaynak != args.haric_kaynak]
        dog = dog[dog.kaynak != args.haric_kaynak]
    egt, dog = egt.reset_index(drop=True), dog.reset_index(drop=True)
    sinif_sayisi = 2 if args.gorev == "kaynak" else 3
    yaz(f"Cihaz: {cihaz} ({torch.cuda.get_device_name(0) if cihaz.type == 'cuda' else 'CPU'})")
    yaz(f"Görev: {args.gorev}, girdi: {args.girdi}, eğitim {len(egt)}, doğrulama {len(dog)}")

    Xe, Xd = goruntuleri_yukle(egt, args.girdi), goruntuleri_yukle(dog, args.girdi)
    ye, yd = etiketler(egt, args.gorev), etiketler(dog, args.gorev)
    ort, std = kanal_istatistigi(Xe)
    std = [max(s, 1e-3) for s in std]
    renk = args.girdi == "renkli"
    de = YaprakVeri(Xe, ye, ort, std, egitim=True, renk_artirma=renk)
    dd = YaprakVeri(Xd, yd, ort, std, egitim=False)
    g = torch.Generator().manual_seed(args.tohum)
    yukleyici_e = DataLoader(de, batch_size=args.batch, shuffle=True, generator=g, num_workers=0, drop_last=True)
    yukleyici_d = DataLoader(dd, batch_size=128, shuffle=False, num_workers=0)

    sayilar = np.bincount(ye, minlength=sinif_sayisi)
    agirlik = torch.tensor(len(ye) / (sinif_sayisi * sayilar), dtype=torch.float32)
    yaz(f"Eğitim sınıf sayıları: {sayilar.tolist()}  -> kayıp ağırlıkları: {[round(a, 3) for a in agirlik.tolist()]}")

    model = model_mod.FasulyeCNN(sinif_sayisi=sinif_sayisi).to(cihaz)
    n_param = model_mod.parametre_sayisi(model)
    yaz(f"Parametre sayısı: {n_param:,}")
    kayip_f = nn.CrossEntropyLoss(weight=agirlik.to(cihaz))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    gecmis = []
    en_iyi, en_iyi_epoch, sabir = float("inf"), 0, 0
    t0 = time.time()
    for ep in range(1, args.epoch + 1):
        model.train()
        tk, td, tn = 0.0, 0, 0
        for x, y in yukleyici_e:
            x, y = x.to(cihaz), y.to(cihaz)
            opt.zero_grad()
            c = model(x)
            k = kayip_f(c, y)
            k.backward()
            opt.step()
            tk += k.item() * len(y)
            td += (c.argmax(1) == y).sum().item()
            tn += len(y)
        model.eval()
        dk, tahmin = 0.0, []
        with torch.no_grad():
            for x, y in yukleyici_d:
                x, y = x.to(cihaz), y.to(cihaz)
                c = model(x)
                dk += kayip_f(c, y).item() * len(y)
                tahmin.append(c.argmax(1).cpu())
        tahmin = torch.cat(tahmin).numpy()
        dk /= len(yd)
        dacc = float((tahmin == yd).mean())
        dF1 = float(f1_score(yd, tahmin, average="macro"))
        plan.step(dk)
        lr = opt.param_groups[0]["lr"]
        gecmis.append({"epoch": ep, "egitim_kayip": tk / tn, "egitim_dogruluk": td / tn,
                       "dogrulama_kayip": dk, "dogrulama_dogruluk": dacc, "dogrulama_makro_f1": dF1, "lr": lr})
        iyi = dk < en_iyi - 1e-4
        if iyi:
            en_iyi, en_iyi_epoch, sabir = dk, ep, 0
            torch.save({"model": model.state_dict(), "ortalama": ort, "std": std, "epoch": ep,
                        "girdi": args.girdi, "gorev": args.gorev, "sinif_sayisi": sinif_sayisi},
                       klasor / "model_en_iyi.pt")
        else:
            sabir += 1
        yaz(f"Epoch {ep:3d} | eğitim kayıp {tk / tn:.4f} doğ. {td / tn:.4f} | doğrulama kayıp {dk:.4f} "
            f"doğ. {dacc:.4f} makroF1 {dF1:.4f} | lr {lr:.2e} | {time.time() - t0:.0f} sn{'  *' if iyi else ''}")
        if sabir >= args.sabir:
            yaz(f"Erken durdurma: {args.sabir} epoch boyunca doğrulama kaybı iyileşmedi.")
            break
    sure = time.time() - t0
    ozet = {"ad": args.ad, "gorev": args.gorev, "girdi": args.girdi, "tohum": args.tohum, "haric_kaynak": args.haric_kaynak,
            "egitim_suresi_sn": round(sure, 1), "calisan_epoch": len(gecmis), "en_iyi_epoch": en_iyi_epoch,
            "en_iyi_dogrulama_kayip": en_iyi, "parametre_sayisi": n_param,
            "egitim_ornek": len(ye), "dogrulama_ornek": len(yd), "egitim_sinif_sayilari": sayilar.tolist(),
            "kayip_agirliklari": agirlik.tolist(), "normalizasyon": {"ortalama": ort, "std": std},
            "ayarlar": {"optimizer": "AdamW", "lr": args.lr, "weight_decay": 1e-4, "batch": args.batch,
                        "maks_epoch": args.epoch, "erken_durdurma_sabir": args.sabir,
                        "lr_plani": "ReduceLROnPlateau(factor=0.5, patience=5)"},
            "cihaz": str(cihaz), "gecmis": gecmis}
    (klasor / "egitim_ozeti.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    yaz(f"Bitti: {sure:.0f} sn, en iyi epoch {en_iyi_epoch}, en iyi doğrulama kaybı {en_iyi:.4f}")
    log.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--girdi", choices=["renkli", "siluet"], default="renkli")
    p.add_argument("--gorev", choices=["hastalik", "kaynak"], default="hastalik")
    p.add_argument("--ad", default="ana")
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--sabir", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tohum", type=int, default=TOHUM)
    p.add_argument("--haric_kaynak", default="", help="ör. embrapa_pddb: bu kaynak hiç kullanılmaz")
    calistir(p.parse_args())
