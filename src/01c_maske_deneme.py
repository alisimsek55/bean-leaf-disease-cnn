# -*- coding: utf-8 -*-
"""01c - Maske yöntemini hızlı denemek için: verilen kaynaktaki bayraklı + rastgele görüntüleri
yeniden maskeler ve tek bir temas sayfası üretir (diske işlenmiş görüntü yazmaz).
Kullanım: python src/01c_maske_deneme.py mendeley_ykvcrjffzd [n]"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, ISLENMIS, VERI  # noqa: E402

on = importlib.import_module("01_on_isleme")
kaynak = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
rap = pd.read_csv(ISLENMIS / "on_isleme_raporu.csv").fillna({"bayraklar": ""})
d = rap[rap.kaynak == kaynak]
sec = pd.concat([d[d.bayraklar != ""].sample(min(n // 2, (d.bayraklar != "").sum()), random_state=1),
                 d[d.bayraklar == ""].sample(n // 2, random_state=2)])
fn = {"mendeley_ykvcrjffzd": on.maske_siyah_zemin, "mendeley_3km9d246z2": on.maske_beyaz_zemin}.get(kaynak, on.maske_kart_zemin)
K = 128
tuval = Image.new("RGB", (8 * K, int(np.ceil(len(sec) / 8)) * K))
for i, r in enumerate(sec.itertuples()):
    rgb = on.goruntu_oku(VERI / r.dosya)
    g, _ = on.kirp_ve_boyutlandir(rgb, fn(rgb))
    if g is None:  # yaprak bulunamadı (boş maske): kutu siyah kalır
        print(f"yaprak yok, atlandı: {r.dosya}")
        continue
    tuval.paste(Image.fromarray(g).resize((K, K)), ((i % 8) * K, (i // 8) * K))
tuval.save(CIKTI / "maske_kontrol" / f"deneme_{kaynak}.png")
print("ok")
