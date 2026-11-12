# -*- coding: utf-8 -*-
"""
01b - MASKE GÖRSEL DENETİMİ: ön işlenmiş görüntülerden numaralı temas sayfaları üretir.
Her kaynak x sınıf bileşiminden rastgele örnekler ve kalite bayrağı alan tüm görüntüler
sayfalara konur; sayfalar elle (gözle) incelenir. Kötü maskelenenler
data/islenmis_dislanan.csv dosyasına (sütunlar: dosya, neden) yazılır ve 01 tekrar
çalıştırıldığında 'dislandi' olarak işaretlenir.

Kullanım: python src/01b_maske_kontrol.py [rastgele|bayrakli]
Çıktı: ciktilar/maske_kontrol/*.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, ISLENMIS, TOHUM  # noqa: E402

KUTU = 128
SUTUN = 8


def sayfa(df, ad):
    satir = int(np.ceil(len(df) / SUTUN))
    tuval = Image.new("RGB", (SUTUN * KUTU, satir * (KUTU + 14)), (40, 40, 40))
    c = ImageDraw.Draw(tuval)
    for i, r in enumerate(df.itertuples()):
        g = Image.open(ISLENMIS / r.islenmis).convert("RGB").resize((KUTU, KUTU))
        x, y = (i % SUTUN) * KUTU, (i // SUTUN) * (KUTU + 14)
        tuval.paste(g, (x, y))
        c.text((x + 2, y + KUTU), f"{i}", fill=(255, 255, 0))
    klasor = CIKTI / "maske_kontrol"
    klasor.mkdir(parents=True, exist_ok=True)
    tuval.save(klasor / f"{ad}.png")
    df.reset_index(drop=True)[["dosya", "sinif", "kaynak", "bayraklar"]].to_csv(klasor / f"{ad}.csv")


def goruntulu_satirlar(rap):
    """Raporu (gösterilebilir, görüntüsüz) olarak ikiye ayırır. Görüntüsüz: `islenmis`/`maske` yolu
    boş/NaN olan satırlar (ör. `yaprak_yok`: yaprak bulunamadı, diske görüntü yazılmadı); bunlar temas
    sayfasına konamaz. Elle/renksiz_maske ile dışlananların görüntüsü diskte durduğu için gösterilir."""
    yol_var = pd.Series(True, index=rap.index)
    for sutun in ("islenmis", "maske"):
        yol_var &= rap[sutun].notna() & (rap[sutun].astype(str).str.strip() != "")
    return rap[yol_var], rap[~yol_var]


def main():
    mod = sys.argv[1] if len(sys.argv) > 1 else "rastgele"
    rap = pd.read_csv(ISLENMIS / "on_isleme_raporu.csv").fillna({"bayraklar": ""})
    rap, goruntusuz = goruntulu_satirlar(rap)
    if len(goruntusuz):
        print(f"Görüntüsü olmayan (yaprak_yok vb.) {len(goruntusuz)} satır atlandı:")
        for r in goruntusuz.itertuples():
            print(f"  {r.dosya}  [{r.bayraklar}]")
    if mod == "rastgele":
        for (s, k), d in rap.groupby(["sinif", "kaynak"]):
            sayfa(d.sample(min(len(d), 32), random_state=TOHUM), f"rastgele_{s}_{k}")
    else:
        b = rap[rap.bayraklar != ""]
        for i in range(0, len(b), 48):
            sayfa(b.iloc[i:i + 48], f"bayrakli_{i // 48:02d}")


if __name__ == "__main__":
    main()
