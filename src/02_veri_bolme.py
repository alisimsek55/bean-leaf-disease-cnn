# -*- coding: utf-8 -*-
"""
02 - VERİ BÖLME: %70 eğitim / %30 doğrulama, yakın-kopya GRUBUNA göre ve (sınıf, kaynak)
katmanlarına göre katmanlı.

* Aynı `grup` (aynı yaprağın neredeyse aynı kareleri) asla iki tarafa birden düşmez;
  aksi halde doğrulama başarımı şişer (sızıntı).
* Katmanlama sınıf + kaynak bileşimine göre yapılır; böylece örneğin sağlıklı sınıfın hem A
  (siyah kumaş) hem B (beyaz zemin) alt kümeleri ve CBB'nin Embrapa (C) görüntüleri her iki
  tarafta orantılı yer alır ve sonuçlar kaynak bazında raporlanabilir.
* Yöntem: her katmanda gruplar rastgele (sabit tohum) karıştırılır; doğrulamadaki görüntü
  sayısı katmanın %30'una en yakın olacak şekilde gruplar sırayla doğrulamaya eklenir.

Dengeleme burada YAPILMAZ (veri atılmaz); sınıf dengesizliği eğitimde sınıf ağırlıklı kayıp
ve veri artırma ile ele alınır (gerekçe SONUCLAR.md'de).

Çıktılar: data/bolme_egitim.csv, data/bolme_dogrulama.csv, ciktilar/bolme_tablosu.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, ISLENMIS, SINIFLAR, VERI, TOHUM  # noqa: E402

DOGRULAMA_ORANI = 0.30


def grup_bolme(rap, dogrulama_orani=DOGRULAMA_ORANI, tohum=TOHUM):
    """`rap` içine 'bolum' sütunu (egitim|dogrulama) ekleyip döndürür; gruplar bölünmez."""
    rng = np.random.default_rng(tohum)
    dog_gruplar = set()
    for (s, k), d in rap.groupby(["sinif", "kaynak"]):
        boy = d.groupby("grup").size()
        gruplar = list(boy.index)
        rng.shuffle(gruplar)
        hedef = dogrulama_orani * len(d)
        n = 0
        for g in gruplar:
            # bu grubu eklemek hedefe olan uzaklığı azaltıyorsa ekle
            if abs(n + boy[g] - hedef) <= abs(n - hedef):
                dog_gruplar.add(g)
                n += boy[g]
    rap["bolum"] = np.where(rap["grup"].isin(dog_gruplar), "dogrulama", "egitim")
    return rap


def main():
    rap = pd.read_csv(ISLENMIS / "on_isleme_raporu.csv")
    rap = rap[~rap["dislandi"]].reset_index(drop=True)
    rap = grup_bolme(rap)
    # Sızıntı denetimi: hiçbir grup iki tarafta olmamalı
    kesisim = set(rap.loc[rap.bolum == "egitim", "grup"]) & set(rap.loc[rap.bolum == "dogrulama", "grup"])
    assert not kesisim, "Grup sızıntısı!"

    sut = ["islenmis", "maske", "dosya", "sinif", "kaynak", "grup"]
    rap[rap.bolum == "egitim"][sut].to_csv(VERI / "bolme_egitim.csv", index=False)
    rap[rap.bolum == "dogrulama"][sut].to_csv(VERI / "bolme_dogrulama.csv", index=False)

    # Sınıf x (eğitim, doğrulama, toplam) tablosu + kaynak kırılımı
    CIKTI.mkdir(exist_ok=True)
    tab = pd.crosstab(rap["sinif"], rap["bolum"]).reindex(SINIFLAR)[["egitim", "dogrulama"]]
    tab["toplam"] = tab.sum(axis=1)
    tab["dogrulama_orani"] = (tab["dogrulama"] / tab["toplam"]).round(3)
    tab.loc["TOPLAM"] = [tab["egitim"].sum(), tab["dogrulama"].sum(), tab["toplam"].sum(),
                         round(tab["dogrulama"].sum() / tab["toplam"].sum(), 3)]
    tab.to_csv(CIKTI / "bolme_tablosu.csv")
    kay = pd.crosstab([rap["sinif"], rap["kaynak"]], rap["bolum"])
    kay.to_csv(CIKTI / "bolme_tablosu_kaynak.csv")
    grup_say = rap.groupby(["sinif", "bolum"])["grup"].nunique().unstack()
    ozet = {
        "sinif_tablosu": tab.reset_index().to_dict(orient="records"),
        "sinif_kaynak_tablosu": [{"sinif": a, "kaynak": b, **{c: int(v) for c, v in r.items()}}
                                 for (a, b), r in kay.iterrows()],
        "grup_sayilari": {s: {c: int(v) for c, v in r.items()} for s, r in grup_say.iterrows()},
        "dislanan_goruntu": int(pd.read_csv(ISLENMIS / "on_isleme_raporu.csv")["dislandi"].sum()),
        "grup_sizintisi": len(kesisim),
    }
    (CIKTI / "bolme_ozeti.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tab.to_string())
    print(kay.to_string())
    print(grup_say.to_string())


if __name__ == "__main__":
    main()
