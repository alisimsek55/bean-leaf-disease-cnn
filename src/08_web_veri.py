# -*- coding: utf-8 -*-
"""
08 - EK KAYNAK (D = PlantWild web görüntüleri): kopya ayıklama + ön işleme.

Girdi : data/raw_web/ (scripts/indir_web.py), etiketler_web.csv
Adımlar:
  1. 256-bit dHash (scripts/yakin_kopya.py ile aynı) — web içi yakın kopyalar aynı `grup`a alınır.
  2. Web görüntülerinden A/B/C (data/raw) görüntülerinden herhangi birine Hamming <= 20 olanlar
     ATILIR (özellikle Embrapa = C kopyası eğitime sızmasın diye; sayısı raporlanır).
  3. Zemin bilinmediği için `genel_maske.on_isle(otomatik)` ile maskeleme/kırpma; maske alanı
     %2'nin altında kalan (maskeleme başarısız) görüntüler dışlanır.
Çıktı: data/islenmis_web/{goruntu,maske}/<sinif>/*.png ve data/islenmis_web/web_veri.csv
       (bolme_*.csv ile aynı sütunlar; `islenmis`/`maske` yolları data/islenmis/'e göreli).
Kullanım: python src/08_web_veri.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ortak import HAM, KOK, VERI  # noqa: E402
import genel_maske  # noqa: E402

_spec = importlib.util.spec_from_file_location("yakin_kopya", KOK / "scripts" / "yakin_kopya.py")
yk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yk)

WEB = VERI / "raw_web"
CIKIS = VERI / "islenmis_web"
ESIK = 20


def main():
    et = pd.read_csv(WEB / "etiketler_web.csv")
    # birebir kopyalar (sha256) -> ilkini tut
    once = len(et)
    et = et.drop_duplicates("sha256").reset_index(drop=True)
    birebir = once - len(et)
    hw = [yk.dhash(VERI / d) for d in et.dosya]
    ham = pd.read_csv(VERI / "etiketler.csv")
    ham = ham[ham.kume == "raw"].reset_index(drop=True)
    hh = [yk.dhash(VERI / d) for d in ham.dosya]
    # web <-> A/B/C yakın kopya
    yakin_abc = {}
    for i, a in enumerate(hw):
        for j, b in enumerate(hh):
            d = (a ^ b).bit_count()
            if d <= ESIK and (i not in yakin_abc or d < yakin_abc[i][1]):
                yakin_abc[i] = (ham.kaynak[j], d, ham.dosya[j])
    gruplar = yk.grup_etiketleri(len(hw), [(i, j) for i, j, _ in yk.dhash_baglantilari(hw, ESIK)])
    et["grup"] = ["w" + g for g in gruplar]
    kayit, dislanan = [], []
    for i, r in et.iterrows():
        if i in yakin_abc:
            dislanan.append({"dosya": r.dosya, "neden": f"yakin_kopya_{yakin_abc[i][0]}", "esli": yakin_abc[i][2]})
            continue
        rgb = np.asarray(Image.open(VERI / r.dosya).convert("RGB"))
        g, m, bilgi = genel_maske.on_isle(rgb, "otomatik")
        if not bilgi["basarili"]:
            dislanan.append({"dosya": r.dosya, "neden": "maske_basarisiz", "esli": ""})
            continue
        ad = Path(r.dosya).stem + ".png"
        for alt, dizi in (("goruntu", g), ("maske", m)):
            (CIKIS / alt / r.sinif).mkdir(parents=True, exist_ok=True)
            Image.fromarray(dizi).save(CIKIS / alt / r.sinif / ad)
        kayit.append({"islenmis": f"../islenmis_web/goruntu/{r.sinif}/{ad}",
                      "maske": f"../islenmis_web/maske/{r.sinif}/{ad}", "dosya": r.dosya, "sinif": r.sinif,
                      "kaynak": "plantwild_web", "grup": r.grup, "zemin": bilgi["zemin"],
                      "alan_orani": bilgi["alan_orani"]})
    df = pd.DataFrame(kayit)
    df.to_csv(CIKIS / "web_veri.csv", index=False)
    pd.DataFrame(dislanan).to_csv(CIKIS / "web_dislanan.csv", index=False)
    ozet = {"indirilen": once, "birebir_kopya": birebir,
            "abc_yakin_kopya": {k: int(sum(1 for v in yakin_abc.values() if v[0] == k))
                                for k in sorted({v[0] for v in yakin_abc.values()})},
            "maske_basarisiz": int(sum(d["neden"] == "maske_basarisiz" for d in dislanan)),
            "kalan": {s: int(n) for s, n in df.sinif.value_counts().items()},
            "grup_sayisi": {s: int(n) for s, n in df.groupby("sinif").grup.nunique().items()},
            "zemin": {s: int(n) for s, n in df.zemin.value_counts().items()}}
    (CIKIS / "web_ozeti.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ozet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
