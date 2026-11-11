import numpy as np
import pandas as pd

from conftest import yukle

bolme = yukle("02_veri_bolme")


def sentetik(seed=0):
    rng = np.random.default_rng(seed)
    satirlar, g = [], 0
    for sinif, kaynak, n_grup in [("cbb", "A", 150), ("hale", "B", 100),
                                  ("saglikli", "A", 120), ("saglikli", "B", 250)]:
        for _ in range(n_grup):
            for _ in range(int(rng.integers(1, 5))):  # grup basina 1-4 yakin kopya
                satirlar.append({"sinif": sinif, "kaynak": kaynak, "grup": f"g{g:05d}"})
            g += 1
    return pd.DataFrame(satirlar)


def test_hicbir_grup_iki_tarafta_degil():
    df = bolme.grup_bolme(sentetik())
    egitim = set(df.loc[df.bolum == "egitim", "grup"])
    dogrulama = set(df.loc[df.bolum == "dogrulama", "grup"])
    assert egitim and dogrulama
    assert not (egitim & dogrulama)


def test_oran_yaklasik_70_30():
    df = bolme.grup_bolme(sentetik())
    oran = (df.bolum == "dogrulama").mean()
    assert 0.27 < oran < 0.33
    # her (sinif, kaynak) katmani da hedefe yakin olmali
    for _, d in df.groupby(["sinif", "kaynak"]):
        assert 0.22 < (d.bolum == "dogrulama").mean() < 0.38


def test_ayni_tohum_ayni_bolme():
    a = bolme.grup_bolme(sentetik())["bolum"].tolist()
    b = bolme.grup_bolme(sentetik())["bolum"].tolist()
    assert a == b