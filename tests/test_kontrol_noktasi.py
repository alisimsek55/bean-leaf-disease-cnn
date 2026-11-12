"""kontrol_noktasi.EnIyiTakip: en iyi = kesin minimum; 1e-4 tolerans yalniz erken durdurma sabrinda."""
import numpy as np

from kontrol_noktasi import EnIyiTakip


def _eski_kural(kayiplar, tol=1e-4):
    """Duzeltme oncesi kural (sabir referansi icin): yalniz > tol iyilesme sabri sifirlar."""
    en_iyi, sabirlar = float("inf"), []
    sabir = 0
    for k in kayiplar:
        if k < en_iyi - tol:
            en_iyi, sabir = k, 0
        else:
            sabir += 1
        sabirlar.append(sabir)
    return sabirlar


def test_kucuk_iyilesme_en_iyiyi_gunceller_sabri_sifirlamaz():
    # epoch 3 son anlamli iyilesme; epoch 5 ondan yalniz 5e-5 daha dusuk (gercek minimum)
    kayiplar = [1.0, 0.8, 0.5, 0.52, 0.49995, 0.51, 0.53]
    t = EnIyiTakip(tolerans=1e-4)
    kaydet = [t.guncelle(k, ep) for ep, k in enumerate(kayiplar, start=1)]
    assert t.en_iyi_epoch == int(np.argmin(kayiplar)) + 1 == 5
    assert t.en_iyi == min(kayiplar)
    assert kaydet == [True, True, True, False, True, False, False]


def test_sabir_davranisi_degismedi():
    kayiplar = [1.0, 0.8, 0.5, 0.52, 0.49995, 0.49991, 0.51, 0.3, 0.29995, 0.4]
    t = EnIyiTakip(tolerans=1e-4)
    sabirlar = []
    for ep, k in enumerate(kayiplar, start=1):
        t.guncelle(k, ep)
        sabirlar.append(t.sabir)
    assert sabirlar == _eski_kural(kayiplar) == [0, 0, 0, 1, 2, 3, 4, 0, 1, 2]
    assert t.en_iyi_epoch == int(np.argmin(kayiplar)) + 1 == 9


def test_esit_kayip_kaydetmez():
    t = EnIyiTakip()
    assert t.guncelle(0.5, 1) is True
    assert t.guncelle(0.5, 2) is False
    assert t.en_iyi_epoch == 1
