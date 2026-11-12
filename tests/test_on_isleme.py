import cv2
import numpy as np

from conftest import yukle

on = yukle("01_on_isleme")


def beyaz_zeminde_yesil_elips(boy=400):
    rgb = np.full((boy, boy, 3), 255, np.uint8)
    cv2.ellipse(rgb, (boy // 2, boy // 2), (120, 70), 20, 0, 360, (40, 150, 40), -1)
    return rgb


def test_maske_zemini_disarida_birakir():
    rgb = beyaz_zeminde_yesil_elips()
    m = on.maske_beyaz_zemin(rgb)
    assert m[5, 5] == 0 and m[-5, -5] == 0           # kose = zemin
    assert m[200, 200] == 1                           # merkez = yaprak
    elips = (rgb[..., 0] < 100)
    kesisim = ((m > 0) & elips).sum()
    assert kesisim / elips.sum() > 0.97              # yaprak buyuk olcude korunur
    assert ((m > 0) & ~elips).sum() / elips.sum() < 0.05   # zemin sizmaz


def test_zemin_siyaha_donusur_ve_kirpilir():
    rgb = beyaz_zeminde_yesil_elips()
    m = on.maske_beyaz_zemin(rgb)
    g, gm = on.kirp_ve_boyutlandir(rgb, m)
    assert g.shape == (on.KAYDET_BOYUT, on.KAYDET_BOYUT, 3)
    assert gm.shape == (on.KAYDET_BOYUT, on.KAYDET_BOYUT)
    assert g[gm == 0].max() == 0                      # maske disi tamamen siyah
    assert not (g == 255).all(axis=-1).any()         # beyaz zemin kalmadi
    assert g[on.KAYDET_BOYUT // 2, on.KAYDET_BOYUT // 2].tolist() == [40, 150, 40]

def test_tamamen_beyaz_goruntu_cokmez():
    rgb = np.full((300, 300, 3), 255, np.uint8)
    for f in (on.maske_beyaz_zemin, on.maske_kart_zemin):  # beyaz = bu yontemlerde zemin
        m = f(rgb)
        assert m.shape == (300, 300) and m.sum() == 0
    assert on.maske_siyah_zemin(rgb).shape == (300, 300)  # parlak = yaprak sayilir ama cokmez


def test_tamamen_siyah_goruntu_cokmez():
    rgb = np.zeros((300, 300, 3), np.uint8)
    for f in (on.maske_siyah_zemin, on.maske_kart_zemin):  # siyah = bu yontemlerde zemin
        m = f(rgb)
        assert m.shape == (300, 300) and m.sum() == 0
    assert on.maske_beyaz_zemin(rgb).shape == (300, 300)


def test_bos_maske_kirpmada_yaprak_yok_sonucu_verir():
    rgb = np.full((100, 120, 3), 7, np.uint8)
    g, gm = on.kirp_ve_boyutlandir(rgb, np.zeros((100, 120), np.uint8))
    assert g is None and gm is None
