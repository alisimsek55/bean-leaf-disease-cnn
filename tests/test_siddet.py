import numpy as np
import pytest

from conftest import yukle

sid = yukle("06_siddet_tahmini")

# docs/RESULTS.tr.md Bolum 7: T1 = %0,5, T2 = %13
T1, T2 = 0.005, 0.13


def yaprak(boy=256):
    """Siyah zeminde disk seklinde yaprak maskesi ve duz yesil RGB."""
    yy, xx = np.mgrid[:boy, :boy]
    maske = (((yy - boy // 2) ** 2 + (xx - boy // 2) ** 2) < 100 ** 2).astype(np.uint8) * 255
    rgb = np.zeros((boy, boy, 3), np.uint8)
    rgb[maske > 0] = (40, 160, 40)
    return rgb, maske, yy, xx


KAHVE = (120, 70, 30)


def test_renk_esikleri_belgedekiyle_ayni():
    assert sid.A_KIRMIZI == 4
    assert sid.H_SARI == 27
    assert sid.S_RENKLI == 60
    assert sid.V_KOYU == 45
    assert sid.DUZEYLER == ["sağlıklı", "az hastalıklı", "hastalıklı"]


@pytest.mark.parametrize("oran,beklenen", [
    (0.0, "sağlıklı"), (0.0049, "sağlıklı"),
    (0.005, "az hastalıklı"), (0.1299, "az hastalıklı"),
    (0.13, "hastalıklı"), (0.9, "hastalıklı"),
])
def test_duzey_esikleri(oran, beklenen):
    assert sid.duzey(oran, T1, T2) == beklenen


def test_tam_yesil_saglikli():
    rgb, m, _, _ = yaprak()
    oran = sid.lezyon_orani(rgb, m)
    assert oran == 0.0
    assert sid.duzey(oran, T1, T2) == "sağlıklı"


def test_yarisi_kahverengi_hastalikli():
    rgb, m, yy, xx = yaprak()
    rgb[(m > 0) & (xx < 128)] = KAHVE
    oran = sid.lezyon_orani(rgb, m)
    assert 0.35 < oran < 0.65
    assert sid.duzey(oran, T1, T2) == "hastalıklı"


def test_kucuk_leke_az_hastalikli():
    rgb, m, yy, xx = yaprak()
    rgb[((yy - 128) ** 2 + (xx - 128) ** 2) < 15 ** 2] = KAHVE
    oran = sid.lezyon_orani(rgb, m)
    assert T1 <= oran < T2
    assert sid.duzey(oran, T1, T2) == "az hastalıklı"