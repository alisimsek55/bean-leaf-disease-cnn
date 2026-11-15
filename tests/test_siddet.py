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
    assert sid.DUZEYLER == ["sağlıklı", "az hastalıklı", "çok hastalıklı"]


@pytest.mark.parametrize("oran,beklenen", [
    (0.0, "sağlıklı"), (0.0049, "sağlıklı"),
    (0.005, "az hastalıklı"), (0.1299, "az hastalıklı"),
    (0.13, "çok hastalıklı"), (0.9, "çok hastalıklı"),
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
    assert sid.duzey(oran, T1, T2) == "çok hastalıklı"


def test_kucuk_leke_az_hastalikli():
    rgb, m, yy, xx = yaprak()
    rgb[((yy - 128) ** 2 + (xx - 128) ** 2) < 15 ** 2] = KAHVE
    oran = sid.lezyon_orani(rgb, m)
    assert T1 <= oran < T2
    assert sid.duzey(oran, T1, T2) == "az hastalıklı"

def test_siddet_duzeyi_tek_goruntu():
    rgb, m, _, _ = yaprak()
    oran, d = sid.siddet_duzeyi(rgb, m)
    assert oran == 0.0 and d == "sağlıklı"
    rgb[(m > 0) & (np.mgrid[:256, :256][1] < 128)] = KAHVE
    assert sid.siddet_duzeyi(rgb, m)[1] == "çok hastalıklı"


def test_bos_ya_da_asinan_maske_belirsiz():
    sid = yukle("siddet")
    rgb = np.zeros((256, 256, 3), np.uint8)
    assert sid.siddet_duzeyi(rgb, np.zeros((256, 256), np.uint8)) == (0.0, sid.BELIRSIZ)
    kucuk = np.zeros((256, 256), np.uint8)
    kucuk[100:104, 100:104] = 255  # 5x5 aşındırmada tamamen yok olur
    assert sid.siddet_duzeyi(rgb, kucuk) == (0.0, "belirsiz")
    assert sid.BELIRSIZ not in sid.DUZEYLER  # DUZEYLER'e dokunulmaz
    assert sid.lezyon_orani(rgb, np.zeros((256, 256), np.uint8)) == 0.0  # doğrudan çağrı eski davranış


def test_yeterli_yaprakta_duzey_degismez():
    sid = yukle("siddet")
    rgb, m, _, _ = yaprak()
    assert sid.siddet_duzeyi(rgb, m) == (0.0, "sağlıklı")
