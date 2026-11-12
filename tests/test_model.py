import pytest

torch = pytest.importorskip("torch")

from conftest import yukle  # noqa: E402

model_mod = yukle("03_model")


def test_cikti_sekli():
    m = model_mod.FasulyeCNN().eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 128, 128))
    assert tuple(y.shape) == (2, 3)


def test_parametre_sayisi_belgedeki_sayiya_esit():
    # docs/RESULTS.tr.md ve README: 1 772 995 egitilebilir parametre
    assert model_mod.parametre_sayisi(model_mod.FasulyeCNN()) == 1_772_995


def test_son_evrisim_haritasi_8x8x256():
    m = model_mod.FasulyeCNN().eval()
    with torch.no_grad():
        a = m.son_evrisim(m.ozellik(torch.zeros(1, 3, 128, 128)))
    assert tuple(a.shape) == (1, 256, 8, 8)

# --- 05_degerlendirme: dogru/yanlis ornek secimi (zayif modelde cokmemeli) ---
import numpy as np  # noqa: E402

degerlendirme = yukle("05_degerlendirme")


def test_dogru_yanlis_sec_normal_durum():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    p = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 2, 0])
    dogru, yanlis = degerlendirme.dogru_yanlis_sec(y, p, np.random.default_rng(0))
    assert len(dogru) == 6 and len(set(dogru)) == 6
    assert all(y[i] == p[i] for i in dogru)
    assert [int(y[i]) for i in dogru] == [0, 0, 1, 1, 2, 2]
    assert sorted(int(i) for i in yanlis) == [9, 10, 11]


def test_dogru_yanlis_sec_zayif_model():
    # sinif 0: hic dogru yok, sinif 1: tek dogru, sinif 2: iki dogru
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([1, 2, 1, 0, 2, 2])
    dogru, yanlis = degerlendirme.dogru_yanlis_sec(y, p, np.random.default_rng(0))
    assert sorted(int(i) for i in dogru) == [2, 4, 5]
    assert sorted(int(i) for i in yanlis) == [0, 1, 3]


def test_dogru_yanlis_sec_hic_yanlis_yok():
    y = np.array([0, 1, 2])
    dogru, yanlis = degerlendirme.dogru_yanlis_sec(y, y.copy(), np.random.default_rng(0))
    assert sorted(int(i) for i in dogru) == [0, 1, 2] and yanlis == []
