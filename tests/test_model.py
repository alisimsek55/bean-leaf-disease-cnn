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