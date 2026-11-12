"""veri.py: egitim ve degerlendirme yollari maske disini ayni sekilde siyaha boyamali."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
pd = pytest.importorskip("pandas")
from PIL import Image  # noqa: E402

import veri  # noqa: E402


@pytest.fixture
def sentetik(tmp_path, monkeypatch):
    """Siyah zemin uzerinde dairesel, dokulu bir 'yaprak' (boyut 128'in kati degil -> INTER_AREA tasar)."""
    rng = np.random.default_rng(0)
    n = 333
    m = np.zeros((n, n), np.uint8)
    cv2.circle(m, (170, 160), 120, 255, -1)
    g = rng.integers(60, 256, (n, n, 3), dtype=np.uint8) * (m > 0)[..., None]
    Image.fromarray(g.astype(np.uint8)).save(tmp_path / "g.png")
    Image.fromarray(m).save(tmp_path / "m.png")
    monkeypatch.setattr(veri, "ISLENMIS", tmp_path)
    df = pd.DataFrame({"islenmis": ["g.png"], "maske": ["m.png"]})
    return veri.goruntuleri_yukle(df, boyut=128)


def test_ham_tensorde_maske_disi_tasma_var(sentetik):
    # Testin anlamli olmasi icin: boyutlandirilmis RGB, maske disinda siyah olmayan piksel icerir
    rgb, m = sentetik[0, :3], sentetik[0, 3]
    assert int(((rgb.amax(0) > 0) & (m == 0)).sum()) > 0


def test_degerlendirme_maske_disi_tam_siyah(sentetik):
    ds = veri.YaprakVeri(sentetik, [0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], egitim=False)
    x, _ = ds[0]
    disari = sentetik[0, 3] <= 127
    assert x.shape == (3, 128, 128)
    assert float(x[:, disari].abs().max()) == 0.0
    # maske ici degismemeli
    assert torch.equal(x[:, ~disari], sentetik[0, :3].float()[:, ~disari] / 255.0)


def test_egitim_ve_degerlendirme_ayni_maskeleme(sentetik):
    ort, std = [0.4, 0.5, 0.3], [0.2, 0.25, 0.3]
    de = veri.YaprakVeri(sentetik, [1], ort, std, egitim=True, renk_artirma=False)
    de.geo = torch.nn.Identity()  # geometrik artirma kapali: yalniz maskeleme kurali karsilastirilir
    dd = veri.YaprakVeri(sentetik, [1], ort, std, egitim=False)
    (xe, ye), (xd, yd) = de[0], dd[0]
    assert torch.equal(xe, xd) and int(ye) == int(yd) == 1


def test_maske_uygula():
    rgb = torch.full((3, 2, 2), 200, dtype=torch.uint8)
    m = torch.tensor([[[255, 0], [128, 127]]], dtype=torch.uint8)
    out = veri.maske_uygula(rgb, m)
    assert out[:, 0, 0].tolist() == [200] * 3 and out[:, 1, 0].tolist() == [200] * 3
    assert out[:, 0, 1].tolist() == [0] * 3 and out[:, 1, 1].tolist() == [0] * 3
