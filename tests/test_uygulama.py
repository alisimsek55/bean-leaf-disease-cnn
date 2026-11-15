"""uygulama/model_adaptoru: ornek goruntulerde gecerli sinif ve siddet duzeyi dondurur.

Model dosyasi (ciktilar/*.pt, .gitignore'da) ya da ham veri yoksa (CI) atlanir."""
import sys
from pathlib import Path

import numpy as np
import pytest

KOK = Path(__file__).resolve().parent.parent
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))

from uygulama import model_adaptoru as ma  # noqa: E402

HAM = KOK / "data" / "raw"
SINIFLAR = ["adi_yaprak_yanikligi", "hale_yanikligi", "saglikli"]


def _ornekler():
    out = []
    for s in SINIFLAR:
        d = HAM / s
        if d.exists():
            out += sorted(d.glob("*"))[:1]
    return out


def test_duzey_adlari_ve_esikler():
    assert ma.duzey_adi(0.0) == "sağlıklı"
    assert ma.duzey_adi(0.004) == "sağlıklı"
    assert ma.duzey_adi(0.005) == "az hastalıklı"
    assert ma.duzey_adi(0.129) == "az hastalıklı"
    assert ma.duzey_adi(0.13) == "çok hastalıklı"


def test_olasilik_eslemesi():
    ol = ma._olasiliklari_duzenle({"olasiliklar": {"CBB": 2, "Hale": 1, "Sağlıklı": 1}})
    assert set(ol) == set(SINIFLAR) and abs(sum(ol.values()) - 1) < 1e-9


def test_hastalik_olasiligi():
    ol = {"adi_yaprak_yanikligi": 0.3, "hale_yanikligi": 0.5, "saglikli": 0.2}
    assert abs(ma.hastalik_olasiligi(ol) - 0.8) < 1e-9
    assert ma.hastalik_olasiligi({"saglikli": 1.0}) == 0.0


def test_maske_basarisizsa_tum_goruntu():
    from uygulama import yaprak_ayirma as ya
    duz = np.full((200, 300, 3), 128, np.uint8)  # yapraksiz duz gri
    m, yontem, uyari = ya.yaprak_ayir(duz)
    assert m.shape == duz.shape[:2] and m.sum() > 0


@pytest.mark.skipif(not ma.model_var(), reason="model dosyasi yok (ciktilar/genel_model/model.pt)")
@pytest.mark.skipif(not _ornekler(), reason="ham veri yok (data/raw)")
@pytest.mark.parametrize("yol", _ornekler(), ids=lambda p: p.parent.name)
def test_analiz_gecerli_cikti(yol):
    r = ma.analiz_et(yol)
    assert r.sinif in SINIFLAR
    assert set(r.olasiliklar) == set(SINIFLAR)
    assert abs(sum(r.olasiliklar.values()) - 1) < 1e-4
    assert r.duzey in ma.DUZEYLER
    assert 0.0 <= r.lezyon_orani <= 1.0
    assert r.yaprak.shape == (256, 256, 3) and r.lezyon_katmani.shape == r.yaprak.shape
    if ma.GENEL_MODEL.exists():
        assert r.model_adi == ma.GENEL_AD
