"""scripts/bolme_denetimi.py: capraz sizinti denetimi (sentetik + depodaki bolme CSV'leri)."""
from pathlib import Path

from test_gruplama import betik

den = betik("bolme_denetimi")
DATA = Path(__file__).resolve().parent.parent / "data"
A = "mendeley_ykvcrjffzd"


def r(dosya, grup, kaynak=A, sinif="adi_yaprak_yanikligi"):
    return {"dosya": dosya, "grup": grup, "kaynak": kaynak, "sinif": sinif}


def test_temiz_bolme_sizinti_yok():
    eg = [r("bdcp__IMG_20240105_235312_000.jpg", "g1"), r("bdcp__IMG_20240105_235313_000.jpg", "g2"),
          r("bhd__Good_Leaf_1.png", "g3", "mendeley_3km9d246z2", "saglikli")]
    dog = [r("bdcp__IMG_20240105_235412_000.jpg", "g4"), r("bhd__Good_Leaf_2.png", "g5", "mendeley_3km9d246z2")]
    rap = den.denetle(eg, dog)
    assert not den.sizinti_var(rap)
    assert rap["dogrulama_etkilenen"] == 0


def test_ayni_saniye_ve_pencere_sizintisi_bulunur():
    eg = [r("bdcp__IMG_20240105_235312_000.jpg", "g1"), r("bdcp__IMG_20240105_235320_000.jpg", "g2")]
    dog = [r("bdcp__IMG_20240105_235312_500.jpg", "g3"),   # ayni saniye
           r("bdcp__IMG_20240105_235322_000.jpg", "g4"),   # 2 sn sonra: pencereli seri
           r("bdcp__IMG_20240105_235400_000.jpg", "g5")]   # bagimsiz
    rap = den.denetle(eg, dog)
    assert den.sizinti_var(rap)
    assert rap["ortak_grup"] == 0 and rap["ortak_dosya"] == 0
    assert rap["ortak_saniye"] == 1
    assert rap["capraz_seri"] == 2
    assert rap["dogrulama_saniye_etkilenen"] == 1
    assert rap["dogrulama_seri_etkilenen"] == 2
    assert rap["kaynak_bazinda"][A]["dogrulama_seri_etkilenen"] == 2


def test_ortak_grup_ve_dosya_bulunur():
    eg = [r("x.png", "g1", "mendeley_3km9d246z2")]
    dog = [r("x.png", "g1", "mendeley_3km9d246z2")]
    rap = den.denetle(eg, dog)
    assert rap["ortak_grup"] == 1 and rap["ortak_dosya"] == 1 and rap["dogrulama_etkilenen"] == 1


def depodaki_rapor():
    return den.denetle(den.oku(DATA / "bolme_egitim.csv"), den.oku(DATA / "bolme_dogrulama.csv"))


def test_depodaki_bolmede_ortak_grup_ve_dosya_yok():
    # Siradan grup/dosya sizintisi seri denetiminden ayri raporlansin diye ayri bir test.
    rap = depodaki_rapor()
    assert rap["ortak_grup"] == 0, rap["ornek_ortak_grup"]
    assert rap["ortak_dosya"] == 0, rap["ornek_ortak_dosya"]


def test_depodaki_bolmede_seri_sizintisi_yok():
    # Bolme seri-farkinda gruplarla yeniden kuruldu; ilk bolmede 145 ortak cekim saniyesi vardi.
    rap = depodaki_rapor()
    assert rap["ortak_saniye"] == 0 and rap["capraz_seri"] == 0, (
        f"ortak cekim saniyesi={rap['ortak_saniye']}, iki tarafa bolunmus seri={rap['capraz_seri']}, "
        f"etkilenen dogrulama satiri={rap['dogrulama_etkilenen']}/{rap['dogrulama_satiri']}")
