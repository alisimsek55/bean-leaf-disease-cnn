"""scripts/indir.py testleri: ag erisimi yok; getir() sahte fonksiyonla degistirilir."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

try:
    import requests  # noqa: F401
except ImportError:  # ortamda requests yoksa sahte modul (testler ag kullanmaz)
    import sys
    import types
    _r = types.ModuleType("requests")
    _r.Session = lambda: types.SimpleNamespace(headers=types.SimpleNamespace(update=lambda d: None))
    sys.modules["requests"] = _r

YOL = Path(__file__).resolve().parent.parent / "scripts" / "indir.py"
spec = importlib.util.spec_from_file_location("indir_betigi", YOL)
indir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indir)

ICERIK = b"dogru icerik"
OZET = hashlib.sha256(ICERIK).hexdigest()


@pytest.mark.parametrize("ad", ["bean_leaves.zip", "bean_leaves (1).zip", "a..b.zip", "x.jpg"])
def test_guvenli_ad_normal_isimleri_kabul_eder(ad):
    assert indir.guvenli_ad(ad) == ad


@pytest.mark.parametrize("ad", [
    "", "   ", None, "..", ".", "../x", r"..\..\hedef", "a/b", r"a\b", "/etc/passwd",
    r"C:\Windows\x", "C:x", r"\\sunucu\paylas", "x<y", "a?b", "a\x00b", "ad.", "ad ",
])
def test_guvenli_ad_tehlikeli_isimleri_reddeder(ad):
    with pytest.raises(ValueError):
        indir.guvenli_ad(ad)


def test_guvenli_yol_klasor_icinde_kalir(tmp_path):
    y = indir.guvenli_yol(str(tmp_path), "bean_leaves.zip")
    assert Path(y).parent == tmp_path


def test_guvenli_yol_disari_cikamaz(tmp_path):
    with pytest.raises(ValueError):
        indir.guvenli_yol(str(tmp_path), r"..\..\hedef")


def sahte_getir(icerik, sayac):
    def f(url, gecici):
        sayac.append(url)
        Path(gecici).write_bytes(icerik)
    return f


def test_yanlis_ozetli_mevcut_dosya_yeniden_indirilir(tmp_path, monkeypatch):
    hedef = str(tmp_path / "d.zip")
    Path(hedef).write_bytes(b"x" * len(ICERIK))  # ayni boyut, yanlis icerik
    cagrilar = []
    monkeypatch.setattr(indir, "getir", sahte_getir(ICERIK, cagrilar))
    indir.indir("http://u", hedef, len(ICERIK), OZET)
    assert len(cagrilar) == 1
    assert Path(hedef).read_bytes() == ICERIK


def test_dogru_mevcut_dosya_tekrar_indirilmez(tmp_path, monkeypatch):
    hedef = str(tmp_path / "d.zip")
    Path(hedef).write_bytes(ICERIK)
    cagrilar = []
    monkeypatch.setattr(indir, "getir", sahte_getir(b"baska", cagrilar))
    indir.indir("http://u", hedef, len(ICERIK), OZET)
    assert cagrilar == []
    assert Path(hedef).read_bytes() == ICERIK


def test_yanlis_ozet_indirilen_dosya_silinir_ve_hata_verir(tmp_path, monkeypatch):
    hedef = str(tmp_path / "d.zip")
    monkeypatch.setattr(indir, "getir", sahte_getir(b"x" * len(ICERIK), []))
    with pytest.raises(indir.IndirmeHatasi):
        indir.indir("http://u", hedef, len(ICERIK), OZET)
    assert not Path(hedef).exists() and not Path(hedef + ".part").exists()


def test_denemeler_tukenince_hata_firlatir(tmp_path, monkeypatch):
    hedef = str(tmp_path / "d.zip")
    sayac = []

    def hatali(url, gecici):
        sayac.append(1)
        raise ConnectionError("ag yok")

    monkeypatch.setattr(indir, "getir", hatali)
    with pytest.raises(indir.IndirmeHatasi):
        indir.indir_tekrarli("http://u", hedef, len(ICERIK), OZET, deneme=3, bekle=0)
    assert len(sayac) == 3
    assert not Path(hedef).exists()


def test_tekrar_sonrasi_basarili_olursa_hata_yok(tmp_path, monkeypatch):
    hedef = str(tmp_path / "d.zip")
    durum = {"n": 0}

    def once_hatali(url, gecici):
        durum["n"] += 1
        if durum["n"] < 2:
            raise ConnectionError("gecici")
        Path(gecici).write_bytes(ICERIK)

    monkeypatch.setattr(indir, "getir", once_hatali)
    indir.indir_tekrarli("http://u", hedef, len(ICERIK), OZET, deneme=3, bekle=0)
    assert Path(hedef).read_bytes() == ICERIK


def test_km9_kotu_dosya_adi_disariya_yazmaz(tmp_path, monkeypatch):
    meta = {"files": [{"filename": "bean_leaves/../../kacak", "size": 1,
                       "content_details": {"download_url": "http://u"}}]}
    monkeypatch.setattr(indir, "mendeley_meta", lambda kimlik, d: meta)
    monkeypatch.setattr(indir, "getir", sahte_getir(b"x", []))
    with pytest.raises(ValueError):
        indir.km9(str(tmp_path))
    assert not list(tmp_path.rglob("kacak*"))


def test_ykvcrjffzd_basarisiz_indirme_sifirdan_farkli_cikis(tmp_path, monkeypatch):
    meta = {"files": [{"filename": "a.jpg", "folder_id": 2, "size": 3,
                       "content_details": {"download_url": "http://u", "sha256_hash": "0" * 64}}]}
    klasorler = [{"id": 1, "name": "Bean"}, {"id": 2, "name": "Blight", "parent_id": 1}]

    class Yanit:
        def json(self):
            return klasorler

    monkeypatch.setattr(indir, "mendeley_meta", lambda kimlik, d: meta)
    monkeypatch.setattr(indir.S, "get", lambda *a, **k: Yanit(), raising=False)
    monkeypatch.setattr(indir, "getir", sahte_getir(b"abc", []))
    monkeypatch.setattr(indir.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit) as e:
        indir.ykvcrjffzd(str(tmp_path))
    assert e.value.code not in (0, None)
