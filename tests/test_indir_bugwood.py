"""scripts/indir_bugwood.py testleri: ag erisimi yok; getir() sahte fonksiyonla degistirilir."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

YOL = Path(__file__).resolve().parent.parent / "scripts" / "indir_bugwood.py"
spec = importlib.util.spec_from_file_location("indir_bugwood", YOL)
ib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ib)

def _gercek_jpeg(boyut=(16, 16)):
    b = io.BytesIO()
    Image.new("RGB", boyut, (40, 160, 40)).save(b, "JPEG")
    return b.getvalue()


JPEG = _gercek_jpeg()
SAHTE = b"\xff\xd8\xff\xe0" + b"x" * 100  # yalniz imza dogru, icerik bozuk


def test_jpeg_mi_gercek_kesik_ve_sahte():
    assert ib.jpeg_mi(JPEG)
    assert not ib.jpeg_mi(SAHTE)
    assert not ib.jpeg_mi(JPEG[:len(JPEG) // 2])  # kesik
    assert not ib.jpeg_mi(b"")


def test_jpeg_mi_bayt_ve_piksel_siniri(monkeypatch):
    monkeypatch.setattr(ib, "MAKS_BOYUT", len(JPEG) - 1)
    assert not ib.jpeg_mi(JPEG)
    monkeypatch.undo()
    monkeypatch.setattr(ib, "MAKS_PIKSEL", 10)  # 16x16 > 10
    assert not ib.jpeg_mi(JPEG)


def test_mevcut_bozuk_dosya_yeniden_indirilir(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "NUMARALAR", ["1111111", "2222222"])
    raw = tmp_path / "data" / "raw_bugwood"
    raw.mkdir(parents=True)
    (raw / "1111111.jpg").write_bytes(SAHTE)  # bozuk mevcut -> yeniden indir
    (raw / "2222222.jpg").write_bytes(JPEG)  # saglam mevcut -> indirilmez
    resimler = []

    def sahte(url):
        if "api.bugwood" in url:
            return 200, b"{}"
        resimler.append(url)
        return 200, JPEG

    s = ib.main(str(tmp_path), getir_f=sahte, bekle=0)
    assert [x["durum"] for x in s] == ["ok", "ok"]
    assert len(resimler) == 1 and "1111111" in resimler[0]
    assert (raw / "1111111.jpg").read_bytes() == JPEG


def test_indirilen_bozuk_jpeg_ok_sayilmaz_ve_yazilmaz(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "NUMARALAR", ["1111111"])

    def sahte(url):
        return (200, b"{}") if "api.bugwood" in url else (200, SAHTE)

    s = ib.main(str(tmp_path), getir_f=sahte, bekle=0)
    assert s[0]["durum"] == "hata_200"
    assert not (tmp_path / "data" / "raw_bugwood" / "1111111.jpg").exists()


def test_numara_listesi_77_benzersiz_ve_gecerli():
    assert len(ib.NUMARALAR) == 77 == len(set(ib.NUMARALAR))
    for n in ib.NUMARALAR:
        assert ib.gecerli_numara(n) == n


@pytest.mark.parametrize("kotu", ["../123456", "12345a7", "", "1" * 12])
def test_gecersiz_numara_reddedilir(kotu):
    with pytest.raises(ValueError):
        ib.gecerli_numara(kotu)


@pytest.mark.parametrize("metin,beklenen", [
    ("Symptoms on dry bean leaves", "fasulye (dry bean, P. vulgaris)"),
    ("Lesions on cowpea", "cowpea (Vigna)"),
    ("Snap bean pods", "fasulye (snap bean, P. vulgaris)"),
    ("Leaf of Phaseolus vulgaris", "fasulye (P. vulgaris)"),
    ("Bacterial culture on agar", "belirtilmemis"),
])
def test_konak_cikar(metin, beklenen):
    assert ib.konak_cikar(metin) == beklenen


def test_meta_satiri_ve_atif():
    api = {"rows": [{"photographer": "Ad Soyad", "organization": "Kurum X",
                     "description": "<p>Leaf of <em>dry bean</em> &amp; lesions</p>",
                     "scientificName": "Xanthomonas axonopodis pv. phaseoli", "sub_name": "Common  bacterial blight"}]}
    s = ib.meta_satiri("5359904", api)
    assert s["atif"] == "Ad Soyad, Kurum X, Bugwood.org"
    assert s["aciklama"] == "Leaf of dry bean & lesions"
    assert s["konak"].startswith("fasulye")
    assert s["url"].endswith("/1536x1024/5359904.jpg")
    assert ib.meta_satiri("5359904", {})["atif"] == "Bugwood.org"


def test_main_aglsiz_403_zorlanmaz(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "NUMARALAR", ["1111111", "2222222"])
    istekler = []

    def sahte(url):
        istekler.append(url)
        if "api.bugwood" in url:
            return 200, json.dumps({"rows": [{"photographer": "P", "organization": "O"}]}).encode()
        if "1111111" in url:
            return 200, JPEG
        return 403, b""

    s = ib.main(str(tmp_path), getir_f=sahte, bekle=0)
    assert [x["durum"] for x in s] == ["ok", "hata_403"]
    assert (tmp_path / "data" / "raw_bugwood" / "1111111.jpg").read_bytes() == JPEG
    assert not (tmp_path / "data" / "raw_bugwood" / "2222222.jpg").exists()
    assert sum("bugwoodcloud" in u for u in istekler) == 2  # 403 icin tekrar deneme yok
    with open(tmp_path / "data" / "bugwood_metadata.csv", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    assert r[0]["sha256"] == hashlib.sha256(JPEG).hexdigest() and r[1]["sha256"] == ""
