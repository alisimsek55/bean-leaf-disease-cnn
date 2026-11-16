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


# ---------------------------------------------------------------- güvenli model yükleme
GENEL_VAR = ma.GENEL_MODEL.exists() and ma._tahmin is not None


@pytest.mark.parametrize("yol", [ma.GENEL_MODEL, ma.ESKI_MODEL], ids=["genel", "eski"])
def test_model_dosyalari_weights_only_ile_yuklenir(yol):
    torch = pytest.importorskip("torch")
    if not yol.exists():
        pytest.skip("model dosyasi yok")
    ck = torch.load(yol, map_location="cpu", weights_only=True)
    assert "model" in ck and "ortalama" in ck and "std" in ck
    if yol == ma.GENEL_MODEL:
        assert ma._tahmin.kontrol_noktasi_dogrula(ck)["siniflar"] == SINIFLAR


def test_kontrol_noktasi_dogrulama_ve_guvensiz_dosya_reddi(tmp_path):
    torch = pytest.importorskip("torch")
    if ma._tahmin is None:
        pytest.skip("tahmin modulu yuklenemedi")
    t = ma._tahmin
    iyi = {"model": {}, "omurga": "x", "boyut": 224, "ortalama": [0, 0, 0], "std": [1, 1, 1], "siniflar": SINIFLAR}
    t.kontrol_noktasi_dogrula(dict(iyi))
    with pytest.raises(t.ModelDosyasiHatasi, match="eksik"):
        t.kontrol_noktasi_dogrula({k: v for k, v in iyi.items() if k != "boyut"})
    with pytest.raises(t.ModelDosyasiHatasi, match="sınıf"):
        t.kontrol_noktasi_dogrula(dict(iyi, siniflar=list(reversed(SINIFLAR))))
    # pickle içinde keyfî nesne: weights_only=True bunu açmamalı (kod çalıştırılmaz)
    yol = tmp_path / "kotu.pt"
    torch.save({"model": {}, "kotu": _Kotu()}, yol)
    with pytest.raises(t.ModelDosyasiHatasi, match="güvenli"):
        t.kontrol_noktasi_oku(yol)


class _Kotu:
    def __reduce__(self):
        return (print, ("bu calismamali",))


# ---------------------------------------------------------------- yükleme sınırları
def _png(w, h):
    import io
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (w, h), (90, 160, 60)).save(b, "PNG")
    b.seek(0)
    return b


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_buyuk_ve_asiri_oranli_goruntu_reddedilir(tmp_path):
    with pytest.raises(ma.GoruntuHatasi, match="en-boy"):
        ma.goruntu_ac(_png(2000, 100))
    with pytest.raises(ma.GoruntuHatasi, match="küçük"):
        ma.goruntu_ac(_png(10, 10))
    # 8000x6000 = 48 MP: başlıktan reddedilir (piksel çözülmeden)
    from PIL import Image
    yol = tmp_path / "buyuk.png"
    Image.new("L", (8000, 6000)).save(yol, "PNG")
    with pytest.raises(ma.GoruntuHatasi, match="MP"):
        ma.goruntu_ac(yol)
    # bayt sınırı
    import io
    with pytest.raises(ma.GoruntuHatasi, match="MB"):
        ma.goruntu_ac(io.BytesIO(b"\0" * (ma.MAKS_BAYT + 1)))
    with pytest.raises(ma.GoruntuHatasi, match="okunamadı"):
        ma.goruntu_ac(io.BytesIO(b"resim degil"))
    assert ma.goruntu_ac(_png(400, 300)).size == (400, 300)


def test_streamlit_yukleme_siniri_ayari():
    toml = (KOK / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "maxUploadSize = 15" in toml and ma.MAKS_BAYT == 15 * 1024 * 1024


def test_kucult_asiri_oranda_cokmez():
    from uygulama import yaprak_ayirma as ya
    k = ya.kucult(np.zeros((3, 5000, 3), np.uint8))
    assert k.shape[1] == 1024 and k.shape[0] >= 1


# ---------------------------------------------------------------- tek hat + şiddet
def _sahte_model(monkeypatch, kayit):
    def sahte(g, m, bilgi):
        kayit.append((g.copy(), m.copy()))
        return {"olasiliklar": {"adi_yaprak_yanikligi": 0.2, "hale_yanikligi": 0.1, "saglikli": 0.7}}
    monkeypatch.setattr(ma, "_genel_kullanilabilir", lambda: True)
    monkeypatch.setattr(ma, "_genel_tahmin", sahte)


@pytest.mark.skipif(not GENEL_VAR, reason="genel model dosyasi yok")
def test_ayirma_basarisizsa_siddet_hesaplanamadi(monkeypatch):
    from PIL import Image
    kayit = []
    _sahte_model(monkeypatch, kayit)
    # yapraksız gri doku: ayırma başarısız -> tüm görüntü lezyon sayılmamalı
    rng = np.random.default_rng(0)
    gri = rng.integers(110, 140, (300, 400, 1), dtype=np.uint8).repeat(3, axis=2)
    r = ma.analiz_et(Image.fromarray(gri))
    assert r.ayirma_basarili is False
    assert r.duzey == ma.HESAPLANAMADI and r.lezyon_orani is None and not r.siddet_var
    assert any("ayrılamadı" in u for u in r.uyarilar)


@pytest.mark.skipif(not GENEL_VAR, reason="genel model dosyasi yok")
def test_ayirma_basarisizsa_koyu_zemin_lezyon_sayilmaz(monkeypatch):
    """Maskeleme algoritmasından bağımsız: on_isle başarısız derse (tüm koyu fotoğraf) şiddet ölçülmez."""
    from PIL import Image
    _sahte_model(monkeypatch, [])
    koyu = np.full((300, 400, 3), 35, np.uint8)
    monkeypatch.setattr(ma, "on_isle", lambda rgb: (np.full((256, 256, 3), 35, np.uint8),
                                                     np.full((256, 256), 255, np.uint8),
                                                     {"zemin": "genel", "basarili": False, "alan_orani": 1.0}))
    r = ma.analiz_et(Image.fromarray(koyu))
    assert r.duzey == ma.HESAPLANAMADI and r.lezyon_orani is None
    assert np.array_equal(r.lezyon_katmani, r.yaprak)  # lezyon işaretlenmedi


def _yaprak_goruntusu():
    import cv2
    g = np.full((400, 400, 3), 245, np.uint8)  # beyaz zemin
    cv2.ellipse(g, (200, 200), (140, 90), 30, 0, 360, (60, 150, 50), -1)
    cv2.circle(g, (230, 190), 18, (120, 70, 30), -1)  # kahverengi leke
    return g


@pytest.mark.skipif(not GENEL_VAR, reason="genel model dosyasi yok")
def test_tek_hat_ayni_goruntu_ve_maske(monkeypatch):
    from PIL import Image
    kayit = []
    _sahte_model(monkeypatch, kayit)
    rgb = _yaprak_goruntusu()
    r = ma.analiz_et(Image.fromarray(rgb))
    g, m, bilgi = ma.genel_maske.on_isle(rgb, "otomatik")
    assert r.ayirma_basarili and bilgi["basarili"]
    assert len(kayit) == 1
    # modele giden = ekranda gösterilen = şiddeti ölçülen
    assert np.array_equal(kayit[0][0], r.yaprak) and np.array_equal(kayit[0][1], r.maske)
    assert np.array_equal(r.yaprak, g) and np.array_equal(r.maske, m)
    oran, duzey, _ = ma.siddet_hesapla(r.yaprak, r.maske)
    assert r.lezyon_orani == oran and r.duzey == duzey and r.duzey in ma.DUZEYLER


def test_bos_maske_siddet_belirsiz():
    oran, duzey, lez = ma.siddet_hesapla(np.zeros((256, 256, 3), np.uint8), np.zeros((256, 256), np.uint8))
    assert oran is None and duzey == ma.BELIRSIZ and lez.sum() == 0


@pytest.mark.skipif(not GENEL_VAR, reason="genel model dosyasi yok")
def test_islenmisten_tahmin_tahmin_et_ile_ayni():
    from PIL import Image
    rgb = _yaprak_goruntusu()
    g, m, bilgi = ma.genel_maske.on_isle(rgb, "otomatik")
    a = ma._tahmin.tahmin_et(Image.fromarray(rgb), model_yolu=ma.GENEL_MODEL)
    b = ma._tahmin.islenmisten_tahmin(g, m, bilgi, model_yolu=ma.GENEL_MODEL)
    assert a["olasiliklar"] == b["olasiliklar"]


# ---------------------------------------------------------------- önbellek
@pytest.mark.skipif(not GENEL_VAR, reason="genel model dosyasi yok")
def test_model_ikinci_cagrida_yeniden_yuklenmez(monkeypatch):
    t = ma._tahmin
    t._onbellek.clear()
    sayac = {"n": 0}
    asil = t.kontrol_noktasi_oku

    def sayan(yol):
        sayac["n"] += 1
        return asil(yol)
    monkeypatch.setattr(t, "kontrol_noktasi_oku", sayan)
    ma.model_yukle()
    ma.model_yukle()
    ma.analiz_et(_png(300, 300))
    assert ma.model_adi() == ma.GENEL_AD and ma.model_var()
    assert sayac["n"] == 1


# ---------------------------------------------------------------- ikinci kontrol düzeltmeleri
def _gecerli_genel_ck():
    return {"model": {}, "omurga": "x", "boyut": 224, "ortalama": [0.4, 0.4, 0.4], "std": [0.2, 0.2, 0.2],
            "siniflar": list(SINIFLAR)}


def test_genel_ck_siniflar_zorunlu_ve_normalizasyon_sonlu():
    if ma._tahmin is None:
        pytest.skip("tahmin modulu yuklenemedi")
    t = ma._tahmin
    t.kontrol_noktasi_dogrula(_gecerli_genel_ck())
    ck = _gecerli_genel_ck(); del ck["siniflar"]
    with pytest.raises(t.ModelDosyasiHatasi, match="siniflar"):
        t.kontrol_noktasi_dogrula(ck)
    with pytest.raises(t.ModelDosyasiHatasi, match="sınıf"):
        t.kontrol_noktasi_dogrula(dict(_gecerli_genel_ck(), siniflar=SINIFLAR[:2]))
    for ort, std in (([float("nan"), 0.4, 0.4], [0.2] * 3), ([0.4] * 3, [0.2, float("inf"), 0.2]),
                     ([0.4] * 3, [0.2, 0.0, 0.2]), ([0.4] * 3, [0.2, -0.1, 0.2]), (["a", "b", "c"], [0.2] * 3)):
        with pytest.raises(t.ModelDosyasiHatasi):
            t.kontrol_noktasi_dogrula(dict(_gecerli_genel_ck(), ortalama=ort, std=std))


def _gecerli_eski_ck():
    return {"model": {}, "ortalama": [0.2, 0.3, 0.1], "std": [0.3, 0.3, 0.2], "gorev": "hastalik",
            "girdi": "renkli", "sinif_sayisi": 3}


def test_eski_ck_gorev_girdi_ve_normalizasyon_dogrulanir():
    ma.eski_kontrol_noktasi_dogrula(_gecerli_eski_ck())
    for bozuk in ({"gorev": "kaynak"}, {"girdi": "siluet"}, {"sinif_sayisi": 2},
                  {"std": [0.3, 0.0, 0.2]}, {"ortalama": [float("nan"), 0.3, 0.1]}):
        with pytest.raises(ValueError):
            ma.eski_kontrol_noktasi_dogrula(dict(_gecerli_eski_ck(), **bozuk))
    for alan in ("gorev", "girdi"):
        ck = _gecerli_eski_ck(); del ck[alan]
        with pytest.raises(ValueError, match="eksik"):
            ma.eski_kontrol_noktasi_dogrula(ck)


def test_mevcut_eski_model_dogrulamadan_gecer():
    torch = pytest.importorskip("torch")
    if not ma.ESKI_MODEL.exists():
        pytest.skip("eski model dosyasi yok")
    ma.eski_kontrol_noktasi_dogrula(torch.load(ma.ESKI_MODEL, map_location="cpu", weights_only=True))


@pytest.mark.skipif(ma._tahmin is None or not ma.ESKI_MODEL.exists(), reason="eski model ya da tahmin yok")
def test_bozuk_genel_model_eski_modele_duser(tmp_path, monkeypatch):
    bozuk = tmp_path / "model.pt"
    bozuk.write_bytes(b"bu bir model degil")
    monkeypatch.setattr(ma, "GENEL_MODEL", bozuk)
    assert ma.model_yukle() == ma.ESKI_AD
    assert ma.genel_model_hatasi() and ma.model_adi() == ma.ESKI_AD
    r = ma.analiz_et(_png(300, 300))
    assert r.model_adi == ma.ESKI_AD
    assert any("Genel model yüklenemedi" in u for u in r.uyarilar)


@pytest.mark.skipif(ma._tahmin is None, reason="tahmin modulu yuklenemedi")
def test_iki_model_de_yoksa_hata(tmp_path, monkeypatch):
    bozuk = tmp_path / "model.pt"
    bozuk.write_bytes(b"bozuk")
    monkeypatch.setattr(ma, "GENEL_MODEL", bozuk)
    monkeypatch.setattr(ma, "ESKI_MODEL", tmp_path / "yok.pt")
    with pytest.raises(FileNotFoundError, match="genel model"):
        ma.model_yukle()


def test_app_onbellek_anahtari_hashlenir():
    """Streamlit `_` ile başlayan parametreleri hash'lemez: anahtar parametresi alt çizgisiz olmalı
    ve model dosyalarının mtime'ını içermeli."""
    import ast
    agac = ast.parse((KOK / "uygulama" / "app.py").read_text(encoding="utf-8"))
    fonk = {f.name: f for f in ast.walk(agac) if isinstance(f, ast.FunctionDef)}
    h = fonk["_hazirla"]
    assert any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "cache_resource" for d in h.decorator_list)
    assert h.args.args and all(not a.arg.startswith("_") for a in h.args.args)
    # _model_anahtari'yi app'i çalıştırmadan değerlendir ve mtime değişimine duyarlı olduğunu doğrula
    ad_alani = {"ma": ma}
    exec(compile(ast.Module(body=[fonk["_model_anahtari"]], type_ignores=[]), "app.py", "exec"), ad_alani)
    a1 = ad_alani["_model_anahtari"]()
    assert [y for y, _ in a1] == [str(ma.GENEL_MODEL), str(ma.ESKI_MODEL)]


def test_model_anahtari_mtime_degisince_degisir(tmp_path, monkeypatch):
    import ast
    import os
    agac = ast.parse((KOK / "uygulama" / "app.py").read_text(encoding="utf-8"))
    f = next(f for f in ast.walk(agac) if isinstance(f, ast.FunctionDef) and f.name == "_model_anahtari")
    ad_alani = {"ma": ma}
    exec(compile(ast.Module(body=[f], type_ignores=[]), "app.py", "exec"), ad_alani)
    g = tmp_path / "g.pt"; g.write_bytes(b"1")
    monkeypatch.setattr(ma, "GENEL_MODEL", g)
    monkeypatch.setattr(ma, "ESKI_MODEL", tmp_path / "yok.pt")
    a1 = ad_alani["_model_anahtari"]()
    st = g.stat()
    os.utime(g, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    assert ad_alani["_model_anahtari"]() != a1


class _Durdur(Exception):
    pass


@pytest.mark.parametrize("yon", [6, 8])
def test_pil_ve_dosya_ayni_on_isle_girdisi(yon, tmp_path, monkeypatch):
    """EXIF yönlü JPEG: PIL ve dosya yolu `on_isle`'ye aynı (yönü düzeltilmiş) görüntüyü vermeli. Model gerekmez."""
    from PIL import Image
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)
    im = Image.fromarray(a)
    ex = Image.Exif()
    ex[0x0112] = yon
    yol = tmp_path / "exif.jpg"
    im.save(yol, exif=ex)
    yakalanan = []

    def sahte(rgb):
        yakalanan.append(np.array(rgb))
        raise _Durdur

    monkeypatch.setattr(ma, "on_isle", sahte)
    with pytest.raises(_Durdur):
        ma.analiz_et(yol)
    with pytest.raises(_Durdur):
        ma.analiz_et(Image.open(yol))
    assert yakalanan[0].shape == (200, 120, 3)  # yön uygulandı (genişlik/yükseklik yer değiştirdi)
    assert yakalanan[0].shape == yakalanan[1].shape
    assert np.array_equal(yakalanan[0], yakalanan[1])
