"""src/12_selcuk_veri.py testleri: ağ yok, gerçek veri yok; sahte küçük görüntülerle."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import yukle

sv = yukle("12_selcuk_veri")


def _goruntu(tohum, boyut=(120, 60)):
    """Yapılı (dHash'i kararlı) rastgele görüntü: düşük frekanslı desen + biraz gürültü."""
    rng = np.random.default_rng(tohum)
    kaba = rng.integers(0, 256, (6, 12, 3), dtype=np.uint8)
    im = Image.fromarray(kaba).resize(boyut, Image.BICUBIC)
    a = np.asarray(im).astype(int) + rng.integers(-3, 4, (boyut[1], boyut[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _sahte_kaynak(kok, kopya=True):
    """4 klasör x 3 görüntü; Ustun42_Healthy (1) = Akbulut_Healthy (1) birebir kopya."""
    t = 0
    for klasor in sv.KLASORLER:
        (kok / klasor).mkdir(parents=True)
        for n in (1, 2, 3):
            t += 1
            _goruntu(t).save(kok / klasor / f"{klasor} ({n}).jpg", quality=95)
    if kopya:
        (kok / "Ustun42_Healthy" / "Ustun42_Healthy (1).jpg").write_bytes(
            (kok / "Akbulut_Healthy" / "Akbulut_Healthy (1).jpg").read_bytes())


def test_metadata_ve_kopyalama(tmp_path):
    kaynak, hedef = tmp_path / "kaynak", tmp_path / "raw_selcuk"
    _sahte_kaynak(kaynak)
    once = {p: p.read_bytes() for p in kaynak.rglob("*.jpg")}
    df = sv.metadata_olustur(kaynak, hedef)
    assert len(df) == 12 and set(df.columns) >= {"dosya", "orijinal", "cesit", "sinif", "sha256", "genislik", "yukseklik", "bayt"}
    assert set(df.cesit) == {"akbulut", "ustun42"} and set(df.sinif) == {"adi_yaprak_yanikligi", "saglikli"}
    assert (df.groupby(["cesit", "sinif"]).size() == 3).all()
    assert "raw_selcuk/ustun42_saglikli_0001.jpg" in set(df.dosya)
    for r in df.itertuples():  # kopya içerik olarak aynı, SHA-256 doğru
        assert sv.sha256_dosya(hedef / r.dosya.split("/", 1)[1]) == r.sha256
        assert (r.genislik, r.yukseklik) == (120, 60)
    assert df.sha256.duplicated().sum() == 1  # Ustun42_Healthy (1) = Akbulut_Healthy (1)
    assert {p: p.read_bytes() for p in kaynak.rglob("*.jpg")} == once  # kaynak klasör değişmedi
    df2 = sv.metadata_olustur(kaynak, hedef)  # ikinci çalıştırma aynı sonucu verir
    pd.testing.assert_frame_equal(df, df2)


def test_dosya_adlari():
    assert sv.orijinal_no("Akbulut_Healthy (12).jpg") == 12
    assert sv.hedef_ad("ustun42", "adi_yaprak_yanikligi", 7) == "ustun42_cbb_0007.jpg"
    with pytest.raises(ValueError):
        sv.orijinal_no("bozuk.jpg")


def test_dhash_bitleri_yakin_kopya_ile_ayni_tanim(tmp_path):
    im = _goruntu(5)
    yol = tmp_path / "a.png"
    im.save(yol)
    assert sv.bitler_int(sv.dhash_bitleri(im)) == sv.yk.dhash(yol)


def test_8_yon_dhash_dondurulmus_ve_aynalanmis_kopyalari_bulur():
    asil = [_goruntu(t) for t in range(10, 16)]
    turev = [asil[0].transpose(Image.Transpose.ROTATE_90),
             asil[1].transpose(Image.Transpose.FLIP_LEFT_RIGHT),
             asil[2].transpose(Image.Transpose.ROTATE_180).resize((240, 120)),  # döndür + büyüt
             asil[3].crop((2, 1, 118, 59))]  # hafif kırpma
    H8 = np.stack([sv.dhash8(im) for im in asil + turev])
    ciftler, D = sv.ic_yakin_ciftler(H8, sv.ESIK)
    bag = {(i, j) for i, j, _ in ciftler}
    assert bag == {(0, 6), (1, 7), (2, 8), (3, 9)}
    # yönsüz (yalnız özdeşlik) dHash döndürülmüş kopyayı kaçırır -> 8 yön gerekli
    assert np.count_nonzero(H8[0, 0] != H8[6, 0]) > sv.ESIK
    g = sv.gruplar_kur(len(H8), ciftler)
    assert g[0] == g[6] and g[3] == g[9] and len(set(g)) == 6
    # dış kaynak karşılaştırması: F'nin 8 yönüne karşı dışarının özdeşlik hash'i
    Dx = sv.min8_uzaklik(H8[6:7], H8[:6, 0])
    assert Dx.argmin() == 0 and Dx.min() <= sv.ESIK


def test_kopya_ayikla_kurallari():
    df = pd.DataFrame({"dosya": list("abcd"), "sinif": ["s", "s", "s", "c"], "cesit": ["x", "x", "y", "y"]})
    at, bag = sv.kopya_ayikla(df, [(0, 1, 3), (0, 2, 0), (2, 3, 5)])
    assert bag == [(0, 1)]  # aynı klasör -> aynı grup
    assert at[2][0] in ("yakin_kopya_baska_cesit_klasoru", "sinif_celiskili_yakin_kopya")
    assert at[3][0] == "sinif_celiskili_yakin_kopya" and 0 not in at and 1 not in at


def test_seri_baglantilari_blok_ve_benzerlik():
    klasor = ["k"] * 25 + ["m"] * 2
    no = list(range(1, 26)) + [10, 11]
    benzer = {(9, 10)}  # 10 ile 11 (indeks 9-10) blok sınırında ama benzer
    c = sv.seri_baglantilari(klasor, no, lambda i, j: 0.99 if (i, j) in benzer else 0.1, blok=10, esik=0.85)
    g = sv.gruplar_kur(len(no), c)
    assert len({g[i] for i in range(0, 10)}) == 1  # 1-10 aynı seri
    assert g[9] == g[10]  # sınırda benzer -> birleşir (1-20 tek grup)
    assert g[19] != g[20]  # 20 | 21 sınırı, benzer değil -> ayrı
    assert g[25] != g[26] and g[25] not in g[:25]  # başka klasör bağlanmaz; m:10 ve m:11 farklı blok, benzer değil


def test_gruplu_bol_ayrik_katmanli():
    rng = np.random.default_rng(0)
    satir = []
    for s in ("adi_yaprak_yanikligi", "saglikli"):
        for c in ("akbulut", "ustun42"):
            for gidx in range(30):
                for _ in range(int(rng.integers(1, 12))):
                    satir.append({"sinif": s, "cesit": c, "grup": f"{s[:3]}{c}{gidx}"})
    df = pd.DataFrame(satir)
    e, d = sv.gruplu_bol(df, 0.3, tohum=42)
    assert not set(e.grup) & set(d.grup)  # aynı grup iki tarafta yok
    assert len(e) + len(d) == len(df)
    for (s, c), g in df.groupby(["sinif", "cesit"]):  # her katmanda ~%30 doğrulama
        n_d = ((d.sinif == s) & (d.cesit == c)).sum()
        assert 0.25 <= n_d / len(g) <= 0.45
    e2, d2 = sv.gruplu_bol(df, 0.3, tohum=42)  # deterministik
    assert list(d2.grup) == list(d.grup)
    bozuk = df.copy()
    bozuk.loc[0, "grup"] = bozuk.loc[len(bozuk) - 1, "grup"]  # grup iki katmana yayılıyor
    with pytest.raises(ValueError):
        sv.gruplu_bol(bozuk, 0.3)


def test_sha_kopyalari_once_sinif_celiskisi():
    meta = pd.DataFrame({"dosya": [f"raw_selcuk/x_{i}.jpg" for i in range(6)],
                         "sinif": ["saglikli", "saglikli", "adi_yaprak_yanikligi", "saglikli", "saglikli", "saglikli"],
                         "sha256": ["a", "b", "b", "c", "c", "d"]})
    df, dis = sv.sha_ayikla(meta)
    # b: iki farklı sınıf -> ikisi de atılır; c: aynı sınıf -> ilki tutulur
    assert list(df.dosya) == ["raw_selcuk/x_0.jpg", "raw_selcuk/x_3.jpg", "raw_selcuk/x_5.jpg"]
    nedenler = {d["dosya"]: d["neden"] for d in dis}
    assert nedenler == {"raw_selcuk/x_1.jpg": "birebir_kopya_sinif_celiskili",
                        "raw_selcuk/x_2.jpg": "birebir_kopya_sinif_celiskili", "raw_selcuk/x_4.jpg": "birebir_kopya"}


def test_zaman_baglantilari():
    klasor = ["k", "k", "k", "k", "m"]
    zaman = ["2023-09-17 11:29:00", "2023-09-17 11:29:07", "2023-09-17 11:29:30", None, "2023-09-17 11:29:31"]
    assert sv.zaman_baglantilari(klasor, zaman, pencere=10) == [(0, 1)]
    assert sorted(sv.zaman_baglantilari(klasor, zaman, pencere=30)) == [(0, 1), (1, 2)]  # başka klasör bağlanmaz


def _exifli_jpeg(yol, yon=6, boyut=(1200, 600)):
    im = _goruntu(77, boyut)
    ex = Image.Exif()
    ex[274] = yon
    ex[306] = "2023:09:17 11:29:00"
    im.save(yol, quality=95, exif=ex)
    return im


def test_gomme_girdisi_exif_yonunu_uygular(tmp_path):
    """Regresyon: seri gömmesi EXIF yönü uygulanmış görüntüden hesaplanmalı (dHash/ön işleme ile aynı)."""
    im = _exifli_jpeg(tmp_path / "exif.jpg")  # yön 6: görüntü 90° döndürülerek gösterilir
    duz = ImageOps_exif(Image.open(tmp_path / "exif.jpg"))
    duz.save(tmp_path / "duz.png")
    a, b = sv.gomme_girdisi(tmp_path / "exif.jpg"), sv.gomme_girdisi(tmp_path / "duz.png")
    assert a.shape == (448, 224, 3)  # dikey (EXIF sonrası 600x1200)
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 3
    assert sv.gomme_girdisi(tmp_path / "duz.png").shape != np.array(im.resize((448, 224))).shape
    with Image.open(tmp_path / "exif.jpg") as j:
        assert sv.cekim_zamani(j) == "2023-09-17 11:29:00"


def ImageOps_exif(im):
    from PIL import ImageOps
    return ImageOps.exif_transpose(im)


def test_gomme_agirligi_yoksa_yan_etkiden_once_anlasilir_hata(tmp_path):
    with pytest.raises(FileNotFoundError):
        sv.gomme_agirlik_yolu(tmp_path / "yok.pth")


def test_egitim_ve_tahmin_ayni_on_isleme(tmp_path, monkeypatch):
    """Aynı ham EXIF'li JPEG: veri hazırlama (12._isle) ve tahmin_et aynı 256x256 görüntü + maskeyi üretir."""
    pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    yol = tmp_path / "ham.jpg"
    a = np.full((600, 1200, 3), (150, 120, 90), np.uint8)
    cv2.ellipse(a, (600, 300), (300, 180), 10, 0, 360, (40, 150, 60), -1)
    ex = Image.Exif()
    ex[274] = 6
    Image.fromarray(a).save(yol, quality=95, exif=ex)
    g1, m1, b1, rgb = sv._isle(yol)
    assert max(rgb.shape[:2]) == sv.UZUN_KENAR and rgb.shape[0] > rgb.shape[1]  # EXIF uygulandı, 500'e indi
    tahmin = yukle("tahmin")
    yakala = {}
    monkeypatch.setattr(tahmin, "islenmisten_tahmin", lambda g, m, bilgi, model_yolu=None: yakala.update(g=g, m=m, b=bilgi))
    tahmin.tahmin_et(str(yol))
    assert np.array_equal(yakala["g"], g1) and np.array_equal(yakala["m"], m1) and yakala["b"] == b1
    gm = yukle("genel_maske")
    g3, m3, _ = gm.ham_on_isle(Image.open(yol))  # PIL girdisi de aynı
    assert np.array_equal(g3, g1) and np.array_equal(m3, m1)
