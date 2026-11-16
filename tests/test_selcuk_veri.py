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
