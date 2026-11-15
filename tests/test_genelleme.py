"""Kaynak dışı genelleme kodu: genel maskeleme, ışık normalizasyonu, web bölmesi, model ve tahmin_et arayüzü."""
import numpy as np
import pandas as pd
import pytest

from conftest import yukle

cv2 = pytest.importorskip("cv2")
genel_maske = yukle("genel_maske")


def _yaprak(zemin_rengi, boyut=(300, 400)):
    a = np.full((*boyut, 3), zemin_rengi, np.uint8)
    cv2.ellipse(a, (200, 150), (120, 80), 20, 0, 360, (60, 150, 40), -1)
    cv2.circle(a, (230, 140), 15, (120, 80, 30), -1)  # kahverengi leke
    return a


@pytest.mark.parametrize("zemin,beklenen", [((255, 255, 255), "beyaz"), ((10, 10, 25), "siyah"),
                                            ((150, 120, 90), "genel")])
def test_zemin_turu_ve_maske(zemin, beklenen):
    rgb = _yaprak(zemin)
    assert genel_maske.zemin_turu(rgb) == beklenen
    g, m, bilgi = genel_maske.on_isle(rgb)
    assert g.shape == (256, 256, 3) and m.shape == (256, 256)
    assert bilgi["basarili"] and bilgi["zemin"] == beklenen
    assert 0.2 < bilgi["alan_orani"] < 0.4  # elips alanı ~ %25
    assert (g[m == 0] == 0).all()  # zemin siyah


def test_maskeleme_basarisizsa_yine_goruntu_doner():
    rgb = np.full((200, 300, 3), 128, np.uint8)  # yaprak yok, düz gri
    g, m, bilgi = genel_maske.on_isle(rgb)
    assert g.shape == (256, 256, 3) and not bilgi["basarili"]


def test_web_bolmesi_grup_ayrik_ve_oran():
    torch = pytest.importorskip("torch")  # noqa: F841
    gm = yukle("09_genelleme")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sinif": ["hale_yanikligi"] * 100 + ["saglikli"] * 200,
                       "grup": [f"h{rng.integers(0, 20)}" for _ in range(100)] + [f"s{rng.integers(0, 40)}" for _ in range(200)]})
    e, d = gm.web_bol(df)
    assert not set(e.grup) & set(d.grup)
    assert len(e) + len(d) == 300
    assert 0.2 < len(d) / 300 < 0.45


def test_isik_normalize_maske_ici_ortalama():
    torch = pytest.importorskip("torch")
    gm = yukle("09_genelleme")
    x = torch.rand(3, 32, 32) * 0.2
    m = torch.zeros(1, 32, 32)
    m[:, 8:24, 8:24] = 1
    y = gm.isik_normalize(x, m)
    lum = 0.299 * y[0] + 0.587 * y[1] + 0.114 * y[2]
    assert abs(float(lum[m[0] > 0].mean()) - 0.45) < 0.03
    assert float(y[:, m[0] == 0].abs().max()) == 0.0


def test_fasulye_cnn_govdesi_ve_tahmin_et(tmp_path):
    torch = pytest.importorskip("torch")
    gm = yukle("09_genelleme")
    m = gm.model_kur("fasulye_cnn", onegitimli=False).eval()
    with torch.no_grad():
        c, k = m(torch.randn(2, 3, 128, 128), grl=1.0)
    assert tuple(c.shape) == (2, 3) and tuple(k.shape) == (2, 3)
    yol = tmp_path / "model.pt"
    torch.save({"model": m.state_dict(), "omurga": "fasulye_cnn", "boyut": 128, "isik": True,
                "ortalama": gm.IMAGENET_ORT, "std": gm.IMAGENET_STD,
                "siniflar": ["adi_yaprak_yanikligi", "hale_yanikligi", "saglikli"]}, yol)
    tahmin = yukle("tahmin")
    from PIL import Image
    s = tahmin.tahmin_et(Image.fromarray(_yaprak((255, 255, 255))), model_yolu=yol)
    assert s["sinif"] in s["olasiliklar"]
    assert abs(sum(s["olasiliklar"].values()) - 1) < 1e-3
    assert s["on_isleme"]["zemin"] == "beyaz"
    s2 = tahmin.tahmin_et(np.full((50, 50, 3), 128, np.uint8), model_yolu=yol)  # maskesiz: yine çalışır
    assert s2["on_isleme"]["basarili"] is False


def test_grup_bol_tam_grup_ayrik_ve_hedefe_yakin():
    pytest.importorskip("torch")
    gm = yukle("09_genelleme")
    df = pd.DataFrame({"grup": ["a"] * 2 + ["b"] * 3 + ["c"] * 5 + ["d"] * 4 + ["e", "f", "g"] + ["h"] * 3})
    for oran, hedef in ((0.5, 10), (0.3, 6)):
        e, t = gm.grup_bol_tam(df, oran)
        assert not set(e.grup) & set(t.grup)
        assert len(e) + len(t) == 20 and len(t) == hedef
        e2, t2 = gm.grup_bol_tam(df, oran)
        assert sorted(t2.grup) == sorted(t.grup)  # aynı tohum -> aynı bölme


def test_bugwood_kirpma_kutusu_etiketi_atar():
    bw = yukle("11_bugwood_veri")
    assert bw.kirpma_kutusu("", 1536, 1000) == (0, 0, 1536, 930)
    assert bw.kirpma_kutusu("10,20,500,990", 1536, 1000) == (10, 20, 500, 930)
    with pytest.raises(ValueError):
        bw.kirpma_kutusu("0,0,10,10", 1536, 1000)


@pytest.mark.parametrize("sekil", [(1, 10000), (10000, 1), (3, 5000), (1, 1)])
def test_asiri_en_boy_orani_istisna_firlatmaz(sekil):
    rgb = np.full((*sekil, 3), 128, np.uint8)
    g, m, bilgi = genel_maske.on_isle(rgb)
    assert g.shape == (256, 256, 3) and m.shape == (256, 256) and not bilgi["basarili"]


def test_web_bol_bir_tarafi_bos_birakirsa_hata():
    pytest.importorskip("torch")
    gm = yukle("09_genelleme")
    df = pd.DataFrame({"sinif": ["saglikli"] * 100, "grup": ["a"] * 80 + ["b"] * 20})
    with pytest.raises(ValueError, match="saglikli"):
        gm.web_bol(df)


def test_c_bakis_kaydi_iki_kayit(tmp_path):
    pytest.importorskip("torch")
    gm = yukle("09_genelleme")
    yol = tmp_path / "c_bakis_kaydi.jsonl"
    gm.c_kaydi_yaz({"durum": "basladi", "deney": "x"}, yol)
    gm.c_kaydi_yaz({"durum": "tamamlandi", "deney": "x", "C_toplam_dogru": 3}, yol)
    import json
    satirlar = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines()]
    assert [s["durum"] for s in satirlar] == ["basladi", "tamamlandi"]
    assert all("zaman" in s for s in satirlar)
