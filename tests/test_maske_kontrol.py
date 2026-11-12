"""src/01b_maske_kontrol.py: goruntusu olmayan (yaprak_yok) rapor satirlari temas sayfasina girmemeli."""
import pandas as pd
import pytest
from PIL import Image

from conftest import yukle

mk = yukle("01b_maske_kontrol")

SUTUNLAR = ["dosya", "islenmis", "maske", "sinif", "kaynak", "grup", "bayraklar", "dislandi"]


def rapor_yaz(klasor):
    """Bir normal + bir yaprak_yok (bos yollu) + bir elle dislanmis (goruntusu duran) satirlik rapor CSV'si."""
    (klasor / "goruntu").mkdir()
    for ad in ("a.png", "c.png"):
        Image.new("RGB", (16, 16), (0, 120, 0)).save(klasor / "goruntu" / ad)
    pd.DataFrame([
        ["raw/saglikli/a.jpg", "goruntu/a.png", "maske/a.png", "saglikli", "k1", "g1", "", False],
        ["raw/saglikli/b.jpg", "", "", "saglikli", "k1", "g2", "yaprak_yok", True],
        ["raw/saglikli/c.jpg", "goruntu/c.png", "maske/c.png", "saglikli", "k1", "g3", "renksiz_maske", True],
    ], columns=SUTUNLAR).to_csv(klasor / "on_isleme_raporu.csv", index=False)
    return pd.read_csv(klasor / "on_isleme_raporu.csv").fillna({"bayraklar": ""})


def test_bos_yollu_satir_ayrilir(tmp_path):
    rap = rapor_yaz(tmp_path)
    assert rap["islenmis"].isna().sum() == 1  # bos CSV hucresi NaN olarak okunur (hatanin kaynagi)
    var, yok = mk.goruntulu_satirlar(rap)
    assert list(var["dosya"]) == ["raw/saglikli/a.jpg", "raw/saglikli/c.jpg"]
    assert list(yok["dosya"]) == ["raw/saglikli/b.jpg"]
    assert list(yok["bayraklar"]) == ["yaprak_yok"]
    assert var["islenmis"].map(lambda y: isinstance(y, str) and y != "").all()


def test_bosluk_ve_tek_bos_sutun_da_goruntusuz_sayilir():
    rap = pd.DataFrame([["x", "  ", "m.png", "s", "k", "g", "", False],
                        ["y", "i.png", None, "s", "k", "g", "", False],
                        ["z", "i.png", "m.png", "s", "k", "g", "", False]], columns=SUTUNLAR)
    var, yok = mk.goruntulu_satirlar(rap)
    assert list(var["dosya"]) == ["z"] and list(yok["dosya"]) == ["x", "y"]


@pytest.mark.parametrize("mod", ["bayrakli", "rastgele"])
def test_main_bos_yolu_acmaz_ve_cokmez(tmp_path, monkeypatch, capsys, mod):
    rapor_yaz(tmp_path)
    acilan = []
    gercek_open = Image.open

    def izleyen_open(yol, *a, **k):
        acilan.append(yol)
        return gercek_open(yol, *a, **k)

    monkeypatch.setattr(mk, "ISLENMIS", tmp_path)
    monkeypatch.setattr(mk, "CIKTI", tmp_path / "cikti")
    monkeypatch.setattr(mk.Image, "open", izleyen_open)
    monkeypatch.setattr(mk.sys, "argv", ["01b_maske_kontrol.py", mod])
    mk.main()  # duzeltmeden once: TypeError (Path / float)
    assert acilan and tmp_path not in acilan  # bos yol ISLENMIS / "" == ISLENMIS olurdu
    assert {y.name for y in acilan} <= {"a.png", "c.png"}
    cikti = capsys.readouterr().out
    assert "raw/saglikli/b.jpg" in cikti and "yaprak_yok" in cikti
    assert list((tmp_path / "cikti" / "maske_kontrol").glob("*.png"))
