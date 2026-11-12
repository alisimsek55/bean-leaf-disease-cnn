"""scripts/yakin_kopya.py: seri cekim anahtari ve dHash + seri birlesik gruplama (goruntu gerekmez)."""
import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def betik(ad):
    spec = importlib.util.spec_from_file_location(ad, SCRIPTS / f"{ad}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


yk = betik("yakin_kopya")
A = "mendeley_ykvcrjffzd"


def satir(dosya, sinif="adi_yaprak_yanikligi", kaynak=A):
    return {"dosya": dosya, "sinif": sinif, "kaynak": kaynak}


def gruplar(satirlar, dhash_ciftleri=(), pencere=2):
    return yk.grup_etiketleri(len(satirlar), list(dhash_ciftleri) + yk.seri_baglantilari(satirlar, pencere))


def test_saniye_anahtari_ayristirma():
    assert yk.saniye_anahtari("raw/adi_yaprak_yanikligi/bdcp__IMG_20240105_235312_641.jpg") == "20240105_235312"
    assert yk.saniye_anahtari(r"raw\saglikli\bdcp__IMG_20240106_002216_000.jpg") == "20240106_002216"
    assert yk.saniye_anahtari("IMG_20240106_002216.jpg") == "20240106_002216"
    # zaman damgasi olmayan adlar: kaynak B ve C
    assert yk.saniye_anahtari("raw/saglikli/bhd__Good_Leaf_671.png") is None
    assert yk.saniye_anahtari("raw/adi_yaprak_yanikligi/pddb_cbb__DSC_0629.jpg") is None
    # gecersiz tarih/saat ve fazla basamak
    assert yk.saniye_anahtari("bdcp__IMG_20241399_256199_000.jpg") is None
    assert yk.saniye_anahtari("bdcp__IMG_20240105_2353120_000.jpg") is None


def test_ayni_saniye_ayni_grup():
    s = [satir("bdcp__IMG_20240105_235312_434.jpg"), satir("bdcp__IMG_20240105_235312_641.jpg"),
         satir("bdcp__IMG_20240105_235312_823.jpg")]
    assert len(set(gruplar(s))) == 1


def test_bir_iki_saniye_arayla_ayni_grup_saniye_siniri():
    s = [satir("bdcp__IMG_20240105_235959_900.jpg"), satir("bdcp__IMG_20240106_000000_100.jpg"),
         satir("bdcp__IMG_20240106_000002_000.jpg")]
    assert len(set(gruplar(s))) == 1


def test_pencere_siniri_milisaniye_ile_tam_karsilastirilir():
    # 2.998 sn: saniye alanlari 2 sn farkli gorunse de pencere (2 sn) disinda -> ayri
    s = [satir("bdcp__IMG_20240105_000000_001.jpg"), satir("bdcp__IMG_20240105_000002_999.jpg")]
    assert yk.seri_baglantilari(s, 2) == []
    assert len(set(gruplar(s))) == 2
    # tam 2.000 sn: sinir dahil -> ayni grup
    s = [satir("bdcp__IMG_20240105_000000_500.jpg"), satir("bdcp__IMG_20240105_000002_500.jpg")]
    assert len(set(gruplar(s))) == 1
    # 1.999 sn -> ayni grup
    s = [satir("bdcp__IMG_20240105_000000_001.jpg"), satir("bdcp__IMG_20240105_000002_000.jpg")]
    assert len(set(gruplar(s))) == 1
    # 2.001 sn -> ayri
    s = [satir("bdcp__IMG_20240105_000000_000.jpg"), satir("bdcp__IMG_20240105_000002_001.jpg")]
    assert len(set(gruplar(s))) == 2
    # milisaniye alani olmayan adlar 0 ms sayilir
    assert yk.cekim_zamani("IMG_20240106_002216.jpg") == yk.cekim_zamani("IMG_20240106_002216_000.jpg")
    assert yk.cekim_zamani("bdcp__IMG_20240105_235312_641.jpg").microsecond == 641000
    s = [satir("IMG_20240105_000000.jpg"), satir("IMG_20240105_000002.jpg"), satir("IMG_20240105_000005.jpg")]
    assert len(set(gruplar(s))) == 2


def test_pencere_disi_ve_farkli_dakika_ayri_grup():
    s = [satir("bdcp__IMG_20240105_235312_000.jpg"), satir("bdcp__IMG_20240105_235315_000.jpg"),
         satir("bdcp__IMG_20240105_235412_000.jpg")]
    g = gruplar(s)
    assert len(set(g)) == 3


def test_zincirleme_seri_tek_grup():
    # her ardisik fark 2 sn, uc-uca 6 sn: tek seri
    s = [satir(f"bdcp__IMG_20240105_2353{x:02d}_000.jpg") for x in (10, 12, 14, 16)]
    assert len(set(gruplar(s))) == 1


def test_farkli_kaynak_veya_sinif_seri_ile_baglanmaz():
    s = [satir("bdcp__IMG_20240105_235312_000.jpg"),
         satir("bdcp__IMG_20240105_235312_500.jpg", sinif="saglikli"),
         satir("xx__IMG_20240105_235312_000.jpg", kaynak="baska")]
    assert len(set(gruplar(s))) == 3


def test_damgasiz_adlar_etkilenmez():
    s = [satir("bhd__Good_Leaf_1.png", "saglikli", "mendeley_3km9d246z2"),
         satir("bhd__Good_Leaf_2.png", "saglikli", "mendeley_3km9d246z2"),
         satir("pddb_cbb__DSC_0629.jpg", kaynak="embrapa_pddb"),
         satir("pddb_cbb__DSC_0630.jpg", kaynak="embrapa_pddb")]
    assert yk.seri_baglantilari(s) == []
    assert gruplar(s) == ["g00000", "g00001", "g00002", "g00003"]
    # dHash baglantisi varsa yine birlesir
    assert gruplar(s, [(0, 1)]) == ["g00000", "g00000", "g00001", "g00002"]


def test_dhash_ve_seri_baglantilari_gecisli_birlesir():
    s = [satir("bdcp__IMG_20240105_235312_000.jpg"),   # 0 --seri-- 1
         satir("bdcp__IMG_20240105_235313_000.jpg"),   # 1 --dHash-- 2
         satir("bdcp__IMG_20240105_235500_000.jpg"),   # 2 --seri-- 3
         satir("bdcp__IMG_20240105_235501_000.jpg"),
         satir("bhd__Good_Leaf_7.png", "saglikli", "mendeley_3km9d246z2"),  # 3 --dHash-- 4
         satir("bdcp__IMG_20240105_240000_000.jpg")]   # gecersiz saat -> damgasiz, tek basina
    g = gruplar(s, [(1, 2), (3, 4)])
    assert len(set(g[:5])) == 1
    assert g[5] != g[0]


def test_dhash_baglantilari_esik():
    h = [0b0000, 0b0001, 0b0111, 0b1111_1111]
    ciftler = yk.dhash_baglantilari(h, esik=1)
    assert ciftler == [(0, 1, 1)]
    assert yk.grup_etiketleri(4, [(i, j) for i, j, _ in ciftler]) == ["g00000", "g00000", "g00001", "g00002"]


def test_etiketler_ilk_gorulme_sirasinda():
    assert yk.grup_etiketleri(5, [(4, 0), (3, 1)]) == ["g00000", "g00001", "g00002", "g00001", "g00000"]


def test_elle_birlestirilen_embrapa_cifti():
    s = [satir("raw/saglikli/pddb_saglikli__2.jpg", "saglikli", "embrapa_pddb"),
         satir("raw/saglikli/pddb_saglikli__1.jpg", "saglikli", "embrapa_pddb")]
    assert yk.elle_baglantilar(s) == [(1, 0)]
    assert yk.elle_baglantilar(s[:1]) == []
