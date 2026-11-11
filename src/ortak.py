# -*- coding: utf-8 -*-
"""Ortak sabitler ve yardımcı fonksiyonlar (tüm betikler bunu içe aktarır)."""
import os
import random
from pathlib import Path

import numpy as np

# Proje kökü: bu dosyanın bir üst klasörü (PROJE_KOK ortam değişkeniyle değiştirilebilir)
KOK = Path(__file__).resolve().parent.parent
# resolve() sembolik bağlantıları çözebilir; bu yüzden ortam değişkeni ya da betiğin
# çağrıldığı yol tercih edilir.
if os.environ.get("PROJE_KOK"):
    KOK = Path(os.environ["PROJE_KOK"])
else:
    KOK = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

VERI = KOK / "data"
HAM = VERI / "raw"
ISLENMIS = VERI / "islenmis"
CIKTI = KOK / "ciktilar"
SEKIL = CIKTI / "sekiller"

# Sınıf sırası sabit: indeks 0,1,2
SINIFLAR = ["adi_yaprak_yanikligi", "hale_yanikligi", "saglikli"]
SINIF_ADLARI_TR = {
    "adi_yaprak_yanikligi": "Adi yaprak yanıklığı (CBB)",
    "hale_yanikligi": "Hale yanıklığı",
    "saglikli": "Sağlıklı",
}
KISA_AD = {"adi_yaprak_yanikligi": "CBB", "hale_yanikligi": "Hale", "saglikli": "Sağlıklı"}
KAYNAK_KISA = {
    "mendeley_ykvcrjffzd": "A (Mendeley ykvcrjffzd, siyah kumaş)",
    "mendeley_3km9d246z2": "B (Mendeley 3km9d246z2, beyaz zemin)",
    "embrapa_pddb": "C (Embrapa PDDB, kalibrasyon kartı)",
}

TOHUM = 42
KAYDET_BOYUT = 256   # ön işlenmiş görüntülerin diske kaydedildiği boyut
GIRIS_BOYUT = 128    # modelin giriş boyutu (gerekçe: SONUCLAR.md)


def tohum_ayarla(tohum: int = TOHUM):
    """Tüm rastgele sayı üreteçlerini sabitler (yeniden üretilebilirlik)."""
    random.seed(tohum)
    np.random.seed(tohum)
    try:
        import torch
        torch.manual_seed(tohum)
        torch.cuda.manual_seed_all(tohum)
    except ImportError:
        pass


def kaynak_kisa(kaynak: str) -> str:
    """'mendeley_ykvcrjffzd' -> 'A' gibi tek harfli kaynak kodu."""
    return {"mendeley_ykvcrjffzd": "A", "mendeley_3km9d246z2": "B"}.get(kaynak, "C")
