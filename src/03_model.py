# -*- coding: utf-8 -*-
"""
03 - MODEL: sıfırdan tasarlanmış evrişimli sinir ağı ("FasulyeCNN").

Hocanın gösterdiği "Model-2" (mısır yaprağı; 4x4 evrişim + ReLU, 2x2 havuzlama, %20 dropout,
softmax çıkış) ruhuna uygun, kendi tasarımımız. Hazır mimari ya da önceden eğitilmiş ağırlık
KULLANILMAZ; tüm ağırlıklar rastgele başlatılır.

Katmanlar (Keras tarzı sayımla 16 katman; BatchNorm evrişim katmanının parçası sayıldı):
  Giriş 128x128x3
  [1] Evrişim 4x4, 32 filtre, 'same', BatchNorm, ReLU      -> 128x128x32
  [2] MaksHavuz 2x2                                         -> 64x64x32
  [3] Dropout %20
  [4] Evrişim 4x4, 64 filtre, BN, ReLU                       -> 64x64x64
  [5] MaksHavuz 2x2                                         -> 32x32x64
  [6] Dropout %20
  [7] Evrişim 4x4, 128 filtre, BN, ReLU                      -> 32x32x128
  [8] MaksHavuz 2x2                                         -> 16x16x128
  [9] Dropout %20
  [10] Evrişim 4x4, 256 filtre, BN, ReLU                     -> 16x16x256
  [11] MaksHavuz 2x2                                        -> 8x8x256
  [12] Dropout %20
  [13] Evrişim 4x4, 256 filtre, BN, ReLU (Grad-CAM katmanı)  -> 8x8x256
  [14] Global ortalama havuzlama                            -> 256
  [15] Tam bağlı 128 + ReLU + Dropout %30
  [16] Tam bağlı 3 + Softmax (softmax kayıp fonksiyonunun içinde; tahminde açıkça uygulanır)

Tasarım gerekçeleri:
  * 128x128 giriş: 50x50'de küçük lezyonlar ve hale (sarı halka) birkaç piksele iniyor;
    128 piksel lezyon ayrıntısını korurken 8 GB GPU'da hızlı eğitime izin veriyor.
  * Global ortalama havuzlama: düzleştirme (flatten) + büyük tam bağlı katmana göre çok daha
    az parametre (küçük veri setinde aşırı öğrenmeyi azaltır) ve konumdan bağımsızlık.
  * BatchNorm: sıfırdan eğitimi kararlı ve hızlı hale getirir.
"""
import warnings

import torch
import torch.nn as nn

# 4x4 (çift boyutlu) çekirdekte 'same' dolgu için PyTorch'un bilgilendirme uyarısını gizle
warnings.filterwarnings('ignore', message="Using padding='same'")


class EvrisimBlogu(nn.Module):
    """Evrişim(4x4) + BatchNorm + ReLU [+ 2x2 maks havuz + %20 dropout]."""

    def __init__(self, giris, cikis, havuz=True, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(giris, cikis, kernel_size=4, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(cikis)
        self.relu = nn.ReLU(inplace=False)
        self.havuz = nn.MaxPool2d(2) if havuz else nn.Identity()
        self.drop = nn.Dropout(dropout) if havuz and dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.havuz(self.relu(self.bn(self.conv(x)))))


class FasulyeCNN(nn.Module):
    def __init__(self, sinif_sayisi=3, kanallar=(32, 64, 128, 256), son_kanal=256):
        super().__init__()
        bloklar, onceki = [], 3
        for k in kanallar:
            bloklar.append(EvrisimBlogu(onceki, k, havuz=True, dropout=0.2))
            onceki = k
        self.ozellik = nn.Sequential(*bloklar)
        # Son evrişim bloğu havuzsuz: Grad-CAM bu bloğun çıktısında (8x8) hesaplanır
        self.son_evrisim = EvrisimBlogu(onceki, son_kanal, havuz=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.siniflayici = nn.Sequential(
            nn.Flatten(),
            nn.Linear(son_kanal, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, sinif_sayisi),
        )
        for m in self.modules():  # He başlatma (ReLU için uygun)
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        a = self.son_evrisim(self.ozellik(x))
        return self.siniflayici(self.gap(a))  # logit; olasılık için softmax uygulanır


def parametre_sayisi(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def katman_tablosu(model, giris_boyut=128):
    """Katman adı, çıktı boyutu ve parametre sayısını listeler (SONUCLAR.md için)."""
    satirlar, kancalar = [], []

    def kanca(ad):
        def f(mod, i, o):
            satirlar.append((ad, type(mod).__name__, tuple(o.shape[1:]),
                             sum(p.numel() for p in mod.parameters(recurse=False))))
        return f

    for ad, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            kancalar.append(mod.register_forward_hook(kanca(ad)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, giris_boyut, giris_boyut))
    for k in kancalar:
        k.remove()
    return satirlar


if __name__ == "__main__":
    m = FasulyeCNN()
    for s in katman_tablosu(m):
        print(f"{s[0]:28s} {s[1]:18s} {str(s[2]):18s} {s[3]:>9,d}")
    print(f"Toplam eğitilebilir parametre: {parametre_sayisi(m):,}")
