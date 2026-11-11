# -*- coding: utf-8 -*-
"""Veri yükleme ve veri artırma (eğitim ve değerlendirme betikleri ortak kullanır)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).parent))
from ortak import GIRIS_BOYUT, ISLENMIS, SINIFLAR, VERI  # noqa: E402

SILUET_GRI = 160  # siluet deneyinde yaprak içinin boyandığı düz gri değer


def bolme_oku(bolum: str) -> pd.DataFrame:
    return pd.read_csv(VERI / f"bolme_{bolum}.csv")


def goruntuleri_yukle(df: pd.DataFrame, girdi: str = "renkli", boyut: int = GIRIS_BOYUT):
    """Ön işlenmiş görüntüleri ve maskeleri belleğe yükler -> uint8 tensör [N,4,H,W] (RGB + maske).

    girdi='siluet' ise RGB kanalları maske ile değiştirilir: yaprak içi düz gri, dışı siyah.
    Böylece modelin yalnız yaprağın DIŞ ŞEKLİNDEN (siluetinden) ne öğrenebildiği ölçülür.
    """
    X = np.zeros((len(df), 4, boyut, boyut), np.uint8)
    for i, r in enumerate(df.itertuples()):
        g = np.asarray(Image.open(ISLENMIS / r.islenmis).convert("RGB"))
        m = np.asarray(Image.open(ISLENMIS / r.maske).convert("L"))
        g = cv2.resize(g, (boyut, boyut), interpolation=cv2.INTER_AREA)
        m = (cv2.resize(m, (boyut, boyut), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
        if girdi == "siluet":
            g = np.repeat((m > 0)[..., None] * SILUET_GRI, 3, axis=2).astype(np.uint8)
        X[i, :3] = g.transpose(2, 0, 1)
        X[i, 3] = m
    return torch.from_numpy(X)


class YaprakVeri(Dataset):
    """Bellekteki tensörlerden örnek döndürür; eğitimde rastgele veri artırma uygular."""

    def __init__(self, X, y, ortalama, std, egitim=False, renk_artirma=True):
        self.X, self.y = X, torch.as_tensor(y, dtype=torch.long)
        self.ort = torch.tensor(ortalama).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.egitim = egitim
        # Geometrik artırma (RGB + maske birlikte döner): yapraklar her yönde fotoğraflanabilir
        self.geo = v2.Compose([
            v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
            v2.RandomRotation(180, fill=0),
            v2.RandomResizedCrop(GIRIS_BOYUT, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
        ])
        # Renk artırma: hafif; ton (hue) çok az oynatılır çünkü renk hastalık bilgisi taşır
        self.renk = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02) if renk_artirma else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.egitim:
            x = self.geo(x)
            rgb, m = x[:3], x[3:4]
            if self.renk is not None:
                rgb = self.renk(rgb)
            rgb = rgb * (m > 127)  # artırma sonrası arka planı yeniden tam siyah yap
        else:
            rgb = x[:3]
        rgb = rgb.float() / 255.0
        return (rgb - self.ort) / self.std, self.y[i]


def etiketler(df: pd.DataFrame, gorev: str = "hastalik"):
    if gorev == "kaynak":
        return (df["kaynak"] == "mendeley_3km9d246z2").astype(int).to_numpy()  # 0=A, 1=B
    return df["sinif"].map({s: i for i, s in enumerate(SINIFLAR)}).to_numpy()


def kanal_istatistigi(X):
    """Eğitim kümesinin kanal ortalaması ve standart sapması (normalizasyon için)."""
    f = X[:, :3].float() / 255.0
    return f.mean(dim=(0, 2, 3)).tolist(), f.std(dim=(0, 2, 3)).tolist()
