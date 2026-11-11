# -*- coding: utf-8 -*-
"""
06 - HASTALIK ŞİDDETİ TAHMİNİ (kural tabanlı, YAKLAŞIK).

Veri setinde uzman şiddet etiketi YOK. Hocanın "sağlıklı / az hastalıklı / hastalıklı" yaprak
gösterme isteği için, yaprak maskesi içinde "yeşil olmayan" (sarı, kahverengi, kırmızımsı,
nekrotik koyu) piksellerin oranı hesaplanır ve iki eşikle üç düzeye ayrılır.
Bu bir UZMAN ETİKETİ DEĞİLDİR; yalnızca renk/piksel oranına dayalı yaklaşık bir ölçüdür
(ör. damar, parlama, gölge ve alt yüz rengi de lezyon sayılabilir).

Lezyon pikseli tanımı, yaprak maskesi 2 piksel aşındırıldıktan (kenar karışımı atılır) ve 3x3
medyan süzgecinden sonra (OpenCV birimleri: HSV'de H 0-180, S/V 0-255; Lab'de a* 128 merkezli):
    lezyon = (a* - 128 > A_KIRMIZI)                 kırmızımsı / kahverengi / nekrotik doku
             veya (H < H_SARI ve S > S_RENKLI)       sarı-turuncu (kloroz, hale)
             veya (V < V_KOYU)                       çok koyu nekroz
Sağlıklı yeşil doku a* < 128 (yeşil-kırmızı ekseninde yeşil tarafta) olduğundan dışarıda kalır.
İlk denemede "düşük doygunluk (S<35) = solgun kuru doku" ölçütü de vardı; ancak kaynak A'daki
grimsi alt yüzlü SAĞLIKLI yapraklar bu yüzden %70-90 "lezyonlu" çıktı; ölçüt kaldırıldı
(gri-beyaz kuru doku bu yüzden kısmen kaçabilir). 3x3 açma ile tek piksellik gürültü silinir.Eşikler (T1, T2) veriye bakılarak seçilir (bkz. main: sağlıklı sınıfın dağılımı) ve
ciktilar/siddet_ozeti.json'a yazılır.

Çıktılar: ciktilar/siddet_tahmini.csv, ciktilar/siddet_ozeti.json,
          ciktilar/sekiller/siddet_duzeyi_ornekleri.png, ciktilar/sekiller/siddet_dagilimi.png
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, ISLENMIS, SEKIL, SINIFLAR, SINIF_ADLARI_TR, TOHUM  # noqa: E402

A_KIRMIZI = 4             # a* - 128 bunun üstündeyse kırmızımsı/kahverengi
H_SARI = 27               # H < 27 (≈ 54°) sarı-turuncu-kahve tonları
S_RENKLI = 60             # sarılık ölçütü için en az doygunluk (gri pikseller tonsuzdur)
V_KOYU = 45               # bunun altı koyu nekroz
DUZEYLER = ["sağlıklı", "az hastalıklı", "hastalıklı"]


def lezyon_maskesi(rgb: np.ndarray, maske: np.ndarray):
    """rgb: HxWx3 uint8, maske: HxW (0/255). Döndürür: (lezyon ikili maske, değerlendirilen yaprak maskesi)."""
    yap = cv2.erode((maske > 127).astype(np.uint8), np.ones((5, 5), np.uint8))
    rgb = cv2.medianBlur(rgb, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1].astype(int) - 128
    lez = ((a > A_KIRMIZI) | ((h < H_SARI) & (s > S_RENKLI)) | (v < V_KOYU)).astype(np.uint8)
    lez = cv2.morphologyEx(lez, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) & yap
    return lez, yap


def lezyon_orani(rgb, maske) -> float:
    lez, yap = lezyon_maskesi(rgb, maske)
    return float(lez.sum() / max(yap.sum(), 1))


def duzey(oran, t1, t2):
    return DUZEYLER[0] if oran < t1 else (DUZEYLER[1] if oran < t2 else DUZEYLER[2])


def main():
    SEKIL.mkdir(parents=True, exist_ok=True)
    rap = pd.read_csv(ISLENMIS / "on_isleme_raporu.csv")
    rap = rap[~rap["dislandi"]].reset_index(drop=True)
    oranlar = []
    for r in rap.itertuples():
        g = np.asarray(Image.open(ISLENMIS / r.islenmis).convert("RGB"))
        m = np.asarray(Image.open(ISLENMIS / r.maske).convert("L"))
        oranlar.append(lezyon_orani(g, m))
    rap["lezyon_orani"] = np.round(oranlar, 4)

    # --- Eşik seçimi (veriye dayalı) ---
    # T1: sağlıklı sınıfın %99'luk dilimi, yukarı doğru %0.5'in katına yuvarlanır -> "sağlıklı"
    #     etiketli yaprakların en az %99'u bu oranın altında; T1 kural yönteminin taban gürültüsüdür
    #     (damar, parlama, kenar gölgesi).
    # T2: hastalıklı sınıfların (CBB + hale) medyanı, %1'e yuvarlanır -> hastalıklı yaprakların
    #     yaklaşık yarısı "hastalıklı", daha hafif yarısı "az hastalıklı" düzeyine düşer.
    saglikli = rap.loc[rap.sinif == "saglikli", "lezyon_orani"]
    hasta = rap.loc[rap.sinif != "saglikli", "lezyon_orani"]
    t1 = float(np.ceil(np.quantile(saglikli, 0.99) * 200) / 200)
    t2 = round(float(np.median(hasta)), 2)
    if t2 <= t1:
        t2 = round(2 * t1, 3)
    rap["siddet_duzeyi"] = [duzey(o, t1, t2) for o in rap["lezyon_orani"]]
    rap[["dosya", "islenmis", "sinif", "kaynak", "lezyon_orani", "siddet_duzeyi"]].to_csv(
        CIKTI / "siddet_tahmini.csv", index=False)

    capraz = pd.crosstab(rap["sinif"], rap["siddet_duzeyi"]).reindex(index=SINIFLAR, columns=DUZEYLER, fill_value=0)
    istat = rap.groupby("sinif")["lezyon_orani"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(4)
    ozet = {
        "yontem": "yaprak maskesi içinde yeşil olmayan piksel oranı (kural tabanlı, uzman etiketi değil)",
        "renk_parametreleri": {"A_kirmizi_esik": A_KIRMIZI, "H_sari_ust": H_SARI, "S_renkli_alt": S_RENKLI, "V_koyu_alt": V_KOYU,
                              "maske_asindirma": "5x5", "medyan": "3x3", "acma": "3x3"},
        "esikler": {"T1_saglikli_ust": t1, "T2_az_hastalikli_ust": t2,
                    "T1_kural": "saglikli sinifin lezyon oraninin %99'luk dilimi, yukari dogru %0.5'in katina yuvarlandi",
                    "T2_kural": "CBB+hale siniflarinin lezyon orani medyani, %1'e yuvarlandi",
                    "T1_ham": float(np.quantile(saglikli, 0.99)), "T2_ham": float(np.median(hasta))},
        "duzey_sayilari_toplam": rap["siddet_duzeyi"].value_counts().reindex(DUZEYLER, fill_value=0).to_dict(),
        "sinif_x_duzey": {s: {d: int(v) for d, v in r.items()} for s, r in capraz.iterrows()},
        "sinif_lezyon_orani_istatistigi": {s: {k: float(v) for k, v in r.items()} for s, r in istat.iterrows()},
        "kaynak_bazinda_medyan": {f"{s}|{k}": float(v) for (s, k), v in
                                  rap.groupby(["sinif", "kaynak"])["lezyon_orani"].median().round(4).items()},
    }
    (CIKTI / "siddet_ozeti.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: ozet[k] for k in ["esikler", "duzey_sayilari_toplam", "sinif_x_duzey"]},
                     ensure_ascii=False, indent=1))
    print(istat.to_string())

    # --- Şekil 1: her düzeyden örnekler (görüntü + lezyon katmanı) ---
    rng = np.random.default_rng(TOHUM)
    n = 5
    fig, eks = plt.subplots(6, n, figsize=(2.3 * n, 2.5 * 6))
    for di, d in enumerate(DUZEYLER):
        aday = rap[rap.siddet_duzeyi == d]
        # her düzeyde mümkün olduğunca farklı sınıflardan örnek al
        sec = pd.concat([g.sample(min(len(g), 3), random_state=TOHUM) for _, g in aday.groupby("sinif")])
        sec = sec.sample(frac=1, random_state=TOHUM).head(n)
        for j in range(n):
            a1, a2 = eks[2 * di, j], eks[2 * di + 1, j]
            a1.axis("off"); a2.axis("off")
            if j >= len(sec):
                continue
            r = sec.iloc[j]
            g = np.asarray(Image.open(ISLENMIS / r.islenmis).convert("RGB"))
            m = np.asarray(Image.open(ISLENMIS / r.maske).convert("L"))
            lez, _ = lezyon_maskesi(g, m)
            ust = g.copy()
            ust[lez > 0] = (0.4 * ust[lez > 0] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
            a1.imshow(g)
            a1.set_title(f"{d}\n{SINIF_ADLARI_TR[r.sinif].split(' (')[0]} | oran %{100 * r.lezyon_orani:.1f}",
                         fontsize=8)
            a2.imshow(ust)
            a2.set_title("lezyon pikselleri (mor)", fontsize=7)
    fig.suptitle(f"Kural tabanlı şiddet düzeyleri (yaklaşık; uzman etiketi değil)\n"
                 f"lezyon oranı: sağlıklı < %{100 * t1:.1f} ≤ az hastalıklı < %{100 * t2:.0f} ≤ hastalıklı", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(SEKIL / "siddet_duzeyi_ornekleri.png", dpi=130)
    plt.close(fig)

    # --- Şekil 2: sınıflara göre lezyon oranı dağılımı ve eşikler ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, renk in zip(SINIFLAR, ["#c0392b", "#e67e22", "#27ae60"]):
        ax.hist(100 * rap.loc[rap.sinif == s, "lezyon_orani"], bins=50, range=(0, 100), alpha=0.55,
                label=SINIF_ADLARI_TR[s], color=renk, density=True)
    for t, ad in [(t1, "T1"), (t2, "T2")]:
        ax.axvline(100 * t, color="k", ls="--", lw=1)
        ax.text(100 * t + 0.5, ax.get_ylim()[1] * 0.9, f"{ad}=%{100 * t:.1f}", fontsize=9)
    ax.set_xlabel("Yaprak içindeki lezyon (yeşil olmayan) piksel oranı (%)")
    ax.set_ylabel("Yoğunluk")
    ax.set_title("Lezyon oranı dağılımı ve şiddet eşikleri")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SEKIL / "siddet_dagilimi.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
