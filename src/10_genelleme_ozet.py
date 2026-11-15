# -*- coding: utf-8 -*-
"""
10 - Genelleme deneylerinin özeti ve şekilleri (09_genelleme.py çıktılarını okur; yeni eğitim yapmaz).

Çıktılar:
  ciktilar/genel_model/ozet.json                         vekil skorlar, son modeller, C sonuçları
  ciktilar/sekiller/genel_karisiklik_matrisleri.png      dağıtılan model: A+B doğrulama ve C (Embrapa) matrisleri
  ciktilar/sekiller/genel_kaynak_bazinda.png             (sınıf, kaynak) alt kümelerinde doğruluk, eski CNN ile
  ciktilar/sekiller/genel_C_sonucu.png                   C'deki 25 CBB yaprağından doğru sayısı, model/tohum bazında
Kullanım: python src/10_genelleme_ozet.py --dagitilan <deney_klasoru_adi>
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from ortak import CIKTI, KISA_AD, SEKIL, SINIFLAR  # noqa: E402

KL = CIKTI / "genel_model"
ESKI_CNN = {"C_CBB_dogru": 2, "AB": None}  # RESULTS §6.2c: sıfırdan CNN, C hariç eğitim -> 2/25


def oku():
    s = {}
    for d in sorted((KL / "deneyler").iterdir()):
        f = d / "sonuc.json"
        if f.exists():
            s[d.name] = json.loads(f.read_text(encoding="utf-8"))
    return s


def matris_ciz(ax, M, baslik, satirlar=SINIFLAR):
    M = np.array(M)
    oran = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(oran, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            if M[i].sum() == 0:
                continue
            ax.text(j, i, f"{M[i, j]}\n%{100 * oran[i, j]:.0f}", ha="center", va="center",
                    color="white" if oran[i, j] > 0.6 else "black", fontsize=9)
    ad = [KISA_AD[s] for s in SINIFLAR]
    ax.set_xticks(range(3), ad)
    ax.set_yticks(range(3), [KISA_AD[s] for s in satirlar])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(baslik, fontsize=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dagitilan", required=True)
    a = p.parse_args()
    S = oku()
    ozet = {"vekil": {}, "son": {}}
    adaylar = sorted({k.split("_haric_")[0] for k in S if "_haric_saglikli_" in k})
    for ad in adaylar:
        r = {}
        for kay in "AB":
            k = next((x for x in S if x.startswith(f"{ad}_haric_saglikli_{kay}_")), None)
            if k:
                r[f"tutma_{kay}"] = S[k]["vekil_tutma"]
                r[f"AB_makro_f1_{kay}"] = S[k]["dogrulama_AB"]["makro_f1"]
                if "vekil_E_CBB" in S[k]:
                    r[f"E_CBB_{kay}"] = S[k]["vekil_E_CBB"]
                    r[f"E_hastalikli_{kay}"] = S[k]["vekil_E_hastalikli"]
                    r[f"E_tahmin_{kay}"] = S[k]["bugwood_vekil_test"]["karisiklik"][0]
        if "tutma_A" in r and "tutma_B" in r:
            r["vekil_skor"] = (r["tutma_A"] + r["tutma_B"]) / 2
        if "E_CBB_A" in r and "E_CBB_B" in r:  # EK 2 (on_kayit_plani): E-yari vekil CBB testi
            r["vekil_E_CBB"] = (r["E_CBB_A"] + r["E_CBB_B"]) / 2
            r["vekil_skor_E"] = (r["vekil_E_CBB"] + r["vekil_skor"]) / 2
        ozet["vekil"][ad] = r
    for k, v in S.items():
        if "C_test" in v:
            kb = v["C_test"]["kaynak_bazinda"]
            ozet["son"][k] = {"AB_dogruluk": v["dogrulama_AB"]["dogruluk"], "AB_makro_f1": v["dogrulama_AB"]["makro_f1"],
                              "C_CBB_dogru": kb.get("adi_yaprak_yanikligi|C", {}).get("dogru"),
                              "C_CBB_tahmin": kb.get("adi_yaprak_yanikligi|C", {}).get("tahmin"),
                              "C_saglikli_dogru": kb.get("saglikli|C", {}).get("dogru"),
                              "C_toplam_dogru": int(np.trace(np.array(v["C_test"]["karisiklik"]))),
                              "C_n": v["C_test"]["n"]}
            ozet["son"][k]["tum_dogruluk"] = v["dogrulama"]["dogruluk"]
            ozet["son"][k]["tum_makro_f1"] = v["dogrulama"]["makro_f1"]
            ozet["son"][k]["tum_n"] = v["dogrulama"]["n"]
            ozet["son"][k]["C_hastalikli"] = int(sum(kb.get("adi_yaprak_yanikligi|C", {}).get("tahmin", [0, 0, 0])[:2]))
            ozet["son"][k]["E_dogrulama"] = v["dogrulama"]["kaynak_bazinda"].get("adi_yaprak_yanikligi|E")
            if "dogrulama" in v and v["dogrulama"]["n"] != v["dogrulama_AB"]["n"]:
                ozet["son"][k]["D_dogrulama"] = {kk: vv for kk, vv in v["dogrulama"]["kaynak_bazinda"].items() if kk.endswith("|D")}
    # model grubu bazında ortalama ± std
    gruplar = {}
    for k, v in ozet["son"].items():
        gruplar.setdefault(k.rsplit("_t", 1)[0], []).append(v)
    ozet["son_grup"] = {g: {"tohum_sayisi": len(L),
                            **{m: [float(np.mean([x[m] for x in L])), float(np.std([x[m] for x in L]))]
                               for m in ("AB_dogruluk", "AB_makro_f1", "tum_dogruluk", "tum_makro_f1", "C_CBB_dogru",
                                         "C_hastalikli", "C_toplam_dogru")}}
                        for g, L in gruplar.items()}
    bak = KL / "c_bakis_kaydi.jsonl"
    ozet["C_bakis_sayisi"] = sum(1 for _ in open(bak, encoding="utf-8")) if bak.exists() else 0
    (KL / "ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ozet["vekil"], indent=1), json.dumps(ozet["son_grup"], indent=1))

    SEKIL.mkdir(parents=True, exist_ok=True)
    d = S[a.dagitilan]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    matris_ciz(ax[0], d["dogrulama_AB"]["karisiklik"],
               f"A+B doğrulama (n={d['dogrulama_AB']['n']}): doğruluk %{100 * d['dogrulama_AB']['dogruluk']:.1f}, "
               f"makro F1 %{100 * d['dogrulama_AB']['makro_f1']:.1f}")
    Mc = np.array(d["C_test"]["karisiklik"])
    matris_ciz(ax[1], Mc, f"C = Embrapa, eğitimde hiç yok (n={Mc.sum()}): {np.trace(Mc)}/{Mc.sum()} doğru")
    fig.suptitle("Dağıtılan genel model (EfficientNet-B0, ImageNet ön eğitimli)", fontsize=11)
    fig.tight_layout()
    fig.savefig(SEKIL / "genel_karisiklik_matrisleri.png", dpi=150)
    plt.close(fig)

    kb = {**d["dogrulama"]["kaynak_bazinda"], **{k: v for k, v in d["C_test"]["kaynak_bazinda"].items()}}
    eski = json.loads((CIKTI / "metrikler.json").read_text(encoding="utf-8")) if (CIKTI / "metrikler.json").exists() else None
    adlar = sorted(kb, key=lambda x: (x.split("|")[1], x))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(adlar))
    ax.bar(x, [100 * kb[k]["dogruluk"] for k in adlar], color=["#c44e52" if k.endswith("|C") else "#4c72b0" for k in adlar])
    for i, k in enumerate(adlar):
        ax.text(i, 100 * kb[k]["dogruluk"] + 1, f"{kb[k]['dogru']}/{kb[k]['n']}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{KISA_AD[k.split('|')[0]]}\nkaynak {k.split('|')[1]}" for k in adlar], fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_title("Dağıtılan model: (sınıf, kaynak) bazında doğruluk — mavi: doğrulama, kırmızı: C (eğitimde hiç yok)", fontsize=10)
    fig.tight_layout()
    fig.savefig(SEKIL / "genel_kaynak_bazinda.png", dpi=150)
    plt.close(fig)
    _ = eski

    kisa = {"SON_K1_b0_temel": "B0 temel", "SON_K3_b0_guclu_web": "B0 güçlü+web\n(önceki seçilen)",
            "SON_CNN_guclu_web": "Özel CNN\ngüçlü+web", "SON_E1_b0_guclu_web_E": "B0 güçlü+web+E\n(yeni)",
            "SON_E2_b0_guclu_web_E_hucre": "B0+web+E\n+hücre (yeni)"}
    etiket, dag = ["Sıfırdan CNN\n(önceki §6.2c)"], [[2, 13, 10]]
    for k in sorted(ozet["son"], key=lambda x: ("_E" in x, x.rsplit("_tam_", 1)[0] != "SON_K1_b0_temel", x)):
        g, t = k.rsplit("_tam_", 1)
        etiket.append(f"{kisa.get(g, g)}\ntohum {t[1:]}")
        dag.append(ozet["son"][k]["C_CBB_tahmin"])
    dag = np.array(dag)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(etiket)), 4.4))
    alt = np.zeros(len(dag))
    for i, (ad, renk) in enumerate([("CBB (doğru)", "#55a868"), ("Hale", "#dd8452"), ("Sağlıklı", "#4c72b0")]):
        ax.bar(range(len(dag)), dag[:, i], bottom=alt, color=renk, label=ad)
        alt += dag[:, i]
    for i, v in enumerate(dag[:, 0]):
        ax.text(i, 25.4, f"{v}/25", ha="center", fontsize=8)
    ax.axhline(17.5, ls="--", c="k", lw=0.8)
    ax.text(-0.45, 17.8, "hedef %70", fontsize=8)
    ax.set_xticks(range(len(dag)), etiket, fontsize=7)
    ax.set_ylim(0, 27)
    ax.set_ylabel("C'deki 25 CBB yaprağı için tahmin")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.set_title(f"Kaynak dışı test (Embrapa CBB; eğitim/seçimde hiç kullanılmadı) — toplam C bakışı: {ozet['C_bakis_sayisi']}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(SEKIL / "genel_C_sonucu.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
