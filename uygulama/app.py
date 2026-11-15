# -*- coding: utf-8 -*-
"""Fasulye yaprağı hastalık karar destek aracı (yerel web uygulaması).

Çalıştırma:  .\.venv\Scripts\python -m streamlit run uygulama\app.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

KOK = Path(__file__).resolve().parent.parent
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))

from uygulama import model_adaptoru as ma  # noqa: E402

GUVEN_ESIGI = 0.60

AD_TR = {
    "adi_yaprak_yanikligi": "Adi yaprak yanıklığı (CBB)",
    "hale_yanikligi": "Hale yanıklığı",
    "saglikli": "Sağlıklı",
}
ETKEN = {
    "adi_yaprak_yanikligi": "Xanthomonas axonopodis pv. phaseoli (bakteri)",
    "hale_yanikligi": "Pseudomonas syringae pv. phaseolicola (bakteri)",
}
DUZEY_RENK = {"sağlıklı": "green", "az hastalıklı": "orange", "çok hastalıklı": "red"}

BAKTERIYEL_ONERI = [
    "Hastalıklı yaprak ve bitki artıklarını tarladan uzaklaştırın; tarlada bırakmayın veya gömün/yakın.",
    "Sertifikalı, hastalıksız (temiz) tohum kullanın; hastalıklı bitkilerden tohum almayın.",
    "Yağmurlama sulamadan kaçının; damla veya karık sulamayı tercih edin.",
    "Yapraklar ıslakken (yağmur, çiy, sulama sonrası) tarlaya girmeyin ve bitkiler arasında çalışmayın; hastalık el, alet ve giysiyle yayılır.",
    "Ekim nöbetine uyun; aynı tarlaya art arda fasulye ekmeyin.",
    "Bakır içerikli preparatlar konusunda ilçe/il tarım müdürlüğüne veya bir ziraat mühendisine danışın; ilaç ve doz seçimini uzman belirlesin.",
]
SAGLIKLI_ONERI = [
    "Yaprakta belirgin bir hastalık belirtisi saptanmadı. Tarlayı düzenli (haftalık) gezip kontrol etmeyi sürdürün.",
    "Önleyici olarak sertifikalı tohum, damla/karık sulama ve ekim nöbeti uygulayın.",
    "Yeni leke, sararma veya kuruma görürseniz yaprağın yakın çekim fotoğrafını çekip tekrar deneyin.",
]


def oneri_listesi(sinif):
    return SAGLIKLI_ONERI if sinif == "saglikli" else BAKTERIYEL_ONERI


@st.cache_resource(show_spinner="Model yükleniyor...")
def _hazirla():
    return ma.model_adi()


TUR_NOTU = ("Tür ayrımı (adi yanıklık / hale) sınırlı veriyle eğitildi ve farklı koşullardaki "
            "fotoğraflarda güvenilir değil; iki hastalık da bakteriyeldir ve önerilen önlemler aynıdır.")


def goster(r: ma.Sonuc):
    ol = r.olasiliklar
    p_hast = ma.hastalik_olasiligi(ol)
    p_sag = 1.0 - p_hast
    hasta = p_hast >= p_sag
    en_yuksek = max(p_hast, p_sag)
    emin = en_yuksek >= GUVEN_ESIGI

    if not emin:
        st.warning(
            f"Emin değilim (en yüksek olasılık %{100 * en_yuksek:.0f}). "
            "Daha net bir fotoğraf çekin: yaprak tek başına, iyi ışıkta, odakta ve sade bir zeminde olsun.")
    for u in r.uyarilar:
        st.warning(u)

    sol, sag = st.columns(2)
    with sol:
        st.subheader("Tahmin")
        if not emin:
            st.markdown(f"## Belirsiz (en olası: {'hastalık belirtisi var' if hasta else 'sağlıklı görünüyor'})")
        elif hasta:
            st.markdown("## :red[Hastalık belirtisi var]")
        else:
            st.markdown("## :green[Sağlıklı görünüyor]")
        st.write(f"Hastalık olasılığı: **%{100 * p_hast:.0f}** · Sağlıklı olasılığı: **%{100 * p_sag:.0f}**")
        st.progress(min(max(p_hast, 0.0), 1.0))
        if hasta:
            tur = max(("adi_yaprak_yanikligi", "hale_yanikligi"), key=lambda k: ol[k])
            st.caption(f"Olası tür: {AD_TR[tur]} (%{100 * ol[tur]:.0f})")
            st.caption(TUR_NOTU)
    with sag:
        st.subheader("Hastalık şiddeti")
        renk = DUZEY_RENK[r.duzey]
        st.markdown(f"### :{renk}[{r.duzey.capitalize()}]")
        st.write(f"Lezyon (yeşil olmayan doku) oranı: **%{100 * r.lezyon_orani:.1f}**")
        st.caption("Yaklaşık, renk tabanlı bir ölçüdür: < %0,5 sağlıklı, < %13 az hastalıklı, "
                   "≥ %13 çok hastalıklı. Uzman değerlendirmesi yerine geçmez.")
        if hasta and r.duzey == "sağlıklı":
            st.info("Model hastalık görüyor ancak belirgin renk lezyonu bulunamadı; erken evre olabilir.")
        if not hasta and r.duzey != "sağlıklı":
            st.info("Model sağlıklı diyor ama renk kuralı yaprakta yeşil olmayan bölgeler buldu "
                    f"({r.duzey}; kuruma, sararma, gölge veya alt yüz olabilir). Fotoğrafı kontrol edin.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(r.orijinal, caption="Yüklenen fotoğraf", use_container_width=True)
    with c2:
        st.image(cv2.resize(r.yaprak, (384, 384), interpolation=cv2.INTER_CUBIC),
                 caption=f"Ayrılan yaprak ({r.maske_yontemi})", use_container_width=True)
    with c3:
        st.image(cv2.resize(r.lezyon_katmani, (384, 384), interpolation=cv2.INTER_NEAREST),
                 caption="Lezyon bölgeleri (mor)", use_container_width=True)

    st.subheader("Öneriler")
    if hasta and not emin:
        st.write("Tahmin kesin olmadığı için aşağıdakiler genel bakteriyel hastalık önlemleridir; "
                 "önce daha net bir fotoğrafla yeniden deneyin veya bir uzmana gösterin.")
    for o in oneri_listesi("hastalikli" if hasta else "saglikli"):
        st.markdown(f"- {o}")
    if hasta and r.duzey == "çok hastalıklı":
        st.markdown("- Hastalık yaygın görünüyor: bu bitkileri ayırıp en kısa sürede ziraat mühendisine gösterin.")
    st.caption(f"Kullanılan model: {r.model_adi}")


def main():
    st.set_page_config(page_title="Fasulye Yaprağı Hastalık Denetimi", page_icon="🌿", layout="wide")
    st.title("Fasulye Yaprağı Hastalık Denetimi")
    st.write("Bir fasulye yaprağının fotoğrafını yükleyin; hastalık belirtisi, şiddeti ve genel öneriler gösterilir.")
    _hazirla()
    if not ma.model_var():
        st.error("Model dosyası bulunamadı (ciktilar/genel_model/model.pt). Önce modeli eğitin.")
        return
    dosya = st.file_uploader("Yaprak fotoğrafı (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])
    if dosya is not None:
        with st.spinner("Yaprak ayrılıyor ve analiz ediliyor..."):
            try:
                r = ma.analiz_et(dosya)
            except Exception as e:
                st.error(f"Fotoğraf işlenemedi: {e}")
                return
        goster(r)
    st.divider()
    st.warning("Bu bir karar destek aracıdır, kesin teşhis için ziraat mühendisine danışın.")
    st.caption(f"Model: {ma.model_adi()} · Yalnızca 3 sınıfı tanır (CBB, hale yanıklığı, sağlıklı); "
               "başka hastalık, zararlı veya besin eksikliği tanınmaz.")


main()
