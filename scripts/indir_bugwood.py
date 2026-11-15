"""Ek (besinci) kaynak E: Bugwood / IPM Images -> data/raw_bugwood/ (git disi) + data/bugwood_metadata.csv

Konu: 11171 "Common bacterial blight of beans" (Xanthomonas axonopodis pv. phaseoli), 77 goruntu.
Kullanim sartlari: https://www.ipmimages.org/about/image-usage -- goruntuler CC BY 3.0 US ya da
CC BY-NC 3.0 US lisanslidir; egitim amacli kullanim atifla serbesttir. Atif bicimi:
"Fotografci, Kurum, Bugwood.org". Goruntuler depoya KONMAZ; yalniz bu betik ve metadata CSV depoya girer.

- Goruntu: https://bugwoodcloud.org/images/1536x1024/<numara>.jpg
- Metadata: Bugwood API https://api.bugwood.org/rest/api/image.json?imgnum=<n>&include=descriptor,description
  (fotografci, kurum, aciklama). Goruntu basina lisans alani API'de yok, detay sayfasi (ipmimages.org/browse/
  detail.cfm) otomatik istemcilere 403 veriyor -> lisans sutunu sitenin genel kosulu olarak yazilir.
- Konak: aciklama metninden anahtar kelimeyle kestirilir (ornegin "dry bean", "snap bean", "cowpea"); kesin
  karar gorsel ayiklamada verilir (data/bugwood_ayiklama.csv).
- Istekler arasinda ~1 sn beklenir, varsayilan User-Agent (Python-urllib) degistirilmez; 403 gelirse zorlanmaz,
  kaydedilip gecilir.
Kullanim: python -I scripts/indir_bugwood.py <proje_koku>
"""
import csv
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

NUMARALAR = [
    "1634010", "5357506", "5357504", "5365235", "5363936", "5363934", "5363932", "5363930", "5363580", "5363006",
    "5361440", "5361438", "5361436", "5359946", "5359944", "5359942", "5359940", "5359938", "5359936", "5359934",
    "5359932", "5359930", "5359928", "5359926", "5359924", "5359922", "5359920", "5359918", "5359916", "5359914",
    "5359912", "5359910", "5359908", "5359906", "5359904", "0656088", "0454014", "5411466", "5402949", "5357507",
    "5357505", "5357503", "5365234", "5363935", "5363933", "5363931", "5363581", "5363007", "5363005", "5361439",
    "5361437", "5359947", "5359945", "5359943", "5359941", "5359939", "5359937", "5359935", "5359933", "5359931",
    "5359929", "5359927", "5359925", "5359923", "5359921", "5359919", "5359917", "5359915", "5359913", "5359911",
    "5359909", "5359907", "5359905", "5359903", "0454013", "0454012", "5361117",
]
GORUNTU_URL = "https://bugwoodcloud.org/images/1536x1024/{n}.jpg"
API_URL = "https://api.bugwood.org/rest/api/image.json?imgnum={n}&include=descriptor,description"
DETAY_URL = "https://www.ipmimages.org/browse/detail.cfm?imgnum={n}"
LISANS = "CC BY 3.0 US veya CC BY-NC 3.0 US (ipmimages.org/about/image-usage; goruntu basina alan API'de yok)"
BEKLE = 1.0
MAKS_BOYUT = 20 * 1024 * 1024
ALANLAR = ["numara", "fotografci", "kurum", "lisans", "konak", "aciklama", "tur", "konu", "url", "detay_url",
           "atif", "sha256", "bayt", "durum"]

# (desen, konak adi) -- ilk eslesen kazanir; ozel adlar genel "bean"den once
KONAK_DESENLERI = [
    (r"cowpea|vigna", "cowpea (Vigna)"),
    (r"lima bean|phaseolus lunatus", "lima fasulyesi"),
    (r"soybean|glycine", "soya"),
    (r"tepary", "tepary fasulyesi"),
    (r"dry bean|pinto|navy bean|great northern|kidney bean|black bean", "fasulye (dry bean, P. vulgaris)"),
    (r"snap bean|green bean", "fasulye (snap bean, P. vulgaris)"),
    (r"phaseolus vulgaris|common bean|\bbeans?\b", "fasulye (P. vulgaris)"),
]


def gecerli_numara(n):
    if not re.fullmatch(r"\d{6,8}", n):
        raise ValueError(f"gecersiz goruntu numarasi: {n!r}")
    return n


def aciklama_temizle(h):
    """API'deki HTML aciklamayi duz metne cevirir."""
    if not h:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h))).strip()


def konak_cikar(metin):
    m = (metin or "").lower()
    for desen, ad in KONAK_DESENLERI:
        if re.search(desen, m):
            return ad
    return "belirtilmemis"


def meta_satiri(n, api):
    """API yaniti (dict) -> metadata satiri (sha256/bayt/durum haric)."""
    r = (api or {}).get("rows") or [{}]
    r = r[0]
    fot, kur = (r.get("photographer") or "").strip(), (r.get("organization") or "").strip()
    ac = aciklama_temizle(r.get("description"))
    return {"numara": n, "fotografci": fot, "kurum": kur, "lisans": LISANS, "konak": konak_cikar(ac),
            "aciklama": ac, "tur": (r.get("scientificName") or "").strip(),
            "konu": re.sub(r"\s+", " ", r.get("sub_name") or "").strip(),
            "url": GORUNTU_URL.format(n=n), "detay_url": DETAY_URL.format(n=n),
            "atif": ", ".join(x for x in (fot, kur, "Bugwood.org") if x)}


def getir(url, zaman_asimi=60):
    """(durum_kodu, bayt). Varsayilan User-Agent; hata kodlari istisna yerine dondurulur."""
    try:
        with urllib.request.urlopen(url, timeout=zaman_asimi) as y:
            veri = y.read(MAKS_BOYUT + 1)
            if len(veri) > MAKS_BOYUT:
                return -1, b""
            return y.status, veri
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, b""


def getir_tekrarli(getir_f, url, deneme=3, bekle=BEKLE):
    """Yalniz baglanti hatasinda (kod 0, zaman asimi) tekrar dener; 403/404 gibi HTTP yanitlari zorlanmaz."""
    for i in range(deneme):
        kod, veri = getir_f(url)
        if kod != 0:
            return kod, veri
        time.sleep(bekle * (i + 2))
    return kod, veri


def jpeg_mi(veri):
    return veri[:3] == b"\xff\xd8\xff"


def main(kok, getir_f=getir, bekle=BEKLE):
    hedef = os.path.join(kok, "data", "raw_bugwood")
    os.makedirs(hedef, exist_ok=True)
    satirlar = []
    for n in NUMARALAR:
        n = gecerli_numara(n)
        kod, ham = getir_tekrarli(getir_f, API_URL.format(n=n), bekle=bekle)
        try:
            api = json.loads(ham.decode("utf-8")) if kod == 200 else {}
        except ValueError:
            api = {}
        time.sleep(bekle)
        satir = meta_satiri(n, api)
        yol = os.path.join(hedef, f"{n}.jpg")
        if os.path.exists(yol):
            with open(yol, "rb") as f:
                veri = f.read()
            kod = 200
        else:
            kod, veri = getir_tekrarli(getir_f, GORUNTU_URL.format(n=n), bekle=bekle)
            time.sleep(bekle)
            if kod == 200 and jpeg_mi(veri):
                with open(yol, "wb") as f:
                    f.write(veri)
        if kod == 200 and jpeg_mi(veri):
            satir.update(sha256=hashlib.sha256(veri).hexdigest(), bayt=len(veri), durum="ok")
        else:
            satir.update(sha256="", bayt=0, durum=f"hata_{kod}")  # 403 vb. zorlanmaz
        satirlar.append(satir)
        print(n, satir["durum"], satir["atif"], "|", satir["konak"], flush=True)
    with open(os.path.join(kok, "data", "bugwood_metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ALANLAR)
        w.writeheader()
        w.writerows(satirlar)
    print("ok:", sum(s["durum"] == "ok" for s in satirlar), "/", len(satirlar))
    return satirlar


if __name__ == "__main__":
    main(sys.argv[1])
