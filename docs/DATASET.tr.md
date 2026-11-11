> Not: Bu belge proje sırasında yazılmış Türkçe çalışma notudur. Anılan bazı çıktı dosyaları (loglar, ara CSV'ler, model ağırlığı, görüntü verisi) depoya dahil edilmemiştir; betiklerle yeniden üretilir. Ana özet için [README.md](../README.md).

# Veri Seti Raporu — Fasulye Yaprağı: Adi Yaprak Yanıklığı / Hale Yanıklığı / Sağlıklı

Hazırlanma tarihi: 2026-11-11. Tüm indirmeler 2026-11-11 tarihinde yapıldı.

> **Önce bunu okuyun (özet uyarı):** Üç hedef sınıf da açık ve kimliksiz indirilebilen kaynaklardan dolduruldu.
> Ancak **hale yanıklığı için bulunabilen tek kaynak** (Mendeley 3km9d246z2) Bangladeş kaynaklıdır. Görsel
> kontrolde bu sınıftaki yaprakların çoğu klasik hale yanıklığı belirtisi (küçük lekenin çevresinde belirgin
> sarı-yeşil hale) değil, **adi yanıklığa (CBB) benzeyen büyük kahverengi kenar nekrozu** gösteriyor. Ayrıca
> her sınıf ağırlıklı olarak **farklı bir kaynaktan ve farklı arka plandan** geliyor (kaynak = sınıf karışması).
> Ayrıntılar §6 ve §7'de.

## 1. Kaynaklar

| # | Kaynak | URL / DOI | Lisans | İndirilen dosya(lar) | Boyut |
|---|---|---|---|---|---|
| A | **Image Dataset for Bean and Cowpea Plant Leaf Disease Detection and Freshness Assessment from Bangladesh** (Mendeley Data, v1, 2024-08-06). Yazarlar: M. R. A. Rashid, M. Hasan, R. Gani, R. Kamara, T. K. Tarin, S. F. Rabbi. Makale: *Data in Brief* (2024), doi:10.1016/j.dib.2024.111023 (PMC11539515) | https://data.mendeley.com/datasets/ykvcrjffzd/1 — doi:10.17632/ykvcrjffzd.1 | CC BY 4.0 | Yalnızca `Bean/Blight` (510 dosya) ve `Bean/Fresh Leaf` (554 dosya), tek tek JPG (Mendeley public API; her dosyanın SHA-256'sı API'deki değerle doğrulandı) | ≈ 207 MB |
| B | **Bean Health and Disease Dataset: A Resource for Agricultural Research** (Mendeley Data, v3, 2025-11-20). Yazarlar: M. U. Mojumdar, S. D. Dip, T. E. Alahi, S. Tonmoy (Daffodil International University) | https://data.mendeley.com/datasets/3km9d246z2/3 — doi:10.17632/3km9d246z2.3 | CC BY 4.0 | `bean_leaves-20251022T043706Z-1-001.zip` (SHA-256 doğrulandı). Ek olarak karşılaştırma için `Bean Disease.zip` içindeki `Halo-Blight.zip` (HTTP Range ile yalnızca bu üye çekildi; aynı 250 görüntünün başka bir kopyası, **veri setine eklenmedi**) | 1 320 260 628 B (+118 772 155 B) |
| C | **Embrapa PDDB — Base de Imagens de Sintomas de Doenças de Plantas** (Redape Dataverse, 2024-07-30). Barbedo J. G. A. ve ark.; makale: Barbedo et al., *IEEE Latin America Transactions* 16(6), 2018, doi:10.1109/TLA.2018.8444395 | https://doi.org/10.48432/XA1OVL (https://www.redape.dados.embrapa.br/) | **CC BY-NC 4.0 (Embrapa)** — ticari kullanım yok; ders projesi için uygun | `Feijao (Dry bean) - Crestamento Bacteriano (Common bacterial blight) - 1.zip` (dosya id 5972), `Feijao (Dry bean) - Saudavel (Healthy) - 1.zip` (id 6015) | 45 895 912 B + 6 325 189 B |

İndirilen ham dosyalar: `data\_indirilen\<kaynak>\` (her kaynağın API meta verisi `_mendeley_metadata.json` /
`_dataverse_metadata.json` olarak yanında). Hiçbir kaynak hesap, anahtar, form ya da lisans onayı istemedi.
Toplam indirme ≈ 1.6 GB.

Atıf yükümlülüğü: A ve B için CC BY 4.0 gereği yukarıdaki yazar/DOI atfı rapor ve sunumda verilmelidir; C için
CC BY-NC 4.0 gereği atıf + ticari olmayan kullanım.

## 2. Sınıf → kaynak eşlemesi ve görüntü sayıları (temizlik sonrası)

`data\raw\` (hedef üç sınıf):

| Sınıf klasörü | Kaynak A (ykvcrjffzd) | Kaynak B (3km9d246z2) | Kaynak C (Embrapa) | **Toplam** | Yakın-kopya grubu sayısı* |
|---|---|---|---|---|---|
| `adi_yaprak_yanikligi` | 508 (`Bean/Blight`) | – | 25 | **533** | 532 |
| `hale_yanikligi` | – | 231 (`halo_blight`) | – | **231** | 146 |
| `saglikli` | 549 (`Bean/Fresh Leaf`) | 1145 (`healthy_leaves`) | 2 | **1696** | 936 |

`data\raw_ek\` (kaynak B'den gelen, işe yarayabilecek ek sınıflar; hedef sınıflara karıştırılmadı):

| Klasör | Orijinal sınıf | Görüntü | Yakın-kopya grubu* |
|---|---|---|---|
| `kose_leke_als` | angular_leaf_spot | 270 | 52 |
| `fungal_patojenler` | fungi_pathogens | 258 | 60 |
| `tutun_tirtili_zarari` | tobacco_caterpillar | 266 | 51 |
| `sari_mozaik` | yellow_mosaic | 240 | 89 |
| `cercospora_yaprak_lekesi` | cercospora_leaf_spot | 86 | 18 |

\* Bkz. §4: aynı yaprağın neredeyse aynı çekimleri tek "grup" sayıldı. Kaynak B'deki görüntülerin önemli kısmı
aynı yaprağın çok benzer kareleridir; **gerçek bağımsız örnek sayısı grup sayısına yakındır.**

**Şiddet (severity) etiketi:** Hiçbir kullanılabilir kaynakta yaprak düzeyinde şiddet etiketi bulunamadı.
`data\raw_siddet\` oluşturulmadı. (Hocanın "az hastalıklı" isteği için: hastalıklı sınıflarda hafif → ağır
arası doğal bir çeşitlilik var; ileride lezyon alanı oranıyla otomatik bir şiddet vekili hesaplanabilir, ama
bu bir uzman etiketi olmaz.)

## 3. Görüntü özellikleri

| Kaynak | Biçim | Çözünürlük | Arka plan / çekim |
|---|---|---|---|
| A | JPEG, RGB | çoğu 960×1280 veya 1280×960 (birkaçı ~870–980×1280) | Akıllı telefon; yaprak **siyah/koyu kumaş** üzerinde, kopartılmış tek yaprakçık; çok sayıda alt yüz çekimi |
| B | PNG, RGB (sağlıklıların 143'ü RGBA) | hepsi 768×960 | Arka planı **silinmiş (beyaz)**, tek yaprakçık, yakın çekim |
| C | JPEG, RGB | 3920×2204 (CBB), 4128×3096 (sağlıklı) | Laboratuvarda **renk/kalibrasyon kartı** (siyah-beyaz desen + renk paletleri) üzerinde; çoğu üç yaprakçıklı tam yaprak — *Phaseolus vulgaris* |

RGBA PNG'ler eğitimde `convert("RGB")` ile dönüştürülmelidir (şeffaf alanlar beyaz/siyah olabilir — tutarlı
olması için beyaz zemine yapıştırmak önerilir).

## 4. Yapılan temizlik

Betik: `scripts\duzenle.py` (rapor: `data\temizlik_raporu.json`).

- Aday dosya: 3 731 (A: 1 064, B: 2 640, C: 27).
- **Bozuk dosya: 0.** Her dosya PIL ile `verify()` + yeniden açıp `load()` ile tam çözüldü.
- **Birebir kopya (SHA-256) atılan: 151.** Hepsi aynı sınıf içinde: cercospora 74, saglikli 24 (B'den 19 + A'dan 5),
  sari_mozaik 20, hale_yanikligi 19, kose_leke_als 10, adi_yaprak_yanikligi 2, fungal 2.
- Sınıflar arası çakışan (aynı dosya iki farklı etikette) kopya: **0**.
- Kalan: 3 580 dosya (raw 2 460 + raw_ek 1 120). Dosya baytları değiştirilmedi (yeniden kodlama yok);
  yalnızca yeniden adlandırıldı: `<kaynak-öneki>__<orijinal-ad>` (`bdcp`=A, `bhd`=B, `pddb_cbb`/`pddb_saglikli`=C).
- **Yakın kopya taraması** (`scripts\yakin_kopya.py`, 256-bit dHash, Hamming ≤ 20): 5 679 çift, **hepsi aynı
  sınıf içinde**; bağlantılı bileşenler `grup` sütunu olarak `data\etiketler.csv`'ye yazıldı (çift listesi:
  `data\yakin_kopyalar.csv`). Örnek çiftleri açıp baktım: gerçekten aynı yaprağın hafif kaymış/yeniden
  çekilmiş kareleri. C'deki iki sağlıklı görüntü de aynı yaprağın iki karesi olduğu için elle aynı gruba alındı.
  **Bu dosyalar silinmedi** (veri az); bunun yerine eğitim/doğrulama/test bölmesi **`grup` sütununa göre
  yapılmalı** (ör. `GroupShuffleSplit` / `StratifiedGroupKFold`), aksi halde test başarımı şişer.

`data\etiketler.csv` sütunları: `dosya, sinif, kume(raw|raw_ek), kaynak, kaynak_yol, kaynak_url, lisans,
sha256, bicim, mod, genislik, yukseklik, grup`.

## 5. Etiket doğrulama (görüntüleri tek tek açarak)

Her sınıftan rastgele örnekleri (tohum 42 ve 7) numaralı temas sayfalarında açıp inceledim
(`scripts\temas_sayfasi.py`). İncelenen: adi_yaprak_yanikligi 8 (A) + 8 (C, ayrıca 3 tam çözünürlük),
hale_yanikligi 8 + 12 (+ ön incelemede 3 tam çözünürlük), saglikli 8 (A+B karışık) + 2 (C) + 2 tam çözünürlük.

**adi_yaprak_yanikligi**
- *Kaynak C (Embrapa, fitopatolog etiketli, P. vulgaris):* 8/8 fasulye yaprağı. DSC_0684, DSC_0701, DSC_0658'de
  ders kitabı CBB'si var: yaprak kenarından başlayan, **sarı kenarlı düzensiz kahverengi nekroz**. Ancak DSC_0663,
  DSC_0666, DSC_0660'ta baskın belirti **alt yüzde damarlar boyunca kırmızımsı-kahverengi damar nekrozu**; bu
  görünüm antraknozu da andırıyor. Uzman etiketi olduğu için korundu, ama kusursuz değil.
- *Kaynak A (Bangladeş):* 8/8 yaprak, 8/8 lekeli. Makale bu sınıfın etkenini açıkça *Xanthomonas axonopodis* pv.
  *phaseoli* olarak veriyor ve etiketlerin çiftçi + fitopatolog yardımıyla yapıldığını söylüyor. Görülen
  belirtiler: pembemsi-kırmızı/kahverengi, yuvarlakça-düzensiz lekeler, birçoğunda sarımsı kenar, yer yer delik.
  Sarı kenarlı nekrotik leke ile uyumlu, ama tipik "kenardan başlayan geniş V şekilli yanık" daha az. Lekelerin
  bir kısmı konsantrik/yuvarlak görünümüyle mantar lekelerine (ör. Cercospora/antraknoz) de benziyor.
  **Güven: orta.**
- Yaprak türü: A ve B'de tür (*Phaseolus vulgaris* mi, Bangladeş'te yaygın *Lablab purpureus* "sheem" mi)
  belirtilmemiş. Tek yaprakçık morfolojisi her ikisiyle de uyumlu; **P. vulgaris olduğu doğrulanamadı.**

**hale_yanikligi** (tamamı Kaynak B)
- 20/20 fasulye(-benzeri) yaprakçık, arka planı silinmiş.
- Klasik hale yanıklığına (küçük su emmiş/nekrotik nokta çevresinde belirgin sarı-yeşil hale) uyan: yalnızca
  birkaç tanesi (ör. Halo-Blight_115, _65, _36, _224, _58'de dağınık açık sarı-yeşil klorotik benekler, erken
  dönem hale ile uyumlu olabilir).
- Çoğunluğu (ör. _7, _84, _172, _167, _88, _5/_152, _69, _82, _248, _198): **büyük, kenardan ya da damar
  arasından başlayan kahverengi nekrotik alanlar, bazıları sarı kenarlı** → bu görünüm daha çok **adi yanıklık
  (CBB)** tanımına uyuyor.
- Birkaçında (_208, _244, _4) baskın hasar **böcek yeme delikleri** ve küçük kenar kuruması; hastalık belirtisi
  zayıf.
- **Değerlendirme: Bu sınıfın etiketleri güvenilir görünmüyor.** "Hale yanıklığı" adı yazarlarındır; ben
  gözlemsel olarak ancak ~%20–30'unda hale yanıklığıyla uyumlu belirti görebildim. Ayrıca veri seti hakkında
  makale/ yöntem açıklaması yok (patojen doğrulaması yapıldığına dair bilgi yok). Etiketleri değiştirmedim
  (kendi tahminimle yeniden etiketlemek de güvenilir olmazdı); ama bu sınıfla elde edilecek bir modelin
  "hale yanıklığını tanıdığı" iddiası bilimsel olarak zayıf olur.

**saglikli**
- 12/12 lekesiz, sağlıklı görünümlü yaprak (A: siyah kumaş, bir kısmı alt yüz; B: beyaz zemin; C: kalibrasyon
  kartı üzerinde üç yaprakçıklı tam yaprak). **Güven: yüksek.**

## 6. Bilinen sınırlılıklar (eğitimi yapacak kişi için önemli)

1. **Kaynak–sınıf karışması (shortcut riski, en kritik sorun).**
   - `hale_yanikligi` = %100 Kaynak B (beyaz zemin, PNG, 768×960).
   - `adi_yaprak_yanikligi` = %95 Kaynak A (siyah kumaş, JPEG, 960×1280) + %5 Kaynak C (kalibrasyon kartı).
   - `saglikli` = %32 A + %68 B + 2 C.
   Bir CNN hastalık yerine **arka plan/çözünürlük/biçim** öğrenerek yüksek doğruluk alabilir (ör. "beyaz zemin +
   leke → hale", "siyah kumaş + leke → CBB"). Öneriler: (a) tüm görüntüleri yaprağı ortalayıp aynı boyuta
   getirip **arka planı maskelemek/nötr renge çevirmek**; (b) sonuçları **kaynak bazında** raporlamak; (c)
   Embrapa CBB görüntülerini (farklı arka plan, gerçek *P. vulgaris*) ayrı bir dış test olarak kullanmak;
   (d) Grad-CAM ile modelin lekeye mi zemine mi baktığını göstermek.
2. **Hale yanıklığı etiket güvenilirliği düşük** (§5). Ayrıca bağımsız örnek sayısı az: 231 görüntü ama ~146
   yakın-kopya grubu.
3. **Sınıf dengesizliği:** saglikli 1696 / CBB 533 / hale 231. Sınıf ağırlığı ya da örnekleme gerekir; sağlıklı
   sınıfı alt-örneklemek de bir seçenek.
4. **Yakın kopyalar:** Kaynak B'de yoğun. Bölme mutlaka `grup` sütununa göre yapılmalı.
5. **Tür belirsizliği:** A ve B'nin "bean"i büyük olasılıkla Bangladeş fasulyesi; *P. vulgaris* olduğu
   kaynaklarda belirtilmemiş. Yalnızca C kesin olarak *P. vulgaris* (Feijão).
6. **Şiddet etiketi yok.**
7. Kaynak C yalnızca 25 CBB + 2 (aynı yaprağın) sağlıklı görüntü — az ama uzman etiketli.

## 7. Denenen ama kullanılamayan kaynaklar

| Kaynak | Neden kullanılmadı |
|---|---|
| ibean / Makerere "Beans" (Hugging Face `AI-Lab-Makerere/beans`, `Project-AgML/bean_disease_uganda`, Harvard Dataverse doi:10.7910/DVN/TCKVEW) | Yalnızca angular leaf spot, bean rust, healthy; CBB/hale yok. Sağlıklı sınıf zaten yeterli olduğu ve tarla fotoğrafları kaynak karışmasını daha da artıracağı için yedek olarak bile eklenmedi. |
| Kaynak B'nin `beans-...zip` arşivi (`bacterial_pathogen` 417, `no_disease` 420) | Klasör adı (`beans/`, yapraklar ayrı `bean_leaves/` arşivinde) ve veri seti açıklamasına ("bean plants and leaves") göre yaprak değil bitki/bakla görüntüleri; içeriği açılıp doğrulanmadı. Ayrıca bakteri türü belirtilmemiş. İndirilmedi. |
| HF `darcieg/bean-disease-uganda` | ibean kopyası; ayrıca anlamsız bir `win32rcparser` sınıf adı içeriyor — güvenilmez, atlandı. |
| HF `hjin-r/bean_disease` | ibean'in artırılmış (augmented) .npy sürümü; CBB/hale yok. |
| Embrapa PDDB "Mancha Aureolada (Halo Blight)" | **Turunçgil (Citrus)** hastalığı, fasulye değil — kullanılmadı. PDDB'de fasulye için hale yanıklığı sınıfı yok. |
| Zenodo 20544633 "Common Bean RGB Imagery Dataset" (CIAT, Kolombiya, CC BY 4.0) | İHA (drone) ile parsel düzeyinde görüntü; yaprak düzeyi değil; etiketler muamele (kontrol/CBB inokülasyonu), şiddet değil. Yaprak sınıflandırmasına uygun değil. |
| Zenodo 15685315 / 16751336 / 8286126 (Tanzanya) | Yalnızca healthy, rust, anthracnose. |
| Mendeley 89c9r5m2gx (Bangladeş, yüksek çözünürlüklü fasulye) | ALS, antraknoz, mozaik, pas, sağlıklı — CBB/hale yok. |
| Mendeley 6m8ms6ptd4, 3k46s46x3s (BLD-BD), fvvf72y3gt (BeanVision-3) | Sınıf listelerinde CBB/hale yok. |
| Scientific Reports 2024 YOLO çalışması (CIAT; ALS, CBB, mozaik, pas, antraknoz; ~9 500 görüntü) | Veri "talep üzerine yazarlardan" — açık değil. |
| Research Square rs-281525 ("bean halo blight" 820 görüntü içeren tablo) | İlgili çalışma geri çekilmiş (retracted) olarak işaretli; veri kaynağı belirsiz. |
| Kaggle (ör. `therealoise/bean-disease-dataset`) | Kimlik gerekiyor; kimliksiz aynası bulunamadı. |
| Wikimedia Commons | "Halo blight"/*phaseolicola*/CBB kategorilerinde fasulye yaprağı fotoğrafı yok. |
| Bugwood / IPM Images | Sunucu ve API 502/500 hatası döndürdü; otomatik erişim yapılamadı, zorlanmadı. |
| PMC (NCBI) makale sayfaları | reCAPTCHA doğrulaması istedi; aşılmaya çalışılmadı (makale metni Europe PMC REST API'den okundu). |

Yedek plana (kendi derlemesi) gerek kalmadı; bu yüzden `data\kaynaklar.csv` yok. Dosya başına kaynak/lisans
bilgisi `data\etiketler.csv` içinde.

## 8. Ortam ve teknik notlar

- Sanal ortam: `.venv\` (Python 3.13.7). Kurulu: `requests 2.34.2`, `pillow 13.0.0`, `huggingface_hub 2.4.0`
  (+ bağımlılıkları). Sistem Python'una paket kurulmadı.
- **Uzun yol notu (Windows):** Proje yolu çok uzunsa Windows bazı DLL yüklemelerinde ("Dosya adı veya uzantısı çok uzun")
  hata verebilir; bu durumda projeyi kısa bir yola taşıyın.
- Betikler (`scripts\`): `indir.py` (indirme + boyut/SHA-256 doğrulama), `uzak_zip_listele.py` /
  `uzak_zip_ornek.py` (büyük ZIP'leri indirmeden HTTP Range ile listeleme/örnek alma), `duzenle.py` (doğrulama,
  kopya ayıklama, `data\raw`, `data\raw_ek`, `etiketler.csv`), `yakin_kopya.py` (dHash grupları),
  `temas_sayfasi.py` (görsel kontrol sayfaları). İndirilen arşivlerdeki hiçbir betik çalıştırılmadı; arşiv içi
  yollar dosya yazarken kullanılmadı.
