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

| Sınıf klasörü | Kaynak A (ykvcrjffzd) | Kaynak B (3km9d246z2) | Kaynak C (Embrapa) | **Toplam** | Grup sayısı: yalnız dHash* | Grup sayısı: dHash + seri* |
|---|---|---|---|---|---|---|
| `adi_yaprak_yanikligi` | 508 (`Bean/Blight`) | – | 25 | **533** | 532 | 78 |
| `hale_yanikligi` | – | 231 (`halo_blight`) | – | **231** | 146 | 146 |
| `saglikli` | 549 (`Bean/Fresh Leaf`) | 1145 (`healthy_leaves`) | 2 | **1696** | 936 | 444 |

`data\raw_ek\` (kaynak B'den gelen, işe yarayabilecek ek sınıflar; hedef sınıflara karıştırılmadı):

| Klasör | Orijinal sınıf | Görüntü | Yakın-kopya grubu* |
|---|---|---|---|
| `kose_leke_als` | angular_leaf_spot | 270 | 52 |
| `fungal_patojenler` | fungi_pathogens | 258 | 60 |
| `tutun_tirtili_zarari` | tobacco_caterpillar | 266 | 51 |
| `sari_mozaik` | yellow_mosaic | 240 | 89 |
| `cercospora_yaprak_lekesi` | cercospora_leaf_spot | 86 | 18 |

\* Bkz. §4: aynı yaprağın neredeyse aynı çekimleri tek "grup" sayıldı. Kaynak B'deki görüntülerin önemli kısmı
aynı yaprağın çok benzer kareleridir; **gerçek bağımsız örnek sayısı grup sayısına yakındır.** Tablodaki
"yalnız dHash" sütunu kaynak A'da bağımsız yaprak sayısını çok abartır: A'nın seri çekim kareleri dHash eşiğini
çoğu zaman aştığı için neredeyse her kare ayrı grup sayılır (A'da 1 057 görüntü, 1 056 dHash grubu). dHash
gruplarını seri çekim penceresiyle (§4) birleştirince A'da 110 grup kalıyor (CBB 53, sağlıklı 57; 100'ü 10
karelik seri); `raw` içindeki CBB 532 → 78, sağlıklı 936 → 444 gruba iniyor. `data\etiketler.csv`'deki `grup`
sütunu ikinci (dHash + seri) gruplamadır; `raw` içinde toplam 668 grup. `raw_ek` sınıflarında (kaynak B, zaman
damgası yok) iki gruplama aynıdır.

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
- **Sonradan bulunan eksik: seri çekim kareleri.** dHash tek başına yetmedi. Kaynak A'nın dosya adları kamera
  zaman damgası taşıyor (`bdcp__IMG_20240105_235312_641.jpg` = tarih, saat-dakika-saniye, milisaniye) ve
  yapraklar çoğunlukla 10'ar karelik seriler hâlinde çekilmiş; aynı serinin kareleri çoğu zaman Hamming ≤ 20
  eşiğini aşıyor ve ayrı gruplara düştü. Aynı sınıf içinde zamana göre sıralanınca ardışık iki kare arasındaki
  fark ya 0–1 sn ya da en az 10 sn; arada değer yok, bu yüzden 2 sn'lik pencere serileri ayırmak için güvenli. İlk
  bölme yalnızca dHash gruplarıyla yapıldığı için ortak grup sayısı 0 olmasına rağmen sızıntılıydı:
  `scripts/bolme_denetimi.py` 145 çekim saniyesini hem eğitimde hem doğrulamada buldu; kaynak A'nın 314 doğrulama
  görüntüsünden 299'unun (o 738'lik doğrulama kümesinin %40,5'i) eğitimde aynı saniyeden bir eşi vardı.
  `yakin_kopya.py` artık aynı kaynak + sınıfta zaman damgaları arasında en çok 2 sn olan
  kareleri de (zincirleme) aynı gruba birleştiriyor; zaman damgası olmayan adlarda (B: `Good_Leaf_671.png`,
  C: `DSC_0629.jpg`; bu sayılar sıra numarasıdır, çekim zamanı değil) yalnızca dHash kullanılıyor. C'deki iki
  sağlıklı karenin elle birleştirilmesi de artık betikte (`ELLE_BIRLESTIR`). Depodaki `etiketler.csv` bu
  gruplamayla yeniden üretildi (seri bağlantısı 946, grup sayısı tüm 3 580 dosyada 1 884 → 938) ve
  `bolme_*.csv` bu gruplarla yeniden kuruldu: 1 716 eğitim / 735 doğrulama; denetim ortak grup, dosya, çekim
  saniyesi ya da seri bulmuyor. Tüm modeller yeni bölmede yeniden eğitildi (sonuçlar: `RESULTS.tr.md`).

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
2. **Hale yanıklığı etiket güvenilirliği düşük** (§5). Ayrıca bağımsız örnek sayısı az: 231 görüntü ama 146
   yakın-kopya grubu.
3. **Sınıf dengesizliği:** saglikli 1696 / CBB 533 / hale 231. Sınıf ağırlığı ya da örnekleme gerekir; sağlıklı
   sınıfı alt-örneklemek de bir seçenek.
4. **Yakın kopyalar ve seri çekimler:** dHash yakın kopyaları kaynak B'de yoğun; kaynak A'da ise aynı yaprağın
   çoğunlukla 10'ar karelik seri çekimleri var ve dHash bunları yakalamıyor (§4). Bölme mutlaka seri çekime duyarlı
   `grup` sütununa göre yapılmalı ve `scripts/bolme_denetimi.py` ile denetlenmeli.
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
  kopya ayıklama, `data\raw`, `data\raw_ek`, `etiketler.csv`), `yakin_kopya.py` (dHash + seri çekim grupları), `bolme_denetimi.py` (eğitim/doğrulama sızıntı denetimi),
  `temas_sayfasi.py` (görsel kontrol sayfaları). İndirilen arşivlerdeki hiçbir betik çalıştırılmadı; arşiv içi
  yollar dosya yazarken kullanılmadı.

## 9. Ek CBB kaynağı E: Bugwood / IPM Images (2026-11-15)

| | |
|---|---|
| Kaynak | **Bugwood Center for Invasive Species and Ecosystem Health — IPM Images** (University of Georgia), konu 11171 "Common bacterial blight of beans" (*Xanthomonas axonopodis* pv. *phaseoli*), 77 görüntü |
| URL | https://www.ipmimages.org/ (görüntü: `https://bugwoodcloud.org/images/1536x1024/<numara>.jpg`; metadata: `https://api.bugwood.org/rest/api/image.json?imgnum=<numara>&include=descriptor,description`) |
| Lisans | Kullanım şartları (https://www.ipmimages.org/about/image-usage): görüntüler **CC BY 3.0 US** ya da **CC BY-NC 3.0 US**; eğitim amaçlı kullanım atıfla serbest. Görüntü başına lisans alanı API'de yok ve görüntü detay sayfası otomatik istemcilere 403 verdiği için (zorlanmadı) lisans sütununa site koşulu yazıldı. Bu proje ticari değildir, iki lisansa da uyar. |
| Atıf biçimi | "Fotoğrafçı, Kurum, Bugwood.org" — her görüntünün atfı `data/bugwood_metadata.csv` → `atif` sütununda. Alınan 15 görüntünün atıfları: Howard F. Schwartz, Colorado State University, Bugwood.org (13); Agriculture Canada, Ottawa, Agriculture and Agri-Food Canada, Bugwood.org (1: 0454012); R.W. Samson, Purdue University, Bugwood.org (1: 5402949). |
| Depoda | Yalnız `scripts/indir_bugwood.py`, `data/bugwood_metadata.csv` (numara, fotoğrafçı, kurum, lisans, konak, açıklama, url, atıf, SHA-256) ve `data/bugwood_ayiklama.csv`. Görüntüler (`data/raw_bugwood/`, `data/islenmis_bugwood/`) **depoya konmaz** (`.gitignore`). |

İndirme: `python -I scripts/indir_bugwood.py <proje_koku>`; istekler arasında 1 sn beklenir, varsayılan
User-Agent değiştirilmez, yalnız bağlantı hatasında (HTTP yanıtı yokken) en çok 3 deneme yapılır; 403/404 zorlanmaz.
77/77 görüntü indi (1536 px'lik sürüm, çoğu ~1536×1024), SHA-256'ları kaydedildi.

**Ayıklama (`data/bugwood_ayiklama.csv`, her görüntüye temas sayfasında ve gerektiğinde büyük boyutta bakılarak):**
yalnız **fasulye (*P. vulgaris*) yaprağında görünür CBB belirtisi** olan kareler alındı.

| Karar | Görüntü | Neden |
|---|---|---|
| Alındı | **15** (kırpmalarla **20 örnek**) | Yakın çekim yaprak/yaprakçık; sarı kenarlı kahverengi nekroz, kenar nekrozu, su emmiş lezyon |
| Çıkarıldı | 52 | Tarla genel görünümü: 1999 Burlington (Colorado) deneme parselleri (45 kare), kanopi/ufuk kareleri (7) |
| Çıkarıldı | 3 + 1 | Bakla (1634010, 0454014, 5363933), tohum (0656088) |
| Çıkarıldı | 1 | Konak fasulye değil: cowpea (5361117, açıklamada yazıyor) |
| Çıkarıldı | 1 | Konak belirtilmemiş, yapraklar üst üste (5411466) |
| Çıkarıldı | 1 | Karışık enfeksiyon: CBB + bakteriyel kahverengi leke (5365235) |
| Çıkarıldı | 1 | Kanopi + bakla, tek yaprak sınırı yok (0454013) |

- **Kırpma:** 8 görüntüde elle kırpma yapıldı (koordinatlar CSV'de, orijinal piksel); 5 görüntüden ikişer yaprak
  alındı (aynı grupta). Bugwood'un sağ alttaki numara etiketi (siyah kutu) bir kaynak imzası olduğu için her karenin
  **alt %7'si kesildi**.
- **Gruplama:** aynı fotoğrafçının ardışık numaralı serileri (ör. 5361436–5361439, 5363005–5363007) aynı yaprak/
  sahne olabileceği için elle aynı `seri`ye alındı, ayrıca dHash (≤ 20) bağlantısı → **8 grup** (2, 3, 5, 4, 1, 3, 1, 1 örnek).
- **Kopya denetimi:** A/B/C/D'nin hiçbir görüntüsüyle yakın kopya yok (dHash ≤ 20: 0). Embrapa (C) sızıntısı yok.
- **Ön işleme** (`src/11_bugwood_veri.py`): `genel_maske.on_isle(otomatik)`; 6'sı "siyah", 14'ü "genel" zemin
  yöntemiyle maskelendi, başarısız 0. Gözle kontrol (`ciktilar/genel_model/bugwood_maske_kontrol.jpg`): çoğu makul;
  ancak yaprağı kadrajı dolduran yakın çekimlerde maske kadraj kenarıyla kesik bir yaprak parçası veriyor
  (5361436, 5363930, 5357503_b; sonuncusunda lezyonun bir kısmı maskenin dışında kaldı), 5402949'da iki yaprak birlikte.
- **E küçük:** 20 örnek / 8 grup. Konular tek bir hastalık uzmanının (H. F. Schwartz) arşivinde yoğunlaşıyor (20'nin 17'si).
  Bu, CBB için ikinci bir *bağımsız* kaynak sayılabilir (A: Bangladeş, koparılmış yaprak, siyah kumaş; E: ABD/Kanada,
  çoğu tarlada yaprak üzerinde), ama istatistiksel gücü çok sınırlıdır.

## 10. İkinci gerçek CBB kaynağı F: Selçuk Üniversitesi "Bean Bacterial Blight Disease Dataset" (2026-11-16)

| | |
|---|---|
| Kaynak | **Bean Bacterial Blight Disease Dataset** — Selçuk Üniversitesi (Murat Koklu veri setleri). Türkiye'de tarlada çekilmiş fasulye (*P. vulgaris*) fotoğrafları; iki çeşit (**Akbulut**, **Üstün42**), sınıflar: *Xanthomonas axonopodis* pv. *phaseoli* (CBB) ve sağlıklı. |
| Makale / atıf (zorunlu) | Kursun R., Gur A., Bastas K. K., Koklu M. (2024). *Agronomy* 14(7):1495. doi:10.3390/agronomy14071495 |
| URL | https://www.muratkoklu.com/datasets/Bean_Bacterial_Blight_Disease_Dataset.zip (kullanıcı yazarla yazışıp kendisi indirdi; bu projede ağdan indirme yapılmadı) |
| Lisans | Sayfada açık lisans yazmıyor; **atıf zorunlu**. Bu yüzden **görüntüler yeniden dağıtılmaz**: `data/raw_selcuk/` ve `data/islenmis_selcuk/` `.gitignore`'da. Depoda yalnız `src/12_selcuk_veri.py` ve `data/selcuk_metadata.csv` (dosya, orijinal ad, çeşit, sınıf, SHA-256, genişlik, yükseklik, bayt) var. |
| Ham içerik | 4 klasör × 1000 JPEG = 4000. Boyutlar: CBB'lerin tamamı 500×243; sağlıklılarda 500×281 (1069), 500×243 (834), 4608×2128 (97). Boyut dağılımı sınıfa göre farklı olduğu için tüm görüntüler ön işlemeden önce uzun kenarı 500 px'e indirildi (çözünürlük kısayolu olmasın). |

**Kopya / artırma denetimi** (`src/12_selcuk_veri.py`; sonuç `data/islenmis_selcuk/selcuk_ozeti.json`):
- **Birebir kopya (SHA-256): 351.** `Ustun42_Healthy (1)`–`(351)` dosyaları `Akbulut_Healthy (1)`–`(351)` ile bayt bayt aynı.
  Bunlara ek olarak `Ustun42_Healthy (30)`, `Akbulut_Healthy (30)`'un döndürülmüş kopyası (8 yönlü dHash uzaklığı 0).
  Toplam 352 kopya atıldı (ilk görülen = Akbulut tutuldu). **Bu 352 görüntünün çeşit etiketi güvenilir değil**
  (hangi çeşide ait olduğu bilinemiyor); Akbulut sağlıklı olarak tutuldu.
- **8 yönlü dHash** (4 döndürme × ayna; 256 bit; eşik ≤ 20): klasör içinde döndürülmüş/çevrilmiş/kırpılmış artırma
  kopyası **bulunmadı** (yakın çift 1: yukarıdaki çeşitler arası kopya). En yakın komşu uzaklığı ≤ 30 olan yalnız 4,
  ≤ 50 olan 8 görüntü var (ardışık numaralı, aynı sahnenin ikinci karesi). Yani F'de artırma ile şişirme yok;
  **3648 benzersiz fotoğraf** (CBB Akbulut 1000, CBB Üstün42 1000, sağlıklı Akbulut 1000, sağlıklı Üstün42 648).
- **Seri gruplaması:** Gözle bakıldığında dosyalar çekim sırasıyla numaralı ve ardışık kareler çoğu zaman aynı
  bitki/yaprağın biraz farklı açıdan çekimi (ör. CBB 0390/0391, sağlıklı 0766/0767). Bu, dHash'in yakalamadığı
  bir sızıntı yolu. Kural: aynı klasörde aynı 10'luk numara bloğu = aynı seri; blok sınırında ardışık (fark ≤ 2) iki
  kare, ImageNet EfficientNet-B0 gömmelerinin kosinüs benzerliği ≥ 0,85 ise birleştirilir (eşik, örnek çiftlere
  gözle bakılarak seçildi). Sonuç: **274 grup** (CBB Akbulut 87, CBB Üstün42 95, sağlıklı Akbulut 56, sağlıklı Üstün42 36;
  medyan 10, en büyük 70 görüntü). Bölme bu gruplarla yapılır. Bu gruplama ihtiyatlıdır ama kusursuz değildir:
  farklı bloklarda aynı bitkinin tekrar çekilmiş olması dışlanamaz.
- **Kaynaklar arası:** F'nin 8 yönünden herhangi biri ile A (1057), B (1376), C (27), D (371), E (20) görüntüleri
  arasında dHash ≤ 20 olan çift **0** (en küçük uzaklıklar: A 77, B 67, C 82, D 77, E 81). C (Embrapa) görüntüleri
  bu denetimde **yalnız dHash için** okundu (etiket/model çıktısı yok) — §9 ve RESULTS §12.2'deki istisnayla aynı.
  (Not: Bu denetim sırasında `src/11_bugwood_veri.py`'deki dHash'in bit sırasının `scripts/yakin_kopya.py`'nin tersi
  olduğu fark edildi, yani E'nin A/B/C/D'ye karşı denetimi geçersizdi. Düzeltildi ve yeniden ölçüldü: E'nin A/B/C/D'ye
  en küçük uzaklığı 78/79/85/88 — sonuç (yakın kopya 0) değişmedi.)

**Ön işleme ve gözle kontrol:** `genel_maske.on_isle(otomatik)` (uygulamadaki `tahmin_et` ile aynı yol).
3648/3648 maskelendi (3644 "genel", 4 "siyah" zemin; maske alanı medyanı %36). Kontrol sayfası
`ciktilar/genel_model/selcuk_maske_kontrol.jpg` (48 rastgele örnek): çoğunda merkezdeki yaprak ya da birbirine değen
birkaç yaprak alınıyor, toprak ve gökyüzü atılıyor. Hatalar: bazılarında toprak parçası maskede kalıyor (ör. Üstün42 CBB
0016, sağlıklı 0711), bazılarında lezyonlu yaprak yerine yandaki sağlıklı yaprak ya da yaprak kümesi alınıyor.
Alternatifler (merkez kare kırpma + maskesiz) diğer kaynaklarla (zemin siyah) tutarsız olur ve uygulamada `tahmin_et`
yolu değiştirilmeden kullanılamazdı; bu yüzden seçilmedi. Örnek şekil: `ciktilar/sekiller/rapor/sekil_selcuk_ornekleri.png`.

**Bilinen sınırlar:** (1) Sağlıklı ve CBB fotoğrafları farklı oturumlarda çekilmiş görünüyor (sağlıklılar daha mavi
tonlu, farklı boyutlarda) — F içinde de sınıf ile çekim koşulu kısmen karışık. (2) Tek ülke, tek ekip; iki çeşit aynı
tarla/kamera ile çekildi, bu yüzden "çeşit dışı" vekil test gerçek kaynak dışı test değildir. (3) Kadrajda çok sayıda
yaprak var; etiket fotoğraf düzeyindedir (CBB fotoğrafında maskelenen yaprak sağlıklı olabilir).
