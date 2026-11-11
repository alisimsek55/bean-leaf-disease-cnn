> Not: Bu belge proje sırasında yazılmış Türkçe çalışma notudur. Anılan bazı çıktı dosyaları (loglar, ara CSV'ler, model ağırlığı, görüntü verisi) depoya dahil edilmemiştir; betiklerle yeniden üretilir. Ana özet için [README.md](../README.md).

# SONUÇLAR — Fasulye Yaprağı Hastalık Sınıflandırması (Sıfırdan CNN)

Bu belge raporu yazacak kişi için **tek bilgi kaynağıdır**. Buradaki her sayı gerçekten çalıştırılan koddan
alınmıştır; kaynağı `ciktilar/metrikler.json` (ana kaynak), `ciktilar/siddet_ozeti.json`,
`ciktilar/bolme_ozeti.json`, `ciktilar/*/egitim_ozeti.json` ve `logs/` altındaki loglardır.
Tarih: 2026-11-11. Donanım: NVIDIA RTX 4070 Laptop GPU (8 GB), PyTorch 2.11.0+cu128 (CUDA çalıştı).

---

## 0. Tek paragrafta özet (önce bunu okuyun)

Üç sınıf (adi yaprak yanıklığı = CBB, hale yanıklığı, sağlıklı) için 16 katmanlı, 1 772 995 parametreli,
sıfırdan eğitilmiş bir CNN (FasulyeCNN) kurduk. Arka planı silip siyaha boyadık, yaprağa kırptık,
yakın-kopya gruplarına göre %70/%30 böldük. Doğrulama kümesinde (738 görüntü) **doğruluk %97,6, makro F1
%95,5** elde edildi. **Ancak bu sayı "model hastalığı tanıyor" diye okunmamalıdır.** Kontrol deneyleri şunu
gösterdi: (1) Maskelemeden sonra bile görüntünün hangi veri kaynağından geldiği aynı CNN ile **%100**
ayırt edilebiliyor; hale sınıfının tamamı tek kaynaktan, CBB'nin %95'i başka bir kaynaktan geldiği için
**CBB ↔ hale ayrımı kaynak ayrımıyla birebir örtüşüyor**. (2) Eğitimde hiç görülmeyen üçüncü bir kaynaktaki
(Embrapa, Brezilya, uzman etiketli) 25 CBB yaprağından yalnızca **5'i (%20)** doğru sınıflandı; 13'üne "hale",
7'sine "sağlıklı" dendi. (3) Yalnızca yaprak siluetiyle (renk/doku yok) eğitilen aynı model %52,3 doğrulukta
kaldı (çoğunluk sınıfı tabanı %69,0); yani yüksek başarı yaprak şeklinden gelmiyor. Sonuç: model **aynı
kaynak içinde hastalıklı ile sağlıklıyı** büyük ölçüde ayırabiliyor; fakat **CBB ile hale yanıklığını gerçekten
ayırdığına dair kanıt yok** ve başka kaynaktan gelen görüntülere genelleme zayıf.

---

## 1. Veri kaynakları ve atıflar

(Ayrıntı: `VERI_SETI.md`. Tüm indirmeler 2026-11-11.)

| Kod | Kaynak | Lisans | Bu çalışmada kullanılan |
|---|---|---|---|
| A | Rashid M. R. A., Hasan M., Gani R., Kamara R., Tarin T. K., Rabbi S. F. *Image Dataset for Bean and Cowpea Plant Leaf Disease Detection and Freshness Assessment from Bangladesh*. Mendeley Data v1 (2024), doi:10.17632/ykvcrjffzd.1; makale: *Data in Brief* (2024), doi:10.1016/j.dib.2024.111023 | CC BY 4.0 | `Bean/Blight` → CBB (508), `Bean/Fresh Leaf` → sağlıklı (549). Siyah/lacivert kumaş üzerinde telefon fotoğrafı, JPEG |
| B | Mojumdar M. U., Dip S. D., Alahi T. E., Tonmoy S. *Bean Health and Disease Dataset: A Resource for Agricultural Research*. Mendeley Data v3 (2025), doi:10.17632/3km9d246z2.3 | CC BY 4.0 | `halo_blight` → hale (231), `healthy_leaves` → sağlıklı (1145). Arka planı silinmiş (beyaz), PNG |
| C | Barbedo J. G. A. ve ark. *Embrapa PDDB — Base de Imagens de Sintomas de Doenças de Plantas*, doi:10.48432/XA1OVL; makale: *IEEE Latin America Transactions* 16(6), 2018, doi:10.1109/TLA.2018.8444395 | CC BY-NC 4.0 | CBB (25), sağlıklı (2, aynı yaprak). Renkli kalibrasyon kartı üzerinde, kesin *Phaseolus vulgaris*, fitopatolog etiketli |

Ham sayılar: CBB 533 (A 508 + C 25), hale 231 (B), sağlıklı 1696 (A 549 + B 1145 + C 2) = 2460.
`data/raw_ek/` içindeki ek sınıflar kullanılmadı. Etiketler değiştirilmedi (kaynak yazarlarının etiketi).

**Kaynak = sınıf karışması:** hale %100 B; CBB %95 A + %5 C; sağlıklı %32 A + %68 B (+2 C).

---

## 2. Ön işleme (betik: `src/01_on_isleme.py`)

Amaç: modelin zemini (beyaz PNG / siyah kumaş / kalibrasyon kartı) öğrenmesini engellemek.

Adımlar (sırayla):
1. **Okuma:** PIL ile; RGBA dosyalar (B'deki 143 sağlıklı) beyaz zemine yapıştırılıp RGB yapıldı. Uzun kenar
   1024 piksele küçültüldü (INTER_AREA).
2. **Yaprak maskesi — zemin türüne göre üç yöntem** (kaynak bilgisi `etiketler.csv`'den):
   - **B (beyaz zemin, 1376 görüntü):** HSV'de `S > 25` veya `V < 225` olan pikseller yaprak.
   - **A (koyu kumaş, 1057 görüntü):** GrabCut. Başlatma: V kanalında Otsu eşiği (40–120'ye kısılmış) →
     "olası yaprak"; bunların içinde Lab renk doygunluğu (chroma = √(a*²+b*²)) > 10 olanlar 7×7 aşındırılıp
     "kesin yaprak"; görüntü çerçevesi (%2) ve V < 0,5·eşik pikseller "kesin zemin". 512 pikselde 5 tur.
   - **C (kalibrasyon kartı, 27 görüntü):** yeşil-sarı ton çekirdeği (H 22–90, S > 60, V > 40) → en büyük
     bileşen → GrabCut (5 tur, maske ile başlatma); ardından renksiz-parlak kart pikselleri
     (chroma < 8 ve L > 90/255) çıkarıldı.
3. **Morfoloji:** açma + kapama (eliptik çekirdek; B: 5, A: 5, C: 7–9 piksel), **en büyük bağlı bileşen**.
4. **Delik doldurma:** A'da tüm delikler (deliklerden görünen kumaş zaten koyu), B ve C'de yalnızca yaprak
   alanının %1'inden küçük delikler (gerçek böcek delikleri beyaz/kart göstermesin, siyah kalsın).
5. **Arka plan siyaha** boyandı; yaprağın sınır kutusuna **%4 pay** ile kırpıldı, kare olacak şekilde siyahla
   dolduruldu, **256×256'ya** küçültüldü (görüntü INTER_AREA, maske en yakın komşu), yeniden boyutlandırma
   sonrası maske dışı tekrar sıfırlandı. Hepsi **PNG** olarak kaydedildi (`data/islenmis/goruntu/`,
   maskeler `data/islenmis/maske/`). Böylece biçim (JPEG/PNG), çözünürlük ve zemin farkı ortadan kalktı.
6. **Model girişi:** 256'lık görüntüler belleğe 128×128 (INTER_AREA) olarak yüklendi; eğitim kümesinin kanal
   ortalaması/standart sapmasıyla normalize edildi.

**Maske kalitesi denetimi (gözle):** Her (sınıf, kaynak) bileşiminden 32 rastgele görüntü ve kalite bayrağı
alan tüm görüntüler numaralı temas sayfalarında açılıp incelendi (`ciktilar/maske_kontrol/`).
- *1. deneme* (A için yalnız parlaklık/Otsu): A'daki bazı karelerde ışık alan **gri kumaş kıvrımları
  yaprak sanıldı** (yaprak yerine kumaş şeridi seçildi). *2. deneme* (yalnız renk doygunluğu): bu kez
  grimsi/soluk alt yüzlü yapraklar parçalandı. *3. (son) yöntem*: yukarıdaki GrabCut; 64 görüntülük deneme
  sayfasında yalnızca 2 hata kaldı (ikisi de kumaş).
- Son durumda otomatik bayraklar (2460 görüntü): küçük maske (<%5 alan; çoğu A'da kadrajda küçük duran
  ama doğru maskelenmiş yapraklar) 191, kenara değme 108 (çoğu B'de yaprağın kadrajı doldurması), düşük
  dışbükeylik (<0,80) 28, büyük maske 1, **renksiz maske (maske içi ort. chroma < 4) 8**. Toplam 334 bayraklı
  görüntü gözle incelendi.
- **Dışlanan: 9 görüntü** (hepsi A-sağlıklı): 8'i renksiz-maske kuralıyla (maske içi ortalama chroma
  1,03–3,10; gözle hepsi gri kumaş), 1'i elle (`data/islenmis_dislanan.csv`; kumaşa bitişik küçük yaprak,
  chroma 5,61). Chroma 4,9–6,4 aralığındaki diğer 7 görüntü gözle gerçek (grimsi) yaprak çıktığı için
  tutuldu. Dışlananlar diskte durur, `on_isleme_raporu.csv`'de `dislandi=True` ile işaretlidir.
- Kalan bilinen kusurlar: C (Embrapa) görüntülerinin bazılarında yaprakçıklar arasında küçük kart
  parçacıkları (renkli kareler) kaldı; birkaç A görüntüsünde yaprak sapı/kenarında hafif kumaş artığı var.

**Kaynak karışması gerçekten azaldı mı?** Zemin, biçim ve çözünürlük farkı kaldırıldı; ancak §6.2'deki
kontrol deneyi, kaynakların maskeli yapraklardan bile %100 ayırt edilebildiğini gösterdi (aydınlatma,
kamera rengi, A'da sık görülen alt yüz çekimi, B'nin arka plan silme işleminin bıraktığı kenar izi gibi
ipuçları). Yani maskeleme **zemini** kaldırdı, **kaynak imzasını** kaldırmadı.

---

## 3. Veri bölme (betik: `src/02_veri_bolme.py`)

- Oran: **%70 eğitim / %30 doğrulama** (hocanın örneğindeki gibi; ayrı test kümesi yok).
- **Grup bazlı:** `etiketler.csv`'deki `grup` (aynı yaprağın neredeyse aynı kareleri; dHash) iki tarafa
  bölünmedi. Denetim: ortak grup sayısı = 0.
- **Katmanlı:** her (sınıf, kaynak) katmanında gruplar tohum 42 ile karıştırılıp doğrulamaya, görüntü sayısı
  katmanın %30'una en yakın olacak şekilde eklendi.
- **Dengeleme veri atarak yapılmadı** (gerekçe §5); bu yüzden aşağıdaki sayılar ön işleme + dışlama
  sonrası gerçek sayılardır.

**Tablo 1 — Sınıf başına görüntü sayıları** (`ciktilar/bolme_tablosu.csv`)

| Sınıf | Eğitim | Doğrulama | Toplam |
|---|---|---|---|
| Adi yaprak yanıklığı (CBB) | 373 | 160 | 533 |
| Hale yanıklığı | 162 | 69 | 231 |
| Sağlıklı | 1178 | 509 | 1687 |
| **Toplam** | **1713** | **738** | **2451** |

**Tablo 2 — Kaynak kırılımı** (`ciktilar/bolme_tablosu_kaynak.csv`)

| Sınıf | Kaynak | Eğitim | Doğrulama |
|---|---|---|---|
| CBB | A (siyah kumaş) | 356 | 152 |
| CBB | C (Embrapa) | 17 | 8 |
| Hale | B (beyaz zemin) | 162 | 69 |
| Sağlıklı | A | 378 | 162 |
| Sağlıklı | B | 798 | 347 |
| Sağlıklı | C | 2 | 0 |

Bağımsız yaprak (grup) sayıları — eğitim/doğrulama: CBB 373/159, hale 102/44, sağlıklı 643/284.
Hale sınıfında 231 görüntü yalnızca 146 farklı yaprak grubundan gelir.

---

## 4. Model mimarisi — FasulyeCNN (betik: `src/03_model.py`)

Hazır mimari ya da önceden eğitilmiş ağırlık **yok**; ağırlıklar He (Kaiming) başlatmasıyla rastgele.
Hocanın "Model-2"sinin (4×4 evrişim + ReLU, 2×2 havuzlama, %20 dropout, softmax) ruhuna uygun kendi tasarımımız.

| # | Katman | Ayrıntı | Çıkış boyutu | Parametre |
|---|---|---|---|---|
| – | Giriş | RGB | 128×128×3 | 0 |
| 1 | Evrişim | 32 filtre, 4×4, 'same', BatchNorm, ReLU | 128×128×32 | 1 536 + 64 |
| 2 | Maks. havuzlama | 2×2 | 64×64×32 | 0 |
| 3 | Dropout | %20 | 64×64×32 | 0 |
| 4 | Evrişim | 64 filtre, 4×4, BN, ReLU | 64×64×64 | 32 768 + 128 |
| 5 | Maks. havuzlama | 2×2 | 32×32×64 | 0 |
| 6 | Dropout | %20 | | 0 |
| 7 | Evrişim | 128 filtre, 4×4, BN, ReLU | 32×32×128 | 131 072 + 256 |
| 8 | Maks. havuzlama | 2×2 | 16×16×128 | 0 |
| 9 | Dropout | %20 | | 0 |
| 10 | Evrişim | 256 filtre, 4×4, BN, ReLU | 16×16×256 | 524 288 + 512 |
| 11 | Maks. havuzlama | 2×2 | 8×8×256 | 0 |
| 12 | Dropout | %20 | | 0 |
| 13 | Evrişim | 256 filtre, 4×4, BN, ReLU (Grad-CAM katmanı) | 8×8×256 | 1 048 576 + 512 |
| 14 | Global ortalama havuzlama | | 256 | 0 |
| 15 | Tam bağlı | 128 nöron, ReLU, Dropout %30 | 128 | 32 896 |
| 16 | Tam bağlı + Softmax | 3 sınıf | 3 | 387 |

**Toplam eğitilebilir parametre: 1 772 995.** (Evrişimlerde bias yok, BatchNorm kaydırması üstlenir.)

Gerekçeler: **128×128 giriş** — 50×50'de küçük lezyonlar ve hale (sarı halka) birkaç piksele iniyor; 128
lezyon ayrıntısını korurken hızlı eğitime izin veriyor. **Global ortalama havuzlama** — düzleştirme + büyük
tam bağlı katmana göre çok daha az parametre (küçük veri için aşırı öğrenmeye karşı). **BatchNorm** —
sıfırdan eğitimi kararlı hale getiriyor. 4×4 çift boyutlu çekirdekte 'same' dolgu PyTorch'ta asimetrik
uygulanır.

---

## 5. Eğitim ayarları (betik: `src/04_egitim.py`, log: `ciktilar/egitim_logu.txt`)

| Ayar | Değer |
|---|---|
| Kayıp | Sınıf ağırlıklı çapraz entropi, w_c = N/(K·n_c): CBB 1,531, hale 3,525, sağlıklı 0,485 |
| Optimizer | AdamW, öğrenme oranı 1e-3, weight decay 1e-4 |
| Batch | 32 (eğitimde son eksik batch atıldı) |
| Öğrenme oranı planı | ReduceLROnPlateau (doğrulama kaybı 5 epoch iyileşmezse ×0,5) |
| Epoch | en fazla 100; **erken durdurma**: doğrulama kaybı 15 epoch iyileşmezse dur |
| Seçilen model | en düşük doğrulama kaybı (`ciktilar/model_en_iyi.pt`) |
| Veri artırma (yalnız eğitim, her örnekte rastgele) | yatay + dikey çevirme, 0–360° döndürme, rastgele kırp-boyutlandır (alanın %75–100'ü, en-boy 0,9–1,1), parlaklık ±%20, kontrast ±%20, doygunluk ±%15, ton ±0,02; artırma sonrası maske dışı tekrar siyaha boyandı |
| Tohum | 42 (Python, NumPy, PyTorch; cuDNN deterministik) |

**Dengesizlik kararı ve gerekçesi:** Sağlıklı sınıfı alt-örneklemek yerine sınıf ağırlıklı kayıp + veri
artırma seçildi. Neden: (1) veri zaten az ve sağlıklı sınıf iki kaynaktan oluşuyor; atmak çeşitliliği
azaltırdı; (2) ağırlıklı kayıp, her sınıfın toplam kayba eşit katkı yapmasını sağlar; (3) dengesizlik
altında doğruluk yanıltıcı olacağı için makro F1 ve dengeli doğruluk da raporlandı.

**Eğitim sonucu:** 50 epoch çalıştı (erken durdurma), en iyi epoch **35** (doğrulama kaybı 0,0627; o epochta
eğitim doğruluğu %97,5). Süre **1009 s** (~17 dk; bu sırada GPU'da iki kontrol deneyi de paralel çalıştığı
için tek başına daha kısa sürerdi). Eğitim doğruluğu, artırma ve dropout nedeniyle çoğu epochta doğrulama
doğruluğunun altında kaldı. Doğrulama doğruluğu epochlar arasında %85–99 arasında dalgalandı (eğri şekli).

---

## 6. Sonuçlar

### 6.1 Ana model — doğrulama kümesi (n = 738) (`metrikler.json → ana_model.dogrulama`)

| Ölçüt | Değer |
|---|---|
| **Doğruluk** | **%97,56** (720/738) |
| Dengeli doğruluk | %98,68 |
| **Makro F1** | **%95,48** |

| Sınıf | Kesinlik | Duyarlılık | F1 | Destek |
|---|---|---|---|---|
| CBB | 1,000 | 0,994 | 0,997 | 160 |
| Hale yanıklığı | 0,793 | 1,000 | 0,885 | 69 |
| Sağlıklı | 1,000 | 0,967 | 0,983 | 509 |

**Karışıklık matrisi** (satır = gerçek, sütun = tahmin):

| | CBB | Hale | Sağlıklı |
|---|---|---|---|
| **CBB** | 159 | 1 | 0 |
| **Hale** | 0 | 69 | 0 |
| **Sağlıklı** | 0 | 17 | 492 |

**Kaynak bazında doğruluk:**

| Alt küme | n | Doğru | Doğruluk | Yanlışların gittiği sınıf |
|---|---|---|---|---|
| CBB, kaynak A | 152 | 152 | %100 | – |
| CBB, kaynak C (Embrapa) | 8 | 7 | %87,5 | 1 → hale |
| Hale, kaynak B | 69 | 69 | %100 | – |
| Sağlıklı, kaynak A | 162 | 162 | %100 | – |
| Sağlıklı, kaynak B | 347 | 330 | %95,1 | 17 → hale |
| Tüm A / B / C | 314 / 416 / 8 | | %100 / %95,9 / %87,5 | |

Taban çizgisi (hep "sağlıklı" deme): doğruluk %69,0, makro F1 %27,2.

**CBB ↔ hale karışıklığının yorumu:** Matriste CBB ile hale arasında yalnızca 1 karışma var (Embrapa'dan
bir CBB → hale). Bu, iki hastalığı iyi ayırdığımız anlamına **gelmez**: doğrulamadaki tüm hale görüntüleri
B, CBB'lerin 152/160'ı A kaynaklıdır ve kaynak, maskeli görüntüden %100 tahmin edilebilmektedir (§6.2b).
Yani bu hücre büyük olasılıkla "A mı B mi" sorusunun cevabını yansıtır. Ayrıca önceki aşamadaki görsel
incelemeye göre hale klasöründeki görüntülerin çoğu CBB'ye benzer büyük kahverengi kenar nekrozu
gösteriyor; uzman olmadığımız için etiketler değiştirilmedi. Gerçek hata deseni ise **aynı kaynak içinde**:
B'deki 17 sağlıklı yaprak, aynı kaynaktan gelen tek hastalık sınıfı olan hale olarak tahmin edildi (hale
kesinliği 0,793'e bu yüzden düştü). A'daki sağlıklı yaprakların hiçbiri CBB denmedi.

### 6.2 Kontrol deneyleri (kısayol / sızıntı araştırması)

Doğruluk %97'nin üstünde çıktığı için başarı olarak sunmadan önce aşağıdakiler yapıldı.

**(a) Siluet kontrolü** (`--girdi siluet`; `ciktilar/kontrol_siluet/`): Aynı mimari, aynı bölme, aynı
ayarlar; ama giriş yalnızca maske (yaprak içi düz gri 160, dışı siyah) — renk ve doku yok, yalnızca dış şekil.
- Doğruluk **%52,3**, dengeli doğruluk %56,2, makro F1 **%47,9** (en iyi epoch 31/46, 487 s).
- Sınıf F1: CBB 0,622, hale 0,222, sağlıklı 0,592. Karışıklık: [[117,17,26],[11,34,24],[88,186,235]].
- Yorum: Siluet tek başına çoğunluk tabanının (%69) bile altında doğruluk veriyor; şanstan (makro F1 ~%33)
  biraz iyi — şekilde bir miktar bilgi (ör. kaynaklar arası yaprak biçimi/kırpma farkı, delikler) var ama
  ana modelin başarısını açıklamıyor. **Siluet tek başına yüksek doğruluk vermiyor.**

**(b) Kaynak tahmini kontrolü** (`--gorev kaynak`; `ciktilar/kontrol_kaynak/`): Aynı CNN, yalnız sağlıklı
yapraklarda (A: siyah kumaş, B: beyaz zemin; maskelenmiş ve aynı biçimde işlenmiş), hedef = kaynak.
- Doğrulama doğruluğu **%100** (509/509; A 162/162, B 347/347), 1176 eğitim görüntüsü, 66 epoch.
- Yorum: Maskelemeye rağmen kaynak imzası (renk tonu, aydınlatma, kamera, alt/üst yüz, kenar izi) tamamen
  öğrenilebilir durumda. Hale = B ve CBB ≈ A olduğu için **CBB ↔ hale ayrımı kaynaktan ayırt edilemez**.

**(c) Kaynak dışı test — Embrapa** (`--haric_kaynak embrapa_pddb`; `ciktilar/kontrol_embrapa_haric/`):
Kaynak C eğitimden ve doğrulamadan tamamen çıkarılarak aynı model eğitildi (1694 eğitim / 730 doğrulama,
28 epoch, en iyi 13). Sonra C'nin 27 görüntüsünün tamamında test edildi.
- Aynı modelin A+B doğrulama başarımı: doğruluk **%99,59**, makro F1 %99,19.
- C üzerinde: CBB **5/25 doğru (%20)** → 13 "hale", 7 "sağlıklı"; sağlıklı 2/2 doğru; toplam 7/27 (%25,9).
- Yorum: Model, hiç görmediği bir kaynaktan gelen gerçek (*P. vulgaris*, uzman etiketli) CBB yapraklarını
  büyük ölçüde tanıyamıyor. Ana modelin C doğrulamasındaki 7/8 başarısı, eğitimde C'den 17 görüntü
  görmesine bağlı. Bu, %97,6'nın dağılım-içi (aynı kaynaklar) bir başarı olduğunu gösterir. Not: n = 25 küçük
  ve C görüntüleri farklı bir belirti görünümüne (dağınık kloroz, damar nekrozu) sahip.

**(d) Grad-CAM** (`ciktilar/sekiller/gradcam_ornekleri.png`; nicel özet `metrikler.json →
ana_model.dogrulama.gradcam_nicel`): son evrişim katmanı (8×8), tahmin edilen sınıf için, tüm 738 doğrulama
görüntüsünde CAM ağırlığının nereye düştüğü ölçüldü. "Zenginleşme" = CAM payı / alan payı (>1 o bölgeye
alanından fazla bakılıyor).

| Gerçek sınıf | CAM payı yaprak içi | Yaprak alan payı | CAM payı kenar şeridi (4 px) | Kenar alan payı | Kenar zenginleşme (medyan) | CAM payı lezyon | Lezyon alan payı | Lezyon zenginleşme (medyan) |
|---|---|---|---|---|---|---|---|---|
| CBB | 0,592 | 0,464 | 0,105 | 0,102 | 1,03 | 0,111 | 0,076 | 1,44 |
| Hale | 0,681 | 0,468 | 0,114 | 0,101 | 1,14 | 0,073 | 0,028 | 2,65 |
| Sağlıklı | 0,549 | 0,491 | 0,090 | 0,103 | 0,89 | ~0 | ~0 | – |

- Yorum: Hastalıklı sınıflarda CAM lezyon piksellerine alanlarının 1,4 (CBB) – 2,7 (hale) katı ağırlık
  veriyor; örnek görsellerde kırmızı bölgeler çoğunlukla lekelerin üzerinde. Yaprak kenarına özel bir odak
  yok (zenginleşme ≈ 1). Ancak CAM ağırlığının %32–45'i yaprak dışındaki siyah alana düşüyor: 8×8 haritanın
  büyütülmesinden kaynaklanan bulanıklık bunun bir kısmını açıklar; sağlıklı örneklerde köşelerdeki siyah
  alanın da vurgulandığı görülüyor (lezyon "yokluğu" kararı). Grad-CAM, modelin lekeye baktığını gösterir
  ama lekenin **kaynak-özgü görünümünü** (A'nın pembemsi lekeleri vs B'nin kahverengi nekrozu) mi yoksa
  hastalığa özgü özelliği mi kullandığını ayıramaz.

**Kontrollerin toplam anlamı:** Sızıntı (grup) yok, zemin kaldırıldı, siluet yetmiyor, model lekelere
bakıyor → **aynı kaynak içinde hastalıklı/sağlıklı ayrımı** gerçek görünüyor (A'da sağlıklıların %100'ü
doğru; B'de sağlıklıların %95,1'i doğru, hatalar hale yönünde). Ancak **hastalık türü (CBB vs hale) ayrımı
kaynakla tamamen karışık** ve kaynak dışı genelleme zayıf (%20). %97,6'lık doğruluk bu nedenle bir üst sınır
/ iyimser tahmindir.

### 6.3 İsteğe bağlı ek: transfer öğrenmesi (EfficientNet-B0) — `src/07_ek_efficientnet.py`

Ana teslimat değildir; dersin ikinci yaklaşımıyla karşılaştırma için. ImageNet ön eğitimli EfficientNet-B0
(torchvision, ağırlıklar download.pytorch.org), tüm ağ ince ayar, giriş 224×224, AdamW 3e-4, aynı bölme, aynı
artırma ve ağırlıklı kayıp, erken durdurma sabrı 8.
- Doğrulama doğruluğu **%100**, makro F1 **%100**; tüm kaynak alt kümelerinde %100. 4 011 391 parametre,
  21 epoch (en iyi 13), 699 s.
- Yorum: Ön eğitimli model bu doğrulama kümesini kusursuz ayırıyor; ancak §6.2'deki kaynak karışması aynen
  geçerli olduğundan bu da hastalık ayrımının kanıtı değildir. Kaynak dışı (Embrapa) test bu model için
  **yapılmadı** (C görüntüleri eğitiminde vardı).

---

## 7. Hastalık şiddeti tahmini (kural tabanlı) — `src/06_siddet_tahmini.py`

**Bu bir uzman etiketi değildir.** Veri setinde şiddet etiketi yok; hocanın "sağlıklı / az hastalıklı /
hastalıklı yaprak" gösterme isteği için piksel oranına dayalı yaklaşık bir ölçü üretildi.

**Yöntem:** 256×256 işlenmiş görüntüde yaprak maskesi 5×5 aşındırıldı (kenar karışımı atıldı), 3×3 medyan
süzgeci uygulandı. Bir piksel **lezyon** sayıldı eğer:
- Lab'de a* − 128 > 4 (kırmızımsı / kahverengi / nekrotik doku; sağlıklı yeşil doku a* < 128), **veya**
- HSV'de H < 27 (≈54°) ve S > 60 (sarı-turuncu: kloroz, hale), **veya**
- V < 45 (çok koyu nekroz).
3×3 açma ile tek piksellik gürültü silindi. **Lezyon oranı = lezyon pikseli / yaprak pikseli.**
(İlk denemede "düşük doygunluk = kuru doku" ölçütü de vardı; A'daki grimsi alt yüzlü **sağlıklı** yapraklar
%70–90 lezyonlu çıktığı için kaldırıldı.)

**Eşikler (veriye bakılarak):**
- **T1 = %0,5** — sağlıklı sınıfın lezyon oranının %99'luk dilimi (%0,47), yukarı doğru %0,5'e yuvarlandı.
- **T2 = %13** — CBB + hale sınıflarının lezyon oranı medyanı (%13,09), %1'e yuvarlandı.
- Düzey: oran < %0,5 → **sağlıklı**; %0,5 ≤ oran < %13 → **az hastalıklı**; oran ≥ %13 → **hastalıklı**.

**Düzey başına sayılar (2451 görüntü):** sağlıklı 1716, az hastalıklı 347, hastalıklı 388.

| Sınıf (kaynak etiketi) | "sağlıklı" | "az hastalıklı" | "hastalıklı" | Lezyon oranı medyanı |
|---|---|---|---|---|
| CBB (533) | 4 | 161 | 368 | %18,2 |
| Hale (231) | 41 | 170 | 20 | %6,8 |
| Sağlıklı (1687) | 1671 | 16 | 0 | %0,0 |

Kaynak bazında medyan: CBB-A %19,1, CBB-C %1,9, hale-B %6,8, sağlıklı A ve B %0,0.
Yorum ve sınırlılıklar: Hale görüntülerinin 41'inde (%18) kural hiç belirgin lezyon bulmadı — bu, hale
etiketlerinin bir kısmının belirtisiz/çok hafif olduğu gözlemiyle uyumlu. Embrapa CBB yapraklarındaki dağınık
soluk kloroz kuralca çoğu zaman yakalanmıyor (medyan %1,9). A'daki pembe-kırmızı lekeler kolay yakalanıyor;
bu nedenle şiddet ölçüsü de kaynağa bağımlı. Yansıma, sap ve kenar gölgesi yanlış pozitif üretebiliyor
(şekilde "az hastalıklı" düzeyine düşen birkaç sağlıklı örnek görülebilir).

---

## 8. Sınırlılıklar

1. **Kaynak–sınıf karışması (en önemli):** Hale %100 kaynak B, CBB %95 kaynak A. Maskelemeden sonra bile kaynak
   %100 tahmin edilebiliyor; CBB ↔ hale ayrımı hastalık bilgisinden mi kaynak bilgisinden mi geliyor
   ayırt edilemiyor. Kaynak dışı CBB testinde başarı %20.
2. **Hale yanıklığı etiket güveni düşük:** Önceki aşamadaki görsel incelemede hale klasöründeki görüntülerin
   yalnızca ~%20–30'u tipik hale belirtisi gösteriyordu; çoğu CBB'ye benziyor. Kaynak (B) için patojen
   doğrulaması / yöntem makalesi yok. Etiketler değiştirilmedi. Şiddet kuralı da hale görüntülerinin %18'inde
   lezyon bulamadı.
3. **Tür belirsizliği:** A ve B'deki "bean" yaprakları *Phaseolus vulgaris* olmayabilir (Bangladeş fasulyesi /
   *Lablab purpureus* olabilir); yalnızca C kesin *P. vulgaris*.
4. **Ayrı test kümesi yok:** %70/%30 bölmede doğrulama kümesi hem erken durdurma/model seçimi hem de raporlama
   için kullanıldı → sonuçlar hafif iyimser. Doğrulama doğruluğu epochlar arasında %85–99 arasında
   dalgalandı; tek bir tohum/bölme ile çalışıldı (varyans ölçülmedi).
5. **Az bağımsız örnek:** Hale 231 görüntü ama 146 farklı yaprak; doğrulamada 44 hale grubu var.
6. **Maskeleme kusursuz değil:** 9 görüntü dışlandı; C'de kart parçacıkları, A'da az sayıda kumaş artığı kaldı.
7. **Şiddet ölçüsü uzman etiketi değil;** renk kuralına dayalı, kaynağa bağımlı, gri/soluk lezyonları kaçırabilir.
8. **Grad-CAM kaba:** 8×8 çözünürlük; CAM'in %32–45'i yaprak dışı siyah alana düşüyor.
9. Embrapa kaynak dışı testi yalnızca 25 CBB + 2 sağlıklı görüntüyle yapıldı (aynı yaprağın 2 karesi);
   istatistiksel güveni düşük.

## 9. Yapılmayan / başarısız olanlar

- Uzman şiddet etiketi bulunamadı; şiddet yalnız kural tabanlı.
- Kaynak karışmasını tamamen gideren bir çözüm (ör. her sınıf için her kaynaktan veri) mevcut verilerle
  mümkün değildi; yalnızca ölçüldü ve raporlandı.
- EfficientNet-B0 için kaynak dışı (Embrapa) test yapılmadı.
- Çoklu tohum / çapraz doğrulama yapılmadı.
- `calistir.ps1` tek parça olarak baştan sona çalıştırılmadı; betikler aynı sırayla tek tek çalıştırıldı
  (kontrol eğitimleri GPU'da paralel). Son ön işleme koşusunda renksiz-maske eşiği 7 idi; eşik sonradan 4'e
  indirilip elle dışlama listesi eklendi ve bu kural (maskeler değişmediği için) mevcut
  `on_isleme_raporu.csv`'ye aynen uygulandı; kod şu anki hâliyle yeniden çalıştırılınca aynı 9 dışlamayı üretir.
- İlk iki maskeleme denemesi başarısız oldu (§2) ve değiştirildi; ilk şiddet kuralı da sağlıklı grimsi
  yapraklarda başarısız olduğu için değiştirildi (§7).

---

## 10. Şekiller (`ciktilar/sekiller/`) ve diğer çıktılar

| Dosya | Ne gösteriyor |
|---|---|
| `sinif_ornekleri.png` | Her sınıftan 6 ön işlenmiş örnek, kaynak etiketiyle (A/B/C) |
| `on_isleme_oncesi_sonrasi.png` | 5 kaynak×sınıf bileşiminden (CBB-A, sağlıklı-A, hale-B, sağlıklı-B, CBB-C) birer örnek: ham görüntü → yaprak maskesi → zemin siyah + kırpılmış |
| `egitim_egrileri.png` | Ana model eğitim/doğrulama kaybı ve doğruluğu; en iyi epoch (35) kesikli çizgiyle |
| `karisiklik_matrisi.png` | Ana model doğrulama karışıklık matrisi (sayı + satır yüzdesi) |
| `gradcam_ornekleri.png` | Her sınıftan 3 doğru + 1 yanlış örnek ve Grad-CAM ısı haritası |
| `dogru_yanlis_ornekleri.png` | Üstte 6 doğru (her sınıftan 2), altta rastgele seçilmiş 6 yanlış sınıflandırılan doğrulama örneği (bu seçimde hepsi B-sağlıklı → hale; toplam 18 hatanın 17'si bu türdür) |
| `kontrol_deneyleri.png` | Ana model vs siluet kontrolü vs çoğunluk tabanı vs Embrapa kaynak dışı test; başlıkta kaynak tahmini %100 |
| `karisiklik_matrisi_siluet_kontrol.png` | Siluet kontrol modelinin karışıklık matrisi |
| `egitim_egrileri_siluet_kontrol.png` | Siluet kontrol modelinin eğitim eğrileri |
| `siddet_duzeyi_ornekleri.png` | Her şiddet düzeyinden 5 yaprak ve altında lezyon sayılan pikseller (mor) |
| `siddet_dagilimi.png` | Sınıflara göre lezyon oranı histogramı ve T1/T2 eşikleri |

Diğer: `ciktilar/model_en_iyi.pt` (ana model ağırlıkları + normalizasyon), `ciktilar/metrikler.json`,
`ciktilar/egitim_logu.txt`, `ciktilar/egitim_ozeti.json` (epoch geçmişi), `ciktilar/dogrulama_tahminleri.csv`
(her doğrulama görüntüsü için olasılıklar), `ciktilar/siddet_tahmini.csv`, `ciktilar/kontrol_*/`,
`ciktilar/ek_efficientnet_b0/`, `ciktilar/maske_kontrol/` (maske denetim sayfaları: `rastgele_*`,
`bayrakli_*`, `en_dusuk_chroma_A.png`, `deneme_mendeley_ykvcrjffzd.png`), `logs/` (çalıştırma logları).

## 11. Rapor için kısa akış ("şöyle yaptık")

1. Üç açık kaynaktan 2460 fasulye yaprağı görüntüsü toplandı (CBB 533, hale 231, sağlıklı 1696).
2. Ön işleme: yaprak zeminden ayrıldı (eşikleme + morfoloji + GrabCut), zemin siyaha boyandı, yaprağa kırpıldı,
   256×256 PNG; 9 hatalı maske elendi → 2451 görüntü.
3. Yakın-kopya gruplarına göre %70 eğitim (1713) / %30 doğrulama (738) bölündü.
4. Sıfırdan 16 katmanlı CNN tasarlandı (4×4 evrişim + ReLU, 2×2 havuzlama, %20 dropout, softmax; 1,77 M parametre).
5. Sınıf ağırlıklı kayıp ve veri artırmayla 50 epoch eğitildi (en iyi epoch 35).
6. Doğrulama: doğruluk %97,6, makro F1 %95,5.
7. Kontroller: siluet %52,3; kaynak tahmini %100; görülmemiş kaynaktaki CBB %20 → başarı kaynak karışmasıyla şişmiş.
8. Kural tabanlı şiddet: sağlıklı (<%0,5 lezyon), az hastalıklı (%0,5–13), hastalıklı (≥%13).
