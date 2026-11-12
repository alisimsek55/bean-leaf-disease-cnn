# Fasulye yaprağı hastalık sınıflandırması: sıfırdan tasarlanmış CNN

[![tests](https://github.com/alisimsek55/bean-leaf-disease-cnn/actions/workflows/tests.yml/badge.svg)](https://github.com/alisimsek55/bean-leaf-disease-cnn/actions/workflows/tests.yml)

[English](README.md)

Fasulye yaprağı fotoğraflarını adi yaprak yanıklığı (CBB), hale yanıklığı ve sağlıklı olarak sınıflandıran, PyTorch ile sıfırdan tasarlanıp eğitilmiş 16 katmanlı bir evrişimli sinir ağı. Üniversitedeki "Tarımda Yapay Zekâ" dersi projesi olarak yapıldı. Seri çekim sızıntı denetiminden geçen bir bölmede doğrulama doğruluğu %96,6 (ilk bölme sızıntılıydı ve %97,6 vermişti; bkz. [Sonuçlar](#sonuçlar)). Reponun asıl ilginç tarafı bu sayıdan sonrası: bunun ne kadarının hastalığı tanımaktan, ne kadarının veri setindeki yapaylıklardan geldiğini test ettim; cevap "başlıktaki sayının düşündürdüğünden az".

![Ön işleme öncesi ve sonrası](ciktilar/sekiller/on_isleme_oncesi_sonrasi.png)

## Sonuçlar

Doğrulama kümesi, n = 735 (yakın-kopya ve seri çekim gruplarına göre gruplu %70/%30 bölme). Kaynak: [`ciktilar/metrikler.json`](ciktilar/metrikler.json).

| Ölçüt | Değer |
|---|---|
| Doğruluk | %96,60 (710/735) |
| Dengeli doğruluk | %97,23 |
| Makro F1 | %94,06 |
| Çoğunluk sınıfı tabanı (hep "sağlıklı") | %68,6 doğruluk, %27,1 makro F1 |

| Sınıf | Kesinlik | Duyarlılık | F1 | n |
|---|---|---|---|---|
| CBB | 1,000 | 0,969 | 0,984 | 162 |
| Hale yanıklığı | 0,764 | 0,986 | 0,861 | 69 |
| Sağlıklı | 0,992 | 0,962 | 0,977 | 504 |

Karışıklık matrisi (satır = gerçek, sütun = tahmin: CBB, hale, sağlıklı): CBB `[157, 2, 3]`, hale `[0, 68, 1]`, sağlıklı `[0, 19, 485]`. Kaynak bazında: CBB-A 154/154, CBB-C (Embrapa) 3/8 (2'si hale, 3'ü sağlıklı dendi), hale-B 68/69, sağlıklı-A 160/160, sağlıklı-B 325/344 (19 hata, hepsi hale yönünde).

Ayrı bir test kümesi yok: doğrulama kümesi erken durdurma ve model seçiminde de kullanıldı, bu yüzden sayılar daha kontrollerden önce de bir miktar iyimser.

**İlk bölme seri çekim kareleri sızdırıyordu.** İlk bölmede dHash yakın-kopya grupları ayrıktı, ama denetim ([`scripts/bolme_denetimi.py`](scripts/bolme_denetimi.py)) kaynak A'da aynı yaprağın seri çekim karelerinin iki tarafa dağıldığını buldu: 145 çekim saniyesi iki tarafta da vardı ve kaynak A'nın 314 doğrulama görüntüsünden 299'unun (o 738'lik doğrulama kümesinin %40,5'i) eğitimde aynı saniyede çekilmiş bir eşi bulunuyordu (2 sn'lik seri penceresiyle 312). Bölme seri çekime duyarlı gruplarla yeniden kuruldu ve tüm modeller yeniden eğitildi; denetim artık 0 ortak grup, dosya, çekim saniyesi ve seri buluyor.

| | Önceki bölme (seri sızıntılı) | Şimdiki bölme (denetime göre sızıntısız) |
|---|---|---|
| Eğitim / doğrulama | 1 713 / 738 | 1 716 / 735 |
| Doğruluk | %97,56 (720/738) | %96,60 (710/735) |
| Makro F1 | %95,48 | %94,06 |
| Dengeli doğruluk | %98,68 | %97,23 |
| Kaynak A (tümü) | %100 (314/314) | %100 (314/314) |
| Kaynak B (tümü) | %95,9 | %95,2 (393/413) |
| Kaynak C (CBB) | 7/8 | 3/8 |

Başlık sayısı yaklaşık bir puan düştü, ama sızıntının olduğu yerde değil: kaynak A iki bölmede de %100; yani bu koşuda seri sızıntısı A'nın sonucunu görünür biçimde şişirmemiş. Düşüş kaynak C'den (7/8 → 3/8) ve kaynak B'nin sağlıklı yapraklarından geliyor. Tek tohum, farklı doğrulama kümeleri ve iki koşu arasındaki iki kod değişikliği (bkz. [Yeniden üretme](#yeniden-üretme)) varken bir puanlık fark sızıntı düzeltmesine bağlanamaz. Ayrıca kaynak A'nın 314 doğrulama görüntüsü artık yalnızca 33 bağımsız seriden (16 CBB, 17 sağlıklı) geliyor; "A'da %100" 33 yaprağa dayanıyor.

## Doğruluk gerçek mi?

Her sınıfın büyük kısmı farklı bir veri kaynağından ve farklı bir arka plandan geliyor: hale yanıklığının tamamı kaynak B'den, CBB'nin %95'i kaynak A'dan. Ağ hastalığı değil kaynağı tanıyarak da yüksek skor alabilir. Dört kontrol yaptım, hepsi sızıntısız bölmede.

| Kontrol | Ne yapıldı | Sonuç | Ne gösteriyor |
|---|---|---|---|
| Siluet | Aynı model; girdi yalnızca yaprak dış hattı (renk/doku yok) | %52,5 doğruluk, %34,9 makro F1 (çoğunluk tabanı %68,6) | Başarı yaprak şeklinden gelmiyor. |
| Kaynak tahmini | Maskeli sağlıklı yapraklarda aynı CNN, hedef = kaynak (A/B) | %99,2 (500/504; A 159/160, B 341/344) | Maskeleme zemini kaldırdı ama kaynak imzasını değil. Hale = kaynak B, CBB ≈ kaynak A olduğundan CBB ↔ hale ayrımı kaynaktan ayrılamıyor. |
| Görülmemiş kaynak (Embrapa) | Kaynak C eğitimden çıkarılıp model yeniden eğitildi; 27 Embrapa görüntüsünde test | 25 CBB'den 2'si doğru (%8): 13'ü "hale", 10'u "sağlıklı". Sağlıklı 2/2. Aynı modelin A+B doğrulaması: %99,04. | Hiç görülmemiş kaynağa genelleme çok zayıf. n küçük (25). |
| Grad-CAM | Son evrişim katmanı (8x8), 735 doğrulama görüntüsü | Lezyon pikselinde medyan zenginleşme: CBB 1,43x, hale 2,70x; yaprak kenarında ~1x. CAM ağırlığının %34-43'ü siyah arka plana düşüyor. | Model lekeye bakıyor, ama hastalığa özgü özelliği mi kaynağa özgü leke görünümünü mü kullandığı ayırt edilemiyor. |

![Kontrol deneyleri](ciktilar/sekiller/kontrol_deneyleri.png)

**Sonuç.** Aynı kaynak içinde hastalıklı/sağlıklı ayrımı makul görünüyor (sağlıklılarda A'da %100, B'de %94,5; B'deki 19 hatanın hepsi hale yönünde) ve kaynak A seri sızıntısı giderildikten sonra da %100'de kaldı (yalnızca 33 bağımsız doğrulama serisiyle). B ve C'nin dosya adlarında çekim zamanı olmadığından onlar için yalnızca dHash gruplaması geçerli; seri sızıntısı orada ne bulunabiliyor ne dışlanabiliyor. CBB ↔ hale ayrımının hastalıktan geldiği kanıtlanamıyor, çünkü sınıf ile kaynak neredeyse tamamen örtüşüyor; görülmemiş kaynaktaki CBB'de 25'te 2 başarı, %96,6'nın aynı kaynaklar içindeki (dağılım-içi) bir sayı olduğunu gösteriyor. İsteğe bağlı ImageNet ön eğitimli EfficientNet-B0 sızıntısız bölmede de doğrulamada %100 (735/735) verdi; aynı karışma geçerli olduğundan bu da kanıt değil, ve onun için kaynak dışı test yapılmadı.

## Hat (pipeline)

- **Veri:** 3 açık kaynak, toplam 2 460 görüntü (CBB 533, hale 231, sağlıklı 1 696); 9 hatalı maske dışlandı, 2 451 kaldı. Eğitim/doğrulama: CBB 371/162, hale 162/69, sağlıklı 1 183/504 (toplam 1 716/735). Ayrıntı: [docs/DATASET.tr.md](docs/DATASET.tr.md).
- **Ön işleme:** Model zemini (beyaz PNG / siyah kumaş / kalibrasyon kartı) öğrenmesin diye yaprak maskelenir, arka plan siyaha boyanır, yaprağa kırpılıp 256x256 PNG yapılır (model 128x128 okur). [`src/01_on_isleme.py`](src/01_on_isleme.py)
- **Bölme:** Gruplu, (sınıf, kaynak) katmanlı %70/%30 (tohum 42). Grup ([`scripts/yakin_kopya.py`](scripts/yakin_kopya.py)) = dHash yakın kopyaları (256 bit, Hamming ≤ 20) + seri çekim: kaynak A'nın dosya adındaki çekim zamanına göre aynı kaynak ve sınıfta en çok 2 sn arayla çekilmiş kareler zincirleme aynı gruba alınır. Böylece kaynak A 1 056 dHash grubundan 110 gruba (CBB 53, sağlıklı 57) iniyor; 2 451 görüntü toplam 668 grup (eğitim/doğrulama: CBB 54/24, hale 95/51, sağlıklı 310/134). [`scripts/bolme_denetimi.py`](scripts/bolme_denetimi.py) sızıntı bulmuyor. [`src/02_veri_bolme.py`](src/02_veri_bolme.py)
- **Model:** Önceden eğitilmiş ağırlık yok; 4x4 evrişim + BatchNorm + ReLU blokları (32-64-128-256-256 filtre), 2x2 havuzlama, %20 dropout, global ortalama havuzlama, 128'lik tam bağlı katman, 3 sınıflı çıkış; **1 772 995** parametre. [`src/03_model.py`](src/03_model.py)
- **Eğitim:** Sınıf ağırlıklı çapraz entropi, AdamW (lr 1e-3, wd 1e-4), batch 32, ReduceLROnPlateau, erken durdurma (sabır 15), veri artırma, tohum 42. 26 epoch çalıştı, en iyi epoch 11. [`src/04_egitim.py`](src/04_egitim.py)
- **Şiddet tahmini:** Kural tabanlı, **uzman etiketi değil** (veri setinde şiddet etiketi yok). Lezyon oranı eşikleri T1 = %0,5 ve T2 = %13 (T1'in altı sağlıklı, T1 ile T2 arası az hastalıklı, T2 ve üstü hastalıklı); 2 451 görüntüde sağlıklı 1 716, az hastalıklı 347, hastalıklı 388. [`src/06_siddet_tahmini.py`](src/06_siddet_tahmini.py)

![Şiddet düzeyi örnekleri](ciktilar/sekiller/siddet_duzeyi_ornekleri.png)

## Sınırlılıklar

1. Kaynak-sınıf karışması (en önemlisi): hale %100 kaynak B, CBB %95 kaynak A; kaynak maskelemeden sonra bile %99,2 tahmin edilebiliyor. Kaynak dışı CBB başarısı %8 (2/25).
2. Seri sızıntısı yalnızca kaynak A'da denetlenebiliyor. İlk bölme A'nın seri karelerini sızdırıyordu (145 ortak çekim saniyesi; A'nın 314 doğrulama görüntüsünden 299'u etkileniyordu); bölme yeniden kuruldu ve raporlanan tüm sayılar yeni bölmeden. B ve C'de dosya adında zaman damgası olmadığından orada seri sızıntısı ne bulunabiliyor ne dışlanabiliyor. Kaynak A'nın 24 çekim dakikası hâlâ iki tarafta var (eğitim ile doğrulamadaki en yakın iki kare arasında 11,3 sn var; gruplama bunları ayrı seri sayıyor, ama farklı yapraklar oldukları gözle denetlenmedi).
3. Hale yanıklığı etiket güveni düşük: görsel incelemede hale klasörünün yalnızca ~%20-30'u tipik hale belirtisi gösteriyordu; çoğu CBB'ye benziyor. Etiketler değiştirilmedi.
4. Tür belirsizliği: A ve B'deki yapraklar *Phaseolus vulgaris* olmayabilir; yalnızca C kesin.
5. Ayrı test kümesi yok; tek tohum, tek bölme; doğrulama doğruluğu epochlar arasında %74-98 dalgalandı. Koşular arasındaki bir puan civarındaki farklar (yukarıdaki %97,56 ile %96,60 gibi) bu düzenekle ayırt edilemez.
6. Az bağımsız örnek: hale 231 görüntü ama 146 farklı yaprak (doğrulamada 51 grup); kaynak A'da 1 057 görüntü ama yalnızca 110 grup, doğrulamadaki 314 A görüntüsü 33 gruptan.
7. Maskeleme kusursuz değil (9 görüntü dışlandı; bazı kart/kumaş artıkları kaldı).
8. Şiddet ölçüsü kural tabanlı ve kaynağa bağımlı.
9. Grad-CAM kaba (8x8); ağırlığın %34-43'ü yaprak dışında.
10. Embrapa testi yalnızca 25 CBB + aynı yaprağın 2 sağlıklı karesiyle yapıldı.

Yapılmayanlar: uzman şiddet etiketi, çoklu tohum/çapraz doğrulama, EfficientNet-B0 için kaynak dışı test, `calistir.ps1`'in baştan sona tek parça çalıştırılması (betikler aynı sırayla tek tek çalıştırıldı, bazı eğitimler GPU'da paralel).

## Sırada ne yapardım

- Her sınıf için birden fazla kaynaktan veri toplamak (kaynak ile hastalık aynı değişken olmasın) ve kaynak-dışarıda-bırakma ile değerlendirmek.
- Hale etiketlerini bir bitki patoloğuna doğrulatmak ya da sınıfı doğrulanmış bir sınıfla değiştirmek.
- Kaynak dışı testi tüm modellere uygulamak, birden fazla tohumla (ya da gruplu çapraz doğrulamayla) varyans raporlamak, gerçek bir test kümesi ayırmak.
- Zaman damgası olmayan B ve C kaynaklarında aynı yaprağın karelerini gruplamanın bir yolunu bulmak (ör. gözle denetlenen öğrenilmiş bir benzerlik ölçüsü).

## Yeniden üretme

Python 3.13; sürümleri sabitlenmiş [`requirements.txt`](requirements.txt) (CUDA 12.8 PyTorch). CPU için torch/torchvision'ı `https://download.pytorch.org/whl/cpu` dizininden kurun; testlerin ihtiyaçları [`requirements-ci.txt`](requirements-ci.txt) içinde.

**Raporlanan sayılar nasıl üretildi.** [`ciktilar/metrikler.json`](ciktilar/metrikler.json) ve bu README'deki tüm sayılar, yeniden kurulan bölmede ve şimdiki kodla (değerlendirme yeniden boyutlandırmadan sonra yaprak maskesini yeniden uyguluyor; en iyi kontrol noktası gerçekten en düşük doğrulama kaybı; gruplama seri çekime duyarlı) her deney için birer koşudan geliyor. Ayarlar ve tohumlar betiklerin varsayılanları, ilk koşudakiyle aynı. Ön işleme yeniden çalıştırılmadı: işlenmiş görüntü ve maskeler gruplamaya bağlı değil; yalnızca `data/islenmis/on_isleme_raporu.csv`'deki `grup` sütunu `data/etiketler.csv`'den yenilendi (`01_on_isleme.py`'nin kopyaladığı da budur). `02_veri_bolme.py` grupları bu dosyadan okuduğu için `yakin_kopya.py` yeniden çalıştırıldıktan sonra bölmeden önce `01_on_isleme.py` de yeniden çalıştırılmalı. İlk (sızıntılı) koşunun sayıları yalnızca yukarıdaki karşılaştırma tablosunda ve git geçmişinde duruyor.

Veri depoda dağıtılmaz; [`scripts/`](scripts) altındaki betikler indirir ve düzenler (~1,6 GB, hesap gerekmez):

```powershell
python -I scripts/indir.py . mendeley_ykvcrjffzd
python -I scripts/indir.py . mendeley_3km9d246z2
python -I scripts/indir.py . embrapa_pddb
python -I scripts/duzenle.py .
python -I scripts/yakin_kopya.py .        # dHash + seri çekim grupları (`grup` sütunu)
python -I scripts/bolme_denetimi.py .     # 02_veri_bolme'den sonra: sızıntı denetimi (sızıntı varsa çıkış kodu 1)
```

Çalıştırma sırası (ya da `calistir.ps1`; `-EkModel` ile EfficientNet-B0 da): `01_on_isleme`, `02_veri_bolme`, `04_egitim` (ana model; ayrıca `--girdi siluet`, `--gorev kaynak`, `--haric_kaynak embrapa_pddb` kontrolleri), `06_siddet_tahmini`, `07_ek_efficientnet` (isteğe bağlı), `05_degerlendirme`. Tam komutlar [İngilizce README](README.md#reproduce)'de.

Donanım: NVIDIA RTX 4070 Laptop (8 GB), PyTorch 2.11.0+cu128; ana model 901 sn'de eğitildi (başka eğitimler GPU'yu paylaşıyordu); yeniden eğitim ve değerlendirmenin tamamı yaklaşık 40 dakika sürdü. Model ağırlıkları dağıtılmıyor. Proje yolu çok uzunsa Windows'ta bazı DLL'ler yüklenemeyebilir; projeyi kısa bir yola taşıyın.

Testler (veri/GPU gerekmez): `pytest -q`. İki test depodaki bölmeyi denetler; eğitim ile doğrulama arasında ortak grup, dosya, çekim saniyesi ya da seri varsa başarısız olur.

## Veri kaynakları ve lisanslar

- **A.** Rashid M. R. A., Hasan M., Gani R., Kamara R., Tarin T. K., Rabbi S. F. *Image Dataset for Bean and Cowpea Plant Leaf Disease Detection and Freshness Assessment from Bangladesh.* Mendeley Data v1 (2024), doi:[10.17632/ykvcrjffzd.1](https://doi.org/10.17632/ykvcrjffzd.1); makale: *Data in Brief* (2024), doi:[10.1016/j.dib.2024.111023](https://doi.org/10.1016/j.dib.2024.111023). CC BY 4.0.
- **B.** Mojumdar M. U., Dip S. D., Alahi T. E., Tonmoy S. *Bean Health and Disease Dataset: A Resource for Agricultural Research.* Mendeley Data v3 (2025), doi:[10.17632/3km9d246z2.3](https://doi.org/10.17632/3km9d246z2.3). CC BY 4.0.
- **C.** Barbedo J. G. A. ve ark. *Embrapa PDDB: Base de Imagens de Sintomas de Doenças de Plantas.* doi:[10.48432/XA1OVL](https://doi.org/10.48432/XA1OVL); makale: *IEEE Latin America Transactions* 16(6), 2018, doi:[10.1109/TLA.2018.8444395](https://doi.org/10.1109/TLA.2018.8444395). CC BY-NC 4.0.

`ciktilar/sekiller/` altındaki şekiller bu veri setlerinden türetilmiş görüntüler (maskelenmiş, kırpılmış yaprak fotoğrafları) içerir; aynı lisans koşullarıyla, atıfla ve yalnızca ticari olmayan amaçla paylaşılır. `data/` altındaki CSV'ler yalnızca üst veri içerir, görüntü içermez. Kod [MIT lisansıdır](LICENSE); lisans veri setlerine ve onlardan türetilmiş görüntülere uzanmaz.

## Nasıl yapıldı

Bu projeyi yapay zekâ destekli geliştirmeyle, Claude Code'u eşli programlama aracı olarak kullanarak yaptım. Kapsamı ve gereksinimleri ders yönergesinden ben belirledim: sağlıklı yapraklardan ayrılacak iki hastalık, hazır bir model yerine sıfırdan tasarlanan bir ağ, %70/%30 eğitim/doğrulama bölmesi ve sağlıklı, az hastalıklı ve hastalıklı yaprakların gösterilmesi. Uygulamanın büyük kısmını bu yönlendirmeyle yapay zekâ yaptı. Sınırlılıklar dahil sonuçlar, çıktığı haliyle raporlanmıştır.