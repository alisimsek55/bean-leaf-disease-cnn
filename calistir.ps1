# Tüm hattı baştan sona çalıştırır. Proje yolu çok uzunsa Windows'ta Python DLL'leri
# "dosya adı çok uzun" hatası verebilir; projeyi kısa bir yola taşıyın.
#   Örnek:  powershell -ExecutionPolicy Bypass -File <proje-koku>\calistir.ps1
#   İsteğe bağlı EfficientNet karşılaştırması için:  ... calistir.ps1 -EkModel
param([switch]$EkModel)
$ErrorActionPreference = "Stop"
$kok = $PSScriptRoot
$py = Join-Path $kok ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force (Join-Path $kok "logs") | Out-Null

function Adim($ad, $betik, $argumanlar = @()) {
    Write-Host "=== $ad ==="
    & $py (Join-Path $kok "src\$betik") @argumanlar
    if ($LASTEXITCODE -ne 0) { throw "$betik başarısız (çıkış kodu $LASTEXITCODE)" }
}

Adim "1) Ön işleme (maskeleme, kırpma, 256x256)" "01_on_isleme.py"
Adim "1b) Maske denetim sayfaları" "01b_maske_kontrol.py" @("rastgele")
Adim "2) Grup + katman bazlı %70/%30 bölme" "02_veri_bolme.py"
Adim "3) Model özeti" "03_model.py"
Adim "4a) Ana model eğitimi" "04_egitim.py"
Adim "4b) Kontrol: yalnız siluet" "04_egitim.py" @("--girdi", "siluet", "--ad", "siluet")
Adim "4c) Kontrol: sağlıklılarda kaynak tahmini" "04_egitim.py" @("--gorev", "kaynak", "--ad", "kaynak")
Adim "4d) Kontrol: Embrapa (C) kaynak dışı test" "04_egitim.py" @("--haric_kaynak", "embrapa_pddb", "--ad", "embrapa_haric")
Adim "6) Şiddet tahmini" "06_siddet_tahmini.py"
if ($EkModel) { Adim "7) Ek: EfficientNet-B0 (transfer öğrenmesi)" "07_ek_efficientnet.py" }
Adim "5) Değerlendirme, Grad-CAM, şekiller, metrikler.json" "05_degerlendirme.py"
Write-Host "Bitti. Sonuçlar: $kok\ciktilar"
