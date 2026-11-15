# Fasulye yaprağı hastalık uygulamasını başlatır (tarayıcıda http://localhost:8501 açılır).
#   powershell -ExecutionPolicy Bypass -File uygulama_baslat.ps1
$kok = $PSScriptRoot
$py = Join-Path $kok ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
& $py -m streamlit run (Join-Path $kok "uygulama\app.py")
