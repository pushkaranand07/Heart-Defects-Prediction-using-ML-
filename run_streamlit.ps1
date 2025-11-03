# Convenience script to run the Streamlit app on Windows (PowerShell)
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found in PATH. Activate your virtual environment first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting HeartGuard Streamlit app..." -ForegroundColor Cyan
py -3 -m streamlit run .\app.py
