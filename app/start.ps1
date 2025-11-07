# start.ps1 - Activar venv y arrancar la API (Windows PowerShell)
param(
    [int]$Port = 8000
)

Set-Location -Path $PSScriptRoot

if (-Not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing requirements (first run may take a while)..."
python -m pip install --upgrade pip
if (Test-Path -Path "requirements.txt") {
    python -m pip install -r requirements.txt
} elseif (Test-Path -Path "app/requirements.txt") {
    python -m pip install -r app/requirements.txt
}

Write-Host "Starting Uvicorn on http://127.0.0.1:$Port"
python -m uvicorn main:app --reload --port $Port
