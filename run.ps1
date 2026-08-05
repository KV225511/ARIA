# ARIA Launcher Script
# Run this from the ARIA project root: .\run.ps1

$rootDir = (Get-Location).Path

Write-Host "ARIA root: $rootDir" -ForegroundColor Cyan

# --- Cleanup stale backend process on port 8000 ---
$existing = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($existing) {
    $stalePid = ($existing | Select-Object -First 1).OwningProcess
    Write-Host "Killing stale process on port 8000 (PID $stalePid)..." -ForegroundColor Yellow
    Stop-Process -Id $stalePid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# --- Start FastAPI backend in a new visible window ---
$uvicorn = "$rootDir\.venv\Scripts\uvicorn.exe"
Write-Host "Using uvicorn: $uvicorn" -ForegroundColor DarkGray
Write-Host "Starting ARIA Backend (uvicorn)..." -ForegroundColor Cyan
Start-Process "powershell.exe" -ArgumentList "-NoExit", "-Command", "Set-Location '$rootDir'; & '$uvicorn' app:app --reload --port 8000"

# Wait for backend to bind
Write-Host "Waiting for backend..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Probe backend health
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8000/docs" -TimeoutSec 5 -UseBasicParsing
    Write-Host "Backend is UP (HTTP $($r.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Backend not responding on port 8000. Check the backend window for errors." -ForegroundColor Red
}

# --- Start Vite frontend ---
Write-Host "Starting frontend (Vite)..." -ForegroundColor Cyan
Set-Location "$rootDir\frontend"
npm run dev
