# ARIA Launcher Script
# Run this from the ARIA project root: .\run.ps1

$rootDir = (Get-Location).Path

Write-Host "ARIA root: $rootDir" -ForegroundColor Cyan

# --- Start PostgreSQL and apply the approved schema ---
Write-Host "Starting PostgreSQL..." -ForegroundColor Cyan
docker compose up -d postgres
if ($LASTEXITCODE -ne 0) { throw "PostgreSQL could not be started." }
$alembic = "$rootDir\.venv\Scripts\alembic.exe"
& $alembic upgrade head
if ($LASTEXITCODE -ne 0) { throw "Database migrations failed." }

# --- Cleanup stale backend process on port 8000 ---
$existing = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($existing) {
    $stalePid = ($existing | Select-Object -First 1).OwningProcess
    Write-Host "Killing stale process on port 8000 (PID $stalePid)..." -ForegroundColor Yellow
    Stop-Process -Id $stalePid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# --- Start FastAPI backend in a new visible window ---
$python = "$rootDir\.venv\Scripts\python.exe"
Write-Host "Starting ARIA Backend (uvicorn)..." -ForegroundColor Cyan
Start-Process "powershell.exe" -ArgumentList "-NoExit", "-Command", "Set-Location '$rootDir'; & '$python' -m backend.server --port 8000"

# The durable worker processes document extraction and deletion jobs.
Start-Process "powershell.exe" -WindowStyle Hidden -ArgumentList "-Command", "Set-Location '$rootDir'; & '$python' -m backend.worker"

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
