# Start the FastAPI backend orchestrator on port 8000 in a background job
Start-Job -Name "Backend" -ScriptBlock {
    Set-Location -Path $using:PWD
    .venv\Scripts\uvicorn.exe app:app --reload --port 8000
}

# Start the Vite React frontend on port 5173
Set-Location -Path frontend
npm run dev
