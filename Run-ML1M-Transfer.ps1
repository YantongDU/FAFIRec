$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = Join-Path (Split-Path -Parent $projectPath) 'FAFIRec_ML1M\.venv\Scripts\python.exe'
}
try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'The local ML-1M Python environment is missing.'
    }
    $activeRuns = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.ExecutablePath -eq $pythonPath -and
        ($_.CommandLine -like '*train_ml1m.py*' -or $_.CommandLine -like '*validate_ml1m.py*')
    })
    if ($activeRuns.Count -gt 0) {
        throw 'An ML-1M experiment is already active. Wait for it to finish.'
    }
    Set-Location -LiteralPath $projectPath
    $env:PYTHONPATH = ''
    $env:TMP = Join-Path $projectPath 'data\local_temp'
    $env:TEMP = $env:TMP
    New-Item -ItemType Directory -Path $env:TMP -Force | Out-Null
    $configPath = 'configs/ml1m_stable_transfer.json'
    if (Test-Path -LiteralPath 'data/ml1m_stable_transfer_local.json') {
        $configPath = 'data/ml1m_stable_transfer_local.json'
    }
    $selectedConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $runPath = [string]$selectedConfig.output_dir
    if (-not [System.IO.Path]::IsPathRooted($runPath)) {
        $runPath = Join-Path $projectPath $runPath
    }
    Write-Host 'Stable transfer: training-only ridge projection and validation-selected fine-tuning.'
    Write-Host 'Five seeds and matched CF/uniform/diagnostic controls; completed jobs are verified and reused.'
    & $pythonPath train_ml1m.py --config $configPath --resume
    if ($LASTEXITCODE -ne 0) { throw 'Training or artifact verification failed.' }
    & $pythonPath analyze_ml1m.py $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Complete-grid analysis failed.' }
    & $pythonPath analyze_ml1m.py $runPath --baseline cf --output-dir (Join-Path $runPath 'analysis_vs_cf')
    if ($LASTEXITCODE -ne 0) { throw 'CF comparison analysis failed.' }
    Write-Host ''
    Write-Host 'Complete. Results:'
    Write-Host (Join-Path $runPath 'analysis\paper_table.csv')
    exit 0
} catch {
    Write-Host ''
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
