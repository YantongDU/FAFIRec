$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$runPath = Join-Path $projectPath 'runs\ml1m_paper_qwen3_v1'

try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'The local ML-1M Python environment is missing.'
    }
    $activeRuns = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.ExecutablePath -eq $pythonPath -and
        $_.CommandLine -like '*train_ml1m.py*'
    })
    if ($activeRuns.Count -gt 0) {
        throw 'An ML-1M training run is already active. Wait for it to finish.'
    }
    Set-Location -LiteralPath $projectPath
    $env:PYTHONPATH = ''
    $env:TMP = Join-Path $projectPath 'data\local_temp'
    $env:TEMP = $env:TMP
    New-Item -ItemType Directory -Path $env:TMP -Force | Out-Null
    Write-Host 'ML-1M: chronological 80/10/10, real Qwen3 embeddings, five seeds.'
    Write-Host 'Completed jobs will be verified and reused; incomplete jobs resume.'
    & $pythonPath train_ml1m.py --config configs/ml1m_paper_qwen3.json --resume
    if ($LASTEXITCODE -ne 0) { throw 'Training or artifact verification failed.' }
    & $pythonPath analyze_ml1m.py $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Complete-grid analysis failed.' }
    Write-Host ''
    Write-Host 'Complete. Results:'
    Write-Host (Join-Path $runPath 'analysis\paper_table.csv')
    exit 0
} catch {
    Write-Host ''
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
