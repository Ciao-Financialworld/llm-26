param(
    [string]$CondaEnv = "llm-26-gpu",
    [string]$MinGWBin = "D:\APP6\MinGW-w64\mingw64\bin"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] $Message"
}

function Require-Path {
    param([string]$PathToCheck, [string]$Hint)
    if (-not (Test-Path $PathToCheck)) {
        throw "Missing required path: $PathToCheck`nHint: $Hint"
    }
}

function Run-Cmd {
    param([string]$CommandLine)
    cmd /c $CommandLine
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $CommandLine"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$kenlmRoot = Join-Path $repoRoot "kenlm"

$queryExe = Join-Path $kenlmRoot "build-win-native\bin\query.exe"
$buildBinaryExe = Join-Path $kenlmRoot "build-win-native\bin\build_binary.exe"
$arpaPath = Join-Path $kenlmRoot "lm\test.arpa"

$envPython = "E:\conda_data\envs\$CondaEnv\python.exe"
$tempRoot = Join-Path $env:TEMP "kenlm_win_check"
$tempInput = Join-Path $tempRoot "query_input.txt"
$tempArpa = Join-Path $tempRoot "test.arpa"
$tempBinary = Join-Path $tempRoot "test.binary"

Write-Step "Checking required files"
Require-Path $queryExe "Build native binaries first with CMake"
Require-Path $buildBinaryExe "Build native binaries first with CMake"
Require-Path $arpaPath "KenLM test model should exist in lm/test.arpa"
Require-Path $envPython "Ensure conda env '$CondaEnv' exists"

Write-Step "Preparing temporary ASCII-only test directory"
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
Set-Content -Path $tempInput -Value "this is a sentence ." -NoNewline
Copy-Item -Path $arpaPath -Destination $tempArpa -Force

Write-Step "Setting MinGW runtime PATH"
$env:Path = "$MinGWBin;$env:Path"

Write-Step "CLI check: query with ARPA"
Run-Cmd "`"$queryExe`" `"$tempArpa`" < `"$tempInput`""

Write-Step "CLI check: build binary model"
& $buildBinaryExe $tempArpa $tempBinary
if ($LASTEXITCODE -ne 0) {
    throw "build_binary failed with exit code $LASTEXITCODE"
}

Write-Step "CLI check: query with binary"
Run-Cmd "`"$queryExe`" `"$tempBinary`" < `"$tempInput`""

Write-Step "Python check: import kenlm and score sentence"
& $envPython -c "import kenlm; m=kenlm.Model(r'$tempArpa'); print('python_score=', m.score('this is a sentence .', bos=True, eos=True))"
if ($LASTEXITCODE -ne 0) {
    throw "Python kenlm check failed with exit code $LASTEXITCODE"
}

Write-Step "Cleaning temporary files"
Remove-Item -Recurse -Force $tempRoot

Write-Host "[OK] KenLM CLI and Python checks passed for env: $CondaEnv"
