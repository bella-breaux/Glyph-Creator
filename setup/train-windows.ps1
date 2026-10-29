<#
  Glyph Studio - train and serve your own pixel model (glyphdraw) on Windows, in one command.

  Run in PowerShell on the PC with the NVIDIA GPU:
    irm https://raw.githubusercontent.com/bella-breaux/Glyph-Creator/main/setup/train-windows.ps1 | iex

  What it does:
    1. Gets the Glyph-Creator code into %USERPROFILE%\GlyphStudio-training (its own copy, separate from any repo you edit)
    2. Installs Python 3.12 if missing (winget), and makes a private Python environment (.venv)
    3. Installs PyTorch with CUDA, sentence-transformers, numpy, pillow, requests
    4. Runs training\auto_train.py: collect data -> train -> serve on port 11501
       (the studio setup script already publishes that port on your tailnet as :8444)

  Optional before running, for finding many more Glyph Matrix designs on GitHub:
    $env:GITHUB_TOKEN = "github_pat_..."
  Extra options (when run from a downloaded copy): .\train-windows.ps1 -TrainArgs "--big"
#>
param([string]$TrainArgs = "", [string]$Dir = "$env:USERPROFILE\GlyphStudio-training")
$ErrorActionPreference = "Continue"
function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function Fail($t) { Write-Host "  $t" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }
function Refresh-Path { $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") }

Step "Getting the code"
if (Test-Path "$Dir\.git") {
  git -C $Dir fetch origin | Out-Host
  git -C $Dir reset --hard origin/main | Out-Host   # this folder is only used for training; it never pushes
} elseif (Get-Command git -ErrorAction SilentlyContinue) {
  git clone https://github.com/bella-breaux/Glyph-Creator.git $Dir | Out-Host
} else {
  $zip = Join-Path $env:TEMP "glyph-creator.zip"
  Invoke-WebRequest -UseBasicParsing https://github.com/bella-breaux/Glyph-Creator/archive/refs/heads/main.zip -OutFile $zip
  Expand-Archive $zip -DestinationPath $env:TEMP -Force
  New-Item -ItemType Directory -Force -Path $Dir | Out-Null
  Copy-Item "$env:TEMP\Glyph-Creator-main\*" $Dir -Recurse -Force
}
if (-not (Test-Path "$Dir\training\auto_train.py")) { Fail "Couldn't get the code into $Dir." }

Step "Python"
$py = $null
foreach ($v in "3.12", "3.11") { try { $out = & py -$v -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $out) { $py = $out.Trim(); break } } catch {} }
if (-not $py) {
  Write-Host "    Installing Python 3.12..." -ForegroundColor Yellow
  winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements --silent | Out-Host
  Refresh-Path
  try { $py = (& py -3.12 -c "import sys; print(sys.executable)").Trim() } catch {}
}
if (-not $py) { Fail "Couldn't find Python 3.12. Install it from https://www.python.org/downloads/ and run this again." }
$venv = "$Dir\training\.venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) { & $py -m venv $venv }
$vpy = "$venv\Scripts\python.exe"
Write-Host "    Using $vpy" -ForegroundColor Green

Step "Installing PyTorch (CUDA) and helpers - a few GB the first time"
& $vpy -m pip install --upgrade pip -q
$hasCuda = (& $vpy -c "import torch; print(torch.cuda.is_available())" 2>$null) -eq "True"
if (-not $hasCuda) {
  & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cu128 | Out-Host
}
& $vpy -m pip install sentence-transformers numpy pillow requests -q | Out-Host
$gpu = & $vpy -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>$null
Write-Host "    GPU for training: $gpu" -ForegroundColor Green

Step "Training (collect data -> train -> serve)"
Write-Host "    Progress is also written to $Dir\training\auto_train.log"
Push-Location "$Dir\training"
& $vpy auto_train.py $TrainArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
Pop-Location
Read-Host "Press Enter to close"
