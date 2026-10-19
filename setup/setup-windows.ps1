<#
  Glyph Studio - one-time setup for Windows 10/11

  What it does:
    1. Installs Ollama and Tailscale if they're missing (via winget)
    2. Signs this PC into Tailscale if it isn't already
    3. Downloads the studio into %LOCALAPPDATA%\GlyphStudio\site
    4. Sets OLLAMA_HOST / OLLAMA_ORIGINS for you and restarts Ollama
    5. Blocks Ollama's port from your Wi-Fi/LAN (only Tailscale can reach it)
    6. Pulls a model sized for an 8 GB GPU if you have none
    7. Publishes, privately on your tailnet:
         https://<this-pc>.<tailnet>.ts.net/        -> the studio
         https://<this-pc>.<tailnet>.ts.net:8443/   -> Ollama
       Tailscale keeps these running across reboots.

  Run it (PowerShell):
    irm https://raw.githubusercontent.com/bella-breaux/Glyph-Creator/main/setup/setup-windows.ps1 | iex
  or, from a downloaded copy:
    powershell -ExecutionPolicy Bypass -File setup-windows.ps1

  Re-run any time to update the studio or repair the setup.
#>

param([string]$Model = "gemma4:e2b-it-qat")

# "Continue": in Windows PowerShell 5.1, "Stop" turns any stderr text from native tools (tailscale, winget) into a crash.
$ErrorActionPreference = "Continue"
$ScriptUrl = "https://raw.githubusercontent.com/bella-breaux/Glyph-Creator/main/setup/setup-windows.ps1"
$SiteUrl   = "https://raw.githubusercontent.com/bella-breaux/Glyph-Creator/main/index.html"
$OllamaPort = 11434
$OllamaHttpsPort = 8443

function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function Ok($t)   { Write-Host "    $t" -ForegroundColor Green }
function Warn($t) { Write-Host "    $t" -ForegroundColor Yellow }
function Fail($t) { Write-Host ""; Write-Host "  $t" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

# ---------- 0. Run as administrator (needed for the firewall rule and installs) ----------
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  Write-Host "Glyph Studio setup needs administrator rights. Windows will ask you to allow it." -ForegroundColor Yellow
  $self = $PSCommandPath
  if (-not $self) {
    $self = Join-Path $env:TEMP "glyph-studio-setup.ps1"
    Invoke-WebRequest -UseBasicParsing $ScriptUrl -OutFile $self -ErrorAction Stop
  }
  Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$self`" -Model `"$Model`""
  exit
}

function Refresh-Path {
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

# ---------- 1. Ollama + Tailscale ----------
Step "Checking Ollama and Tailscale"
$olDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
$olExe = Join-Path $olDir "ollama.exe"
$olApp = Join-Path $olDir "ollama app.exe"
$tsExe = Join-Path $env:ProgramFiles "Tailscale\tailscale.exe"

$haveWinget = [bool](Get-Command winget -ErrorAction SilentlyContinue)
if (-not (Test-Path $olExe)) {
  if (-not $haveWinget) { Fail "Ollama isn't installed and winget isn't available. Install Ollama from https://ollama.com/download, then run this again." }
  Warn "Installing Ollama..."
  winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements --silent | Out-Host
  Refresh-Path
}
if (-not (Test-Path $tsExe)) {
  if (-not $haveWinget) { Fail "Tailscale isn't installed and winget isn't available. Install it from https://tailscale.com/download, then run this again." }
  Warn "Installing Tailscale..."
  winget install --id tailscale.tailscale -e --accept-source-agreements --accept-package-agreements --silent | Out-Host
  Refresh-Path
}
if (-not (Test-Path $olExe)) { Fail "Couldn't find Ollama at $olExe after installing. Open Ollama once, then run this again." }
if (-not (Test-Path $tsExe)) { Fail "Couldn't find Tailscale at $tsExe after installing." }
Ok "Ollama and Tailscale are installed."

# ---------- 2. Tailscale sign-in ----------
Step "Checking Tailscale sign-in"
function Get-TsStatus { try { & $tsExe status --json 2>$null | Out-String | ConvertFrom-Json } catch { $null } }
$st = Get-TsStatus
if (-not $st -or $st.BackendState -ne "Running") {
  Warn "Signing this PC into Tailscale. A browser window will open. Sign in with the same account as your phone."
  & $tsExe up | Out-Host
  for ($i = 0; $i -lt 60; $i++) { $st = Get-TsStatus; if ($st -and $st.BackendState -eq "Running") { break }; Start-Sleep 2 }
}
if (-not $st -or $st.BackendState -ne "Running") { Fail "Tailscale isn't connected. Open the Tailscale app, sign in, then run this again." }
$dns = ($st.Self.DNSName).TrimEnd(".")
if (-not $dns) { Fail "Tailscale didn't report this PC's name. Turn on MagicDNS at https://login.tailscale.com/admin/dns and run this again." }
$siteOrigin = "https://$dns"
Ok "This PC is $dns"

# ---------- 3. Studio files ----------
Step "Installing the studio"
$siteDir = Join-Path $env:LOCALAPPDATA "GlyphStudio\site"
New-Item -ItemType Directory -Force -Path $siteDir | Out-Null
$localCopy = if ($PSScriptRoot) { Join-Path (Split-Path $PSScriptRoot -Parent) "index.html" } else { $null }
if ($localCopy -and (Test-Path $localCopy)) {
  Copy-Item $localCopy (Join-Path $siteDir "index.html") -Force
  Ok "Copied the studio from this folder."
} else {
  try { Invoke-WebRequest -UseBasicParsing $SiteUrl -OutFile (Join-Path $siteDir "index.html") -ErrorAction Stop } catch { Fail "Couldn't download the studio from GitHub. Check your internet connection." }
  Ok "Downloaded the latest studio from GitHub."
}

# ---------- 4. Ollama settings ----------
Step "Configuring Ollama"
# 0.0.0.0 turns off Ollama's host-name check so the .ts.net name is accepted.
# The firewall rule in step 5 keeps other devices on your network out.
$origins = @($siteOrigin, "https://bella-breaux.github.io")
$existing = [Environment]::GetEnvironmentVariable("OLLAMA_ORIGINS", "User")
if ($existing) { $origins += ($existing -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
$originsStr = ($origins | Select-Object -Unique) -join ","
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:$OllamaPort", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_ORIGINS", $originsStr, "User")
$env:OLLAMA_HOST = "0.0.0.0:$OllamaPort"
$env:OLLAMA_ORIGINS = $originsStr
Ok "OLLAMA_ORIGINS = $originsStr"

Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2
# Start through Explorer so Ollama runs as you, not as administrator.
if (Test-Path $olApp) { Start-Process explorer.exe -ArgumentList "`"$olApp`"" } else { Start-Process $olExe -ArgumentList "serve" -WindowStyle Hidden }
$up = $false
for ($i = 0; $i -lt 30; $i++) {
  try { Invoke-RestMethod "http://127.0.0.1:$OllamaPort/api/tags" -TimeoutSec 2 | Out-Null; $up = $true; break } catch { Start-Sleep 1 }
}
if (-not $up) { Fail "Ollama didn't start. Open it from the Start menu, then run this again." }
Ok "Ollama restarted with the new settings."

# ---------- 5. Firewall ----------
Step "Blocking Ollama from your local network"
$ruleName = "Glyph Studio - block Ollama from network"
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
try {
  New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $OllamaPort -Action Block -Profile Any -ErrorAction Stop | Out-Null
  Ok "Port $OllamaPort is blocked for other devices. Tailscale still reaches it from inside this PC."
} catch { Warn "Couldn't add the firewall rule. Ollama may be reachable from other devices on your Wi-Fi." }

# ---------- 6. Model ----------
Step "Checking models"
$tags = Invoke-RestMethod "http://127.0.0.1:$OllamaPort/api/tags" -ErrorAction Stop
if (-not $tags.models -or $tags.models.Count -eq 0) {
  Warn "No models installed. Downloading $Model (a few GB, this can take a while)..."
  & $olExe pull $Model | Out-Host
} else {
  Ok ("Installed: " + (($tags.models | ForEach-Object { $_.name }) -join ", "))
}

# ---------- 7. Tailscale serve ----------
Step "Publishing on your tailnet"
& $tsExe serve reset 2>$null | Out-Null
& $tsExe serve --bg --https=443 "$siteDir" | Out-Host
if ($LASTEXITCODE -ne 0) { Fail "Tailscale couldn't publish the site. Turn on HTTPS certificates at https://login.tailscale.com/admin/dns and run this again." }
& $tsExe serve --bg --https=$OllamaHttpsPort "http://127.0.0.1:$OllamaPort" | Out-Host
if ($LASTEXITCODE -ne 0) { Fail "Tailscale couldn't publish Ollama on port $OllamaHttpsPort." }
# Optional pixel model (training/pixel_server.py). Harmless until that server is running.
& $tsExe serve --bg --https=8444 "http://127.0.0.1:11501" | Out-Null

# ---------- 8. Check ----------
Step "Testing"
$siteOk = $false; $apiOk = $false
for ($i = 0; $i -lt 20 -and -not ($siteOk -and $apiOk); $i++) {
  if (-not $siteOk) { try { $r = Invoke-WebRequest -UseBasicParsing "$siteOrigin/" -TimeoutSec 5; $siteOk = $r.Content -match "Glyph Studio" } catch {} }
  if (-not $apiOk)  { try { Invoke-RestMethod "${siteOrigin}:$OllamaHttpsPort/api/tags" -TimeoutSec 5 -Headers @{ Origin = $siteOrigin } | Out-Null; $apiOk = $true } catch {} }
  if (-not ($siteOk -and $apiOk)) { Start-Sleep 3 }   # first HTTPS certificate can take a few seconds
}
if ($siteOk) { Ok "Studio is up." } else { Warn "The studio didn't answer yet. The first HTTPS certificate can take a minute; try the link below shortly." }
if ($apiOk)  { Ok "Ollama is reachable over Tailscale." } else { Warn "Ollama didn't answer over Tailscale yet. Check 'tailscale serve status'." }

Write-Host ""
Write-Host "  Done. On your phone, turn Tailscale on and open:" -ForegroundColor White
Write-Host ""
Write-Host "      $siteOrigin/" -ForegroundColor Green
Write-Host ""
Write-Host "  The AI connects by itself. Keep this PC on and awake when you want to use it."
Write-Host "  To turn it all off later:  tailscale serve reset"
Write-Host ""
Read-Host "Press Enter to close"
