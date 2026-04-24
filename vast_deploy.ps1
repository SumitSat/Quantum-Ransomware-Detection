param (
    [Parameter(Mandatory=$true)][string]$VastSshCommand,
    [Parameter(Mandatory=$true)][string]$HfRepo,
    [Parameter(Mandatory=$true)][string]$HfFile,
    [Parameter(Mandatory=$false)][string]$HfToken = ""
)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Q-TERD Vast.ai Remote Execution Orchestrator" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Parse connection string (e.g. ssh -p 12345 root@192.168.1.1)
$Parts = $VastSshCommand -split ' '
$Port = ""
$Server = ""
for ($i=0; $i -lt $Parts.Length; $i++) {
    if ($Parts[$i] -eq "-p") { $Port = $Parts[$i+1] }
    if ($Parts[$i] -match "@") { $Server = $Parts[$i] }
}

if ([string]::IsNullOrEmpty($Port) -or [string]::IsNullOrEmpty($Server)) {
    Write-Host "❌ Invalid SSH command format. Please provide the exact Vast.ai string." -ForegroundColor Red
    exit 1
}

Write-Host "Target Server: $Server  | Port: $Port" -ForegroundColor Yellow

# 2. Rsync / SCP the Local Codebase to Vast.ai
Write-Host "`n🚀 Step 1: Pushing Local Code to Vast.ai GPU..." -ForegroundColor Green
scp -P $Port -r .\src .\scripts .\main.py .\requirements.txt .\config.py "$Server`:~/"

# 3. Create Setup Shell Script for Vast.ai
$SetupScript = @"
#!/bin/bash
echo "Installing Dependencies on Vast.ai GPU..."
pip install huggingface_hub
pip install -r requirements.txt

export HF_TOKEN="$HfToken"
echo "Downloading Dataset from Hugging Face ($HfRepo)..."
python scripts/download_vera.py --repo "$HfRepo" --file "$HfFile"

echo "Data downloaded and unzipped! Ready for Training."
# Python training script execution happens here (Will be added in next phase)
# python main.py --device lightning.kokkos
"@

Set-Content -Path .\remote_setup.sh -Value $SetupScript

Write-Host "Pushing Setup Script..."
scp -P $Port .\remote_setup.sh "$Server`:~/remote_setup.sh"

# 4. Execute Remote Setup and Download
Write-Host "`n🚀 Step 2: Executing Remote Environment Setup & HF Download..." -ForegroundColor Green
ssh -p $Port $Server "bash ~/remote_setup.sh"

Write-Host "`n✅ Infrastructure deployed! The code and dataset are now on the GPU." -ForegroundColor Cyan
