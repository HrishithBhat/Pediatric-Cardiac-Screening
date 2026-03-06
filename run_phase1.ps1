# =============================================================================
#  run_phase1.ps1  —  Dataset preparation + Specialist training
#  Run from the project root:
#    cd "...\pediatric_cardiac_screening"
#    .\run_phase1.ps1
# =============================================================================

# ---------------------------------------------------------------------------
# CONFIGURATION  —  edit these paths to match your machine
# ---------------------------------------------------------------------------
$PY         = "C:\Users\hrish\OneDrive\Documents\6th sem notes\MajorProject\.venv\Scripts\python.exe"
$PROJECT    = $PSScriptRoot   # folder containing this script

$DS_US      = "C:\Users\hrish\OneDrive\Documents\6th sem notes\Major Project Phase-1\cardium_images(Ultrasound)\cardium_images"
$DS_XRAY    = "C:\Users\hrish\OneDrive\Documents\6th sem notes\Major Project Phase-1\congenital-heart-disease(xray)"
$DS_AUDIO   = "C:\Users\hrish\OneDrive\Documents\6th sem notes\Major Project Phase-1\ZCHSound(CHD)\ZCHSound"

$CHECKPOINTS = "$PROJECT\checkpoints"
$LOG_FILE    = "$PROJECT\logs\phase1_run.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line
}

function Run-Step($stepName, $scriptBlock) {
    Log "===== START : $stepName ====="
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $scriptBlock
    $sw.Stop()
    if ($LASTEXITCODE -ne 0) {
        Log "ERROR : $stepName failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Log "===== DONE  : $stepName  ($([int]$sw.Elapsed.TotalMinutes)m $($sw.Elapsed.Seconds)s) ====="
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
Set-Location $PROJECT
New-Item -ItemType Directory -Force -Path "$PROJECT\logs"        | Out-Null
New-Item -ItemType Directory -Force -Path $CHECKPOINTS           | Out-Null
"" | Set-Content $LOG_FILE    # reset log for this run

Log "Python  : $PY"
Log "Project : $PROJECT"

# Verify Python
& $PY --version
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Python not found at $PY"; exit 1 }

# ---------------------------------------------------------------------------
# PHASE 1-A  :  Dataset Preparation
# ---------------------------------------------------------------------------
Write-Host "`n============================================================"
Write-Host "  PHASE 1-A  :  DATASET PREPARATION"
Write-Host "============================================================`n"

Run-Step "Prepare CARDIUM Ultrasound CSVs" {
    & $PY "$PROJECT\data\prepare_cardium_ultrasound.py" `
        --root  $DS_US `
        --fold  all `
        --out   "$PROJECT\data\ultrasound" `
        --mode  per_image
}

Run-Step "Prepare X-Ray CSVs" {
    & $PY "$PROJECT\data\prepare_xray.py" `
        --root  $DS_XRAY `
        --out   "$PROJECT\data\xray"
}

Run-Step "Prepare Audio CSVs" {
    & $PY "$PROJECT\data\prepare_audio.py" `
        --root  $DS_AUDIO `
        --out   "$PROJECT\data\audio"
}

# ---------------------------------------------------------------------------
# PHASE 1-B  :  Specialist Training
# ---------------------------------------------------------------------------
Write-Host "`n============================================================"
Write-Host "  PHASE 1-B  :  SPECIALIST TRAINING"
Write-Host "============================================================`n"

Run-Step "Train NTS-Net (Ultrasound specialist)" {
    & $PY "$PROJECT\training\train_specialist.py" `
        --modality   ultrasound `
        --train_csv  "$PROJECT\data\ultrasound\us_train.csv" `
        --val_csv    "$PROJECT\data\ultrasound\us_val.csv" `
        --output_dir "$CHECKPOINTS\ultrasound"
}

Run-Step "Train EfficientNetV2 (X-Ray specialist)" {
    & $PY "$PROJECT\training\train_specialist.py" `
        --modality   xray `
        --train_csv  "$PROJECT\data\xray\xray_train.csv" `
        --val_csv    "$PROJECT\data\xray\xray_val.csv" `
        --output_dir "$CHECKPOINTS\xray"
}

Run-Step "Train CRNN (Audio specialist)" {
    & $PY "$PROJECT\training\train_specialist.py" `
        --modality   audio `
        --train_csv  "$PROJECT\data\audio\audio_train.csv" `
        --val_csv    "$PROJECT\data\audio\audio_val.csv" `
        --output_dir "$CHECKPOINTS\audio"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Log "============================================================"
Log "  PHASE 1 COMPLETE"
Log "  Checkpoints saved to : $CHECKPOINTS"
Log "  Next step            : run .\run_phase2.ps1"
Log "============================================================"
Write-Host ""
Write-Host "Best checkpoint files:"
Get-ChildItem "$CHECKPOINTS\*\*_best.pth" -ErrorAction SilentlyContinue |
    Select-Object FullName, @{N='Size(MB)';E={[math]::Round($_.Length/1MB,1)}} |
    Format-Table -AutoSize
