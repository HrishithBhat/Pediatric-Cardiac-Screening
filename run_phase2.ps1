# =============================================================================
#  run_phase2.ps1  —  Surgery + GMU Fusion Training
#  Run from the project root after run_phase1.ps1 completes:
#    .\run_phase2.ps1
# =============================================================================

# ---------------------------------------------------------------------------
# CONFIGURATION  —  should match run_phase1.ps1
# ---------------------------------------------------------------------------
$PY          = "C:\Users\hrish\OneDrive\Documents\6th sem notes\MajorProject\.venv\Scripts\python.exe"
$PROJECT     = $PSScriptRoot

$CHECKPOINTS = "$PROJECT\checkpoints"
$DATA        = "$PROJECT\data"
$LOG_FILE    = "$PROJECT\logs\phase2_run.log"

# Phase-1 best checkpoints
$US_BEST     = "$CHECKPOINTS\ultrasound\ultrasound_best.pth"
$XRAY_BEST   = "$CHECKPOINTS\xray\xray_best.pth"
$AUDIO_BEST  = "$CHECKPOINTS\audio\audio_best.pth"

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
New-Item -ItemType Directory -Force -Path "$PROJECT\logs"          | Out-Null
New-Item -ItemType Directory -Force -Path "$CHECKPOINTS\gmu"       | Out-Null
New-Item -ItemType Directory -Force -Path "$DATA\multimodal"       | Out-Null
"" | Set-Content $LOG_FILE

Log "Python  : $PY"
Log "Project : $PROJECT"

# Verify Phase-1 checkpoints exist
foreach ($ckpt in @($US_BEST, $XRAY_BEST, $AUDIO_BEST)) {
    if (-not (Test-Path $ckpt)) {
        Write-Host "ERROR: Missing checkpoint: $ckpt" -ForegroundColor Red
        Write-Host "Make sure run_phase1.ps1 completed successfully first."
        exit 1
    }
}
Log "All Phase-1 checkpoints found ✓"

# ---------------------------------------------------------------------------
# PHASE 2-A  :  Merge per-modality CSVs → multimodal CSV
# ---------------------------------------------------------------------------
Write-Host "`n============================================================"
Write-Host "  PHASE 2-A  :  BUILD MULTIMODAL CSV"
Write-Host "============================================================`n"

Run-Step "Prepare Multimodal CSVs" {
    & $PY "$PROJECT\data\prepare_multimodal.py" `
        --audio_train  "$DATA\audio\audio_train.csv" `
        --audio_val    "$DATA\audio\audio_val.csv"   `
        --audio_test   "$DATA\audio\audio_test.csv"  `
        --us_train     "$DATA\ultrasound\us_train.csv" `
        --us_val       "$DATA\ultrasound\us_val.csv"   `
        --us_test      "$DATA\ultrasound\us_test.csv"  `
        --xray_train   "$DATA\xray\xray_train.csv" `
        --xray_val     "$DATA\xray\xray_val.csv"   `
        --xray_test    "$DATA\xray\xray_test.csv"  `
        --out          "$DATA\multimodal"
}

# ---------------------------------------------------------------------------
# PHASE 2-B  :  GMU Fusion Training
# ---------------------------------------------------------------------------
Write-Host "`n============================================================"
Write-Host "  PHASE 2-B  :  GMU FUSION TRAINING"
Write-Host "============================================================`n"

Run-Step "Train GMU Fusion Layer" {
    & $PY "$PROJECT\training\train_gmu.py" `
        --audio_ckpt  $AUDIO_BEST `
        --us_ckpt     $US_BEST    `
        --xray_ckpt   $XRAY_BEST  `
        --train_csv   "$DATA\multimodal\multimodal_train.csv" `
        --val_csv     "$DATA\multimodal\multimodal_val.csv"   `
        --output_dir  "$CHECKPOINTS\gmu" `
        --patience    10 `
        --batch_size  8
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Log "============================================================"
Log "  PHASE 2 COMPLETE"
Log "  GMU checkpoint : $CHECKPOINTS\gmu\gmu_best.pth"
Log "  Next step      : run inference / dashboard"
Log "============================================================"
Write-Host ""
Write-Host "Final checkpoints:"
Get-ChildItem "$CHECKPOINTS\**\*_best.pth" -Recurse -ErrorAction SilentlyContinue |
    Select-Object FullName, @{N='Size(MB)';E={[math]::Round($_.Length/1MB,1)}} |
    Format-Table -AutoSize
