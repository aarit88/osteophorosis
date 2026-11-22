<#
.SYNOPSIS
    OsteoGuard AI - Safe Read-Only-Source Windows Checkpoint Recovery & BTXRD Verification Script.

.DESCRIPTION
    Searches the original training project directory, OneDrive folders, Downloads, and Desktop
    for .pt and .pth model checkpoints without modifying, moving, or deleting source files.
    Computes SHA-256 and exact byte sizes, copies candidates safely into C:\OsteoGuard-Recovery\,
    isolates and cryptographically verifies the Frozen BTXRD ResNet-34 checkpoint
    (244,849,703 bytes; SHA-256 21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD),
    and exports C:\OsteoGuard-Recovery\recovery_inventory.csv.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\ml_service\scripts\recover_checkpoints.ps1
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = "C:\Users\archi\OneDrive\Desktop\osteophorosis-main",
    [string]$RecoveryRoot = "C:\OsteoGuard-Recovery",
    [switch]$RestoreVerifiedToRepo
)

$ErrorActionPreference = "Stop"

$ExpectedBtxrdSha256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"
$ExpectedBtxrdSize   = 244849703L

$ExpectedRelativePaths = @(
    "ml_artifacts\fracatlas_full_training\best_checkpoint.pt",
    "ml_artifacts\fracatlas_full_training\last_checkpoint.pt",
    "ml_artifacts\fracatlas_pilot\best_checkpoint.pt",
    "ml_artifacts\fracatlas_pilot\last_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_0\best_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_0\last_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_1\best_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_1\last_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_2\best_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_2\last_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_3\best_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_3\last_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_4\best_checkpoint.pt",
    "ml_artifacts\bone_scan_cv\fold_4\last_checkpoint.pt",
    "ml_artifacts\bone_scan_pilot\best_checkpoint.pt",
    "ml_artifacts\bone_scan_pilot\last_checkpoint.pt",
    "ml_artifacts\btxrd_frozen_model\best_checkpoint.pt",
    "ml_artifacts\btxrd_full_training\best_checkpoint.pt",
    "ml_artifacts\btxrd_full_training\last_checkpoint.pt",
    "ml_artifacts\btxrd_pilot\best_checkpoint.pt",
    "ml_artifacts\btxrd_pilot\last_checkpoint.pt"
)

Write-Host "=== OsteoGuard AI Checkpoint Recovery Audit ===" -ForegroundColor Cyan
Write-Host "ProjectRoot : $ProjectRoot (Exists: $(Test-Path $ProjectRoot))"
Write-Host "RecoveryRoot: $RecoveryRoot"

$CandidatesDir    = Join-Path $RecoveryRoot "candidates"
$FrozenVerifiedDir = Join-Path $RecoveryRoot "verified_frozen_btxrd"
$QuarantineDir    = Join-Path $RecoveryRoot "quarantine_btxrd_mismatch"
New-Item -ItemType Directory -Force -Path $RecoveryRoot, $CandidatesDir, $FrozenVerifiedDir, $QuarantineDir | Out-Null

# Build deduplicated list of search roots (local project + OneDrive + Downloads + Desktop + Backups)
$SearchRoots = New-Object System.Collections.Generic.List[string]
foreach ($dir in @(
    $ProjectRoot,
    $env:OneDrive,
    $env:OneDriveConsumer,
    $env:OneDriveCommercial,
    (Join-Path $env:USERPROFILE "OneDrive"),
    (Join-Path $env:USERPROFILE "Downloads"),
    (Join-Path $env:USERPROFILE "Desktop"),
    (Join-Path $env:USERPROFILE "Documents")
)) {
    if (-not [string]::IsNullOrWhiteSpace($dir) -and (Test-Path $dir) -and -not $SearchRoots.Contains($dir)) {
        $SearchRoots.Add($dir)
    }
}

Write-Host "`nSearching roots for *.pt and *.pth files (skipping venv/site-packages/node_modules)..." -ForegroundColor Yellow
$Discovered = New-Object System.Collections.Generic.Dictionary[string, System.IO.FileInfo]

foreach ($root in $SearchRoots) {
    Get-ChildItem -Path $root -Recurse -File -Include *.pt, *.pth -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -notmatch "\\(\.cache|site-packages|node_modules|venvs|\.venv)\\"
        } |
        ForEach-Object {
            if (-not $Discovered.ContainsKey($_.FullName)) {
                $Discovered[$_.FullName] = $_
            }
        }
}

Write-Host "Discovered $($Discovered.Count) candidate .pt/.pth file(s)." -ForegroundColor Green

$InventoryRows = New-Object System.Collections.Generic.List[object]

foreach ($kvp in $Discovered.GetEnumerator()) {
    $file = $kvp.Value
    $sha256 = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash.ToUpper()
    $sizeBytes = [int64]$file.Length

    # Match against known relative path suffix if possible
    $matchedRel = ""
    foreach ($rel in $ExpectedRelativePaths) {
        if ($file.FullName.ToLower().EndsWith($rel.ToLower())) {
            $matchedRel = $rel
            break
        }
    }

    $isExactFrozenBtxrd = ($sizeBytes -eq $ExpectedBtxrdSize -and $sha256 -eq $ExpectedBtxrdSha256)
    $looksLikeBtxrd = ($file.FullName -match "btxrd" -and $file.Name -eq "best_checkpoint.pt" -and $file.FullName -notmatch "btxrd_pilot")

    if ($isExactFrozenBtxrd) {
        $status = "VERIFIED_FROZEN_BTXRD"
        $destFile = Join-Path $FrozenVerifiedDir "best_checkpoint.pt"
    } elseif ($looksLikeBtxrd) {
        $status = "QUARANTINED_BTXRD_HASH_OR_SIZE_MISMATCH"
        $destFile = Join-Path $QuarantineDir "$($sha256.Substring(0,12))_$($file.Name)"
    } else {
        $status = if ($matchedRel) { "CANDIDATE_MATCHED_PATH" } else { "CANDIDATE_UNMAPPED" }
        $subFolder = if ($matchedRel) { Split-Path $matchedRel -Parent } else { "unmapped\$($sha256.Substring(0,12))" }
        $destDir = Join-Path $CandidatesDir $subFolder
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        $destFile = Join-Path $destDir $file.Name
    }

    # Safe copy-only: never overwrite an existing file in recovery storage
    $copiedPath = ""
    if (-not (Test-Path $destFile)) {
        Copy-Item -LiteralPath $file.FullName -Destination $destFile -Force:$false
        $copiedPath = $destFile
    } else {
        $existingHash = (Get-FileHash -Path $destFile -Algorithm SHA256).Hash.ToUpper()
        if ($existingHash -eq $sha256) {
            $copiedPath = "$destFile (already identical)"
        } else {
            $altDest = "$($destFile).$($sha256.Substring(0,8))"
            if (-not (Test-Path $altDest)) {
                Copy-Item -LiteralPath $file.FullName -Destination $altDest -Force:$false
            }
            $copiedPath = $altDest
        }
    }

    # Optional safe restore into $ProjectRoot only when -RestoreVerifiedToRepo is passed and destination does not exist
    if ($RestoreVerifiedToRepo -and (Test-Path $ProjectRoot)) {
        if ($isExactFrozenBtxrd) {
            $repoFrozen = Join-Path $ProjectRoot "ml_artifacts\btxrd_frozen_model\best_checkpoint.pt"
            if (-not (Test-Path $repoFrozen)) {
                New-Item -ItemType Directory -Force -Path (Split-Path $repoFrozen -Parent) | Out-Null
                Copy-Item -LiteralPath $destFile -Destination $repoFrozen -Force:$false
            }
        } elseif ($matchedRel -and -not $looksLikeBtxrd) {
            $repoTarget = Join-Path $ProjectRoot $matchedRel
            if (-not (Test-Path $repoTarget)) {
                New-Item -ItemType Directory -Force -Path (Split-Path $repoTarget -Parent) | Out-Null
                Copy-Item -LiteralPath $file.FullName -Destination $repoTarget -Force:$false
            }
        }
    }

    $InventoryRows.Add([pscustomobject]@{
        SourcePath     = $file.FullName
        MatchedRelPath = $matchedRel
        FileName       = $file.Name
        SizeBytes      = $sizeBytes
        LastWriteUtc   = $file.LastWriteTimeUtc.ToString("o")
        SHA256         = $sha256
        IntegrityState = $status
        RecoveryCopy   = $copiedPath
    })
}

$CsvPath = Join-Path $RecoveryRoot "recovery_inventory.csv"
$InventoryRows | Export-Csv -Path $CsvPath -NoTypeInformation -Encoding UTF8

Write-Host "`nRecovery inventory written to: $CsvPath" -ForegroundColor Cyan
$InventoryRows | Format-Table FileName, SizeBytes, IntegrityState, MatchedRelPath -AutoSize
