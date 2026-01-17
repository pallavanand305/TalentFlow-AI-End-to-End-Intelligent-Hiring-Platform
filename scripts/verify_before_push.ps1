# Security verification script for Windows - Run before pushing to GitHub

Write-Host "TalentFlow AI - Security Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$Errors = 0
$Warnings = 0

function Print-Status {
    param($Success, $Message)
    if ($Success) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] $Message" -ForegroundColor Red
        $script:Errors++
    }
}

function Print-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
    $script:Warnings++
}

Write-Host "1. Checking .gitignore..."
if (Select-String -Path .gitignore -Pattern "^\.env$" -Quiet) {
    Print-Status $true ".env is in .gitignore"
} else {
    Print-Status $false ".env is NOT in .gitignore - CRITICAL!"
}

Write-Host ""
Write-Host "2. Checking if .env is tracked..."
$envTracked = git ls-files | Select-String -Pattern "^\.env$" -Quiet
if ($envTracked) {
    Print-Status $false ".env is tracked in git - REMOVE IT!"
} else {
    Print-Status $true ".env is not tracked"
}

Write-Host ""
Write-Host "3. Checking for AWS credentials..."
$awsFiles = git ls-files | Select-String -Pattern "\.aws|credentials|\.pem$|\.key$" -Quiet
if ($awsFiles) {
    Print-Status $false "AWS credential files found in git!"
} else {
    Print-Status $true "No AWS credential files tracked"
}

Write-Host ""
Write-Host "4. Checking for database files..."
$dbFiles = git ls-files | Select-String -Pattern "\.db$|\.sqlite$" -Quiet
if ($dbFiles) {
    Print-Status $false "Database files tracked in git!"
} else {
    Print-Status $true "No database files tracked"
}

Write-Host ""
Write-Host "5. Checking for large files..."
$largeFiles = Get-ChildItem -Recurse -File | Where-Object { 
    $_.Length -gt 10MB -and 
    $_.FullName -notmatch "\.git|venv|env|node_modules" 
}
if ($largeFiles) {
    Print-Warning "Found $($largeFiles.Count) large files (>10MB)"
    $largeFiles | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}} | Format-Table
} else {
    Print-Status $true "No large files found"
}

Write-Host ""
Write-Host "6. Checking .env.example..."
if (Test-Path .env.example) {
    $hasPlaceholders = Select-String -Path .env.example -Pattern "your-secret-key|change-in-production|example|placeholder" -Quiet
    if ($hasPlaceholders) {
        Print-Status $true ".env.example contains placeholder values"
    } else {
        Print-Warning ".env.example might contain real values - verify manually"
    }
} else {
    Print-Status $false ".env.example not found!"
}

Write-Host ""
Write-Host "7. Verifying important files exist..."
$requiredFiles = @("README.md", "requirements.txt", "docker-compose.yml", ".gitignore")
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Print-Status $true "$file exists"
    } else {
        Print-Status $false "$file is missing!"
    }
}

Write-Host ""
Write-Host "8. Checking git status..."
$untracked = (git status --porcelain | Select-String -Pattern "^\?\?" | Measure-Object).Count
if ($untracked -gt 0) {
    Print-Warning "$untracked untracked files - review before committing"
    git status --porcelain | Select-String -Pattern "^\?\?" | Select-Object -First 5
}

Write-Host ""
Write-Host "========================================"
Write-Host "Summary:"
Write-Host "  Errors: $Errors"
Write-Host "  Warnings: $Warnings"
Write-Host ""

if ($Errors -gt 0) {
    Write-Host "[FAILED] Fix errors before pushing!" -ForegroundColor Red
    exit 1
} elseif ($Warnings -gt 0) {
    Write-Host "[WARNING] Review warnings before pushing" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/N)"
    if ($response -ne "y" -and $response -ne "Y") {
        exit 1
    }
}

Write-Host "[SUCCESS] All checks passed! Safe to push." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  git add ."
Write-Host "  git commit -m `"Initial commit: TalentFlow AI`""
Write-Host "  git push -u origin main"
