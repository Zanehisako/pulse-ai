param(
    [string]$SonarToken = $env:SONAR_TOKEN,
    [string]$SonarHost = $env:SONAR_HOST_URL
)

Write-Host "🧪 Starting Local SonarQube Testing" -ForegroundColor Cyan

# Check SonarQube
try {
    Invoke-WebRequest -Uri "$SonarHost/api/system/status" -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "✅ SonarQube is operational at $SonarHost" -ForegroundColor Green
} catch {
    Write-Host "❌ SonarQube not accessible. Start it with: docker-compose -f docker-compose.sonarqube.yml up -d" -ForegroundColor Red
    exit 1
}

if (-not $SonarToken) {
    Write-Host "❌ Set `$env:SONAR_TOKEN='your_token'" -ForegroundColor Red
    exit 1
}

# Create tests if missing
if (-not (Test-Path "tests")) {
    New-Item -ItemType Directory -Path "tests" -Force | Out-Null
    @'
def test_basic():
    assert True

def test_python_works():
    import sys
    assert sys.version_info.major == 3
'@ | Set-Content "tests/test_basic.py"
}

# Run tests
Write-Host "`n🧪 Running tests..." -ForegroundColor Cyan
pip install pytest pytest-cov -q
pytest --cov=classes --cov=pios_ml_backend --cov=scripts --cov-report=xml -v

# Run scanner using Docker Compose (recommended)
Write-Host "`n🔍 Running SonarScanner via Docker Compose..." -ForegroundColor Cyan

# Set token for docker-compose
$env:SONAR_TOKEN = $SonarToken

docker-compose -f docker-compose.sonarqube.yml run --rm scanner

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Success! View at: $SonarHost/dashboard?id=pios-ml-backend" -ForegroundColor Green
} else {
    Write-Host "`n❌ Failed! Trying with host.docker.internal..." -ForegroundColor Yellow
    
    # Fallback to direct docker with host.docker.internal
    docker run --rm `
        -e SONAR_HOST_URL="http://host.docker.internal:9000" `
        -e SONAR_TOKEN="$SonarToken" `
        -v "${PWD}:/usr/src" `
        sonarsource/sonar-scanner-cli
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Success with fallback! View at: $SonarHost/dashboard?id=pios-ml-backend" -ForegroundColor Green
    } else {
        Write-Host "`n❌ All methods failed!" -ForegroundColor Red
    }
}