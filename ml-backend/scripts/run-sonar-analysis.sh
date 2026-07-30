#!/bin/bash
set -e

echo "🔍 Running SonarQube Analysis for PIOS ML Backend"

# Check if SonarQube is running
if ! docker ps | grep -q "sonarqube"; then
    echo "⚠️  Starting SonarQube infrastructure..."
    docker-compose -f docker-compose.sonarqube.yml up -d sonarqube sonar-db
    
    echo "⏳ Waiting for SonarQube to be ready (this may take 1-2 minutes)..."
    sleep 30
    
    # Wait for SonarQube to be operational
    until curl -s http://localhost:9000/api/system/status | grep -q '"status":"UP"'; do
        echo "   Still waiting..."
        sleep 10
    done
    echo "✅ SonarQube is operational!"
fi

# Run tests with coverage
echo "🧪 Running tests with coverage..."
pytest --cov=pios_ml_backend --cov-report=xml --cov-report=term-missing -v

# Run SonarScanner
echo "🔎 Running SonarScanner..."
docker-compose -f docker-compose.sonarqube.yml run --rm sonar-scanner \
    -Dsonar.projectKey=pios-ml-backend \
    -Dsonar.sources=pios_ml_backend \
    -Dsonar.host.url=http://sonarqube:9000 \
    -Dsonar.login=${SONAR_TOKEN:-admin} \
    -Dsonar.password=${SONAR_PASSWORD:-admin} \
    -Dsonar.python.coverage.reportPaths=coverage.xml

echo "✅ Analysis complete! View results at: http://localhost:9000/dashboard?id=pios-ml-backend"
