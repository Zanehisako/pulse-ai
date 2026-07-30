This guide explains how to set up and run SonarQube code quality analysis locally for the PIOS ML Backend project.

## Prerequisites

- Docker Desktop installed and running
- Python 3.11+ with virtual environment
- PowerShell (Windows) or Terminal (Mac/Linux)
- Git repository cloned locally

## Quick Start (5 minutes)

### 1. Start SonarQube Infrastructure

From the project root (`ml-backend` folder):

```powershell
# Start SonarQube and PostgreSQL
docker-compose -f docker-compose.sonarqube.yml up -d

# Wait 1-2 minutes for SonarQube to initialize
# Check status: http://localhost:9000
```

### 2. Initial Setup (First Time Only)

1. Open http://localhost:9000 in your browser
2. Login with default credentials:
   - **Username**: `admin`
   - **Password**: `admin`
3. Change the password when prompted
4. Create a project:
   - Click **"Create a local project"**
   - Project Key: `pios-ml-backend`
   - Display Name: `PIOS ML Backend`
   - Click **Set Up**

### 3. Generate Authentication Token

1. Click your profile (top right) → **My Account** → **Security**
2. Under **Tokens**, click **Generate Token**
3. Name: `local-dev-token`
4. Type: **User token**
5. Copy the token (starts with `squ_...`)

### 4. Set Environment Variables

In PowerShell:

```powershell
$env:SONAR_TOKEN="squ_your_token_here"
$env:SONAR_HOST_URL="http://localhost:9000"
```

Or add to `.env` file in project root:

```env
SONAR_TOKEN=squ_your_token_here
SONAR_HOST_URL=http://localhost:9000
```

### 5. Run Analysis

Use the provided script:

```powershell
.\scripts\test-local-docker.ps1
```

### 6. View Results

Open http://localhost:9000/dashboard?id=pios-ml-backend

---

## Understanding the Metrics

| Metric                | Good | Bad  | Current Status           |
| --------------------- | ---- | ---- | ------------------------ |
| **Security**          | A    | E    | ✅ **A** (2 issues)      |
| **Reliability**       | A    | E    | ⚠️ **C** (10 bugs)       |
| **Maintainability**   | A    | E    | ✅ **A** (60 smells)     |
| **Coverage**          | >80% | <50% | 🔴 **0%** (CRITICAL)     |
| **Duplications**      | <3%  | >10% | 🟠 **34.7%** (HIGH)      |
| **Security Hotspots** | 0    | >5   | 🔴 **6** (Review needed) |
