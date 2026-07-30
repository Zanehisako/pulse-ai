# Dashboard Module

Real-time dashboard module for the PIOS backend application. Handles model responses with WebSocket support for live updates.

## Overview

The dashboard module manages dashboard data through a REST API and provides real-time WebSocket updates to connected clients. It supports filtering by region, product, period, and alert level.

## Features

- **REST API endpoints** for creating and retrieving dashboard data
- **WebSocket consumer** for real-time dashboard updates
- **Flexible filtering** by region, product, period, and alert level
- **Separate endpoints** for mobile and web applications
- **Automatic signal handling** for database operations

## Models

### ModelResponse

Stores dashboard model responses with metadata and filtering options.

**Fields:**

- `id` - Primary key
- `typeModel` - Type of model (e.g., "donations")
- `jsonResponse` - JSON data containing model response
- `region` - Region choice (Quebec, Montreal)
- `product` - Product choice (Blood)
- `period` - Time period choice (24h, 2d, 1w)
- `alert_level` - Alert level choice (Low, Medium, High)
- `startDate` - Auto-set creation timestamp
- `endDate` - Optional end timestamp (null for active rows)

## API Endpoints

### POST `/dashboard/add/`

**Mobile application endpoint**

Creates a new dashboard row with filter fields. Does not send WebSocket updates.

**Request body:**

```json
{
  "typeModel": "donations",
  "region": "montreal",
  "product": "blood",
  "period": "24h",
  "alert_level": "high",
  "jsonResponse": {
    "data": [...]
  }
}
```

### POST `/dashboard/add-web/`

**Web application endpoint**

Creates a new dashboard row and broadcasts it to all connected WebSocket clients.

**Request body:** Same as `/dashboard/add/`

**Response:**

```json
{
  "status": "success",
  "row": {
    "id": 1,
    "typeModel": "donations",
    "region": "montreal",
    "product": "blood",
    "period": "24h",
    "alert_level": "high",
    "jsonResponse": {...},
    "startDate": "2026-03-31T17:30:41Z",
    "endDate": null
  }
}
```

### GET `/dashboard/latest/`

Retrieves the latest dashboard data for specified filters.

**Query parameters:**

- `region` - Filter by region (default: quebec)
- `product` - Filter by product (default: blood)
- `period` - Filter by period (default: 24h)
- `alert_level` - Filter by alert level (default: low)

**Response:**

```json
{
  "snapshots": {
    "snapshot_key": {
      "id": 1,
      "typeModel": "donations",
      "startDate": "2026-03-31T17:30:41Z",
      ...
    }
  }
}
```

## WebSocket

### Consumer: DashboardConsumer

Real-time updates via WebSocket connection.

**Connection:**

- Connect to `ws://localhost:8000/ws/dashboard/`
- Automatically joins `dashboard_updates` group
- Receives all active rows (endDate is null) upon connection

**Events received:**

```json
{
  "id": 1,
  "typeModel": "donations",
  "jsonResponse": {...},
  "region": "montreal",
  "product": "blood",
  "period": "24h",
  "alert_level": "high",
  "startDate": "2026-03-31T17:30:41Z",
  "endDate": null
}
```

## Signals

Automatic signal handlers for database operations and WebSocket broadcasts.

See `signals.py` for implementation details.

## Scripts

### `seed-web-dashboard.ps1`

PowerShell script to seed the dashboard with sample data. Run from the repository root:

```powershell
.\dashboard\scripts\seed-web-dashboard.ps1
```

## Usage Example

**Create a new dashboard row (Web):**

```bash
curl -X POST "http://localhost:8000/dashboard/add-web/" \
  -H "Content-Type: application/json" \
  -d '{
    "typeModel": "donations",
    "region": "montreal",
    "product": "blood",
    "period": "24h",
    "alert_level": "high",
    "jsonResponse": {
      "data": [
        {"date": "2026-01-01", "value": 10},
        {"date": "2026-01-02", "value": 20}
      ]
    }
  }'
```

**Retrieve latest data:**

```bash
curl "http://localhost:8000/dashboard/latest/?region=montreal&alert_level=high"
```

**Connect to WebSocket:**

```javascript
const socket = new WebSocket("ws://localhost:8000/ws/dashboard/");
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Dashboard update:", data);
};
```

## File Structure

- `models.py` - Database models
- `consumers.py` - WebSocket consumer
- `views.py` - API views
- `urls.py` - URL routing
- `signals.py` - Django signals for automations
- `routing.py` - WebSocket routing configuration
- `admin.py` - Django admin configuration
- `tests.py` - Unit tests
