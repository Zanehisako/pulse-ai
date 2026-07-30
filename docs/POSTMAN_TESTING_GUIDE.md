# 🧪 Testing Alerts System in Postman

## Complete Step-by-Step Guide

---

## 📋 **Prerequisites**

1. **Backend running:**

   ```powershell
   cd D:\Projects\pios\backendMulti
   python manage.py runserver
   ```

2. **Database migrated:**

   ```powershell
   python manage.py migrate
   ```

3. **Postman installed** with WebSocket support (v10+)

---

## 🎯 **Testing Flow Overview**

```
1. Create an Alert Rule (REST API)
   ↓
2. Connect to WebSocket (listen for alerts)
   ↓
3. Trigger Alert (via prediction or manual test-fire)
   ↓
4. Watch WebSocket receive alert in real-time
   ↓
5. Query alerts, acknowledge, resolve (REST API)
```

---

## 📝 **STEP 1: Create an Alert Rule**

### **Request:**

```http
POST http://127.0.0.1:8000/alerts/rules/
Content-Type: application/json
```

### **Body (JSON):**

```json
{
  "name": "Low Stock Critical Alert for hospitals",
  "description": "Trigger when stock is low and predicted stockout is imminent",
  "is_active": true,
  "scope_type": "global",
  "scope_ref": "",
  "severity_base": "critical",
  "trigger_type": "stock_and_forecast",
  "conditions": {
    "all": [
      {
        "field": "current_stock_units",
        "op": "<=",
        "value": 30
      },
      {
        "field": "prediction",
        "op": "<=",
        "value": 6
      }
    ]
  },
  "channels": ["websocket", "in_app"],
  "escalation_after_minutes": 120,
  "dedup_window_minutes": 60
}
```

### **Expected Response (201 Created):**

```json
{
  "id": 8,
  "name": "Low Stock Critical Alert for hospitals",
  "description": "Trigger when stock is low and predicted stockout is imminent",
  "is_active": true,
  "scope_type": "global",
  "scope_ref": "",
  "severity_base": "critical",
  "trigger_type": "stock_and_forecast",
  "conditions": {
    "all": [
      {
        "field": "current_stock_units",
        "op": "<=",
        "value": 30
      },
      {
        "field": "prediction",
        "op": "<=",
        "value": 6
      }
    ]
  },
  "channels": ["websocket", "in_app"],
  "escalation_after_minutes": 120,
  "dedup_window_minutes": 60,
  "created_by": "admin",
  "created_at": "2026-03-28T20:36:55.244783Z",
  "updated_at": "2026-03-28T20:36:55.244783Z"
}
```

### **Postman Settings:**

- **Method:** `POST`
- **URL:** `http://127.0.0.1:8000/alerts/rules/`
- **Headers:**
  - `Content-Type: application/json`
- **Body:** Raw → JSON → paste the JSON above

---

## 🔌 **STEP 2: Connect to WebSocket (Alerts Channel)**

### **In Postman:**

1. **Create New Request** → Select **WebSocket** type
2. **URL:** `ws://127.0.0.1:8000/ws/alerts/`
3. **Click "Connect"**

### **Expected Connection Message:**

```json
{
  "type": "connection",
  "status": "connected",
  "message": "Alerts WebSocket connected successfully"
}
```

### **⚠️ IMPORTANT:**

- The trailing slash `/` is **required**: `ws://127.0.0.1:8000/ws/alerts/`
- Keep this WebSocket connection **open** to receive alerts in real-time

### **Screenshot Guide:**

```
┌─────────────────────────────────────────────────┐
│ Postman - New WebSocket Request                 │
├─────────────────────────────────────────────────┤
│  URL: ws://127.0.0.1:8000/ws/alerts/           │
│  [Connect]                                      │
├─────────────────────────────────────────────────┤
│ Messages:                                       │
│  ← {                                            │
│      "type": "connection",                      │
│      "status": "connected",                     │
│      "message": "Alerts WebSocket connected..." │
│    }                                            │
└─────────────────────────────────────────────────┘
```

---

## 🤖 **STEP 3B: Trigger Alert via ML Prediction**

### **Request:**

```http
POST http://127.0.0.1:8000/api/ml/models/{model_id}/predict/
Content-Type: application/json
```

### **Body (JSON) - Matching the Rule:**

test with this model_id "stockout_days_predictor_champion"

```json
{
  "inputs": [
    {
      "stockout_features_features__blood_product_type": "O-",
      "stockout_features_features__current_stock_units": 25,
      "stockout_features_features__usage_today": 4,
      "stockout_features_features__lead_time_days": 2,
      "stockout_features_features__days_since_last_restock": 0,
      "stockout_features_features__stockout_count_90d": 2,
      "stockout_features_features__scheduled_surgeries_next7d": 2
    }
  ]
}
```

### **Expected Prediction Response:**

```json
{
  "success": true,
  "query": "Run model stockout_days_predictor_champion with these provided feature values.",
  "planner_mode": "direct",
  "plan": {
    "reasoning": "Direct single-model prediction path; xLAM planning skipped.",
    "steps": [
      {
        "tool": "stockout_days_predictor_champion",
        "arguments": {
          "stockout_features_features__blood_product_type": "O-",
          "stockout_features_features__current_stock_units": 25,
          "stockout_features_features__usage_today": 4,
          "stockout_features_features__lead_time_days": 2,
          "stockout_features_features__days_since_last_restock": 0,
          "stockout_features_features__stockout_count_90d": 2,
          "stockout_features_features__scheduled_surgeries_next7d": 2
        }
      }
    ]
  },
  "execution_results": [
    {
      "tool": "stockout_days_predictor_champion",
      "output": {
        "model_id": "stockout_days_predictor_champion",
        "aliases": [
          "stockout_days_predictor_champion",
          "stockout_days_predictor"
        ],
        "model_type": "mlflow",
        "file_path": "models:\\stockout_days_predictor@champion",
        "prediction": 4.935938835144043,
        "used_inputs": {
          "stockout_features_features__blood_product_type": "O-",
          "stockout_features_features__current_stock_units": 25.0,
          "stockout_features_features__usage_today": 4.0,
          "stockout_features_features__lead_time_days": 2,
          "stockout_features_features__days_since_last_restock": 0,
          "stockout_features_features__stockout_count_90d": 2,
          "stockout_features_features__scheduled_surgeries_next7d": 2
        },
        "missing_features": []
      },
      "success": true
    }
  ],
  "natural_language_response": "stockout_days_predictor_champion: 4.935938835144043",
  "verification_data": ["stockout_days_predictor_champion: 4.935938835144043"]
}
```

### **🔔 WebSocket Receives:**

```json
{
  "type": "alert_event",
  "id": "660e8400-e29b-41d4-a716-446655440000",
  "rule_id": 1,
  "source_type": "prediction",
  "source_ref": "stockout-predictor",
  "event_key": "1|H005||stockout-predictor",
  "severity": "critical",
  "status": "open",
  "title": "Low Stock Critical Alert for H005 (O+)",
  "message": "Trigger when stock is low and predicted stockout is imminent | predicted=1.5 | current_stock=5",
  "context": {
    "model_id": "stockout-predictor",
    "source": "predict-model",
    "current_stock_units": 5,
    "blood_product_type": "O+",
    "hospital_id": "H005",
    "predicted_value": 1.5,
    "entity_id": "H005",
    "blood_type": "O+"
  },
  "entity_type": "hospital",
  "entity_id": "H005",
  "blood_type": "O+",
  "predicted_value": 1.5,
  "opened_at": "2026-03-28T20:15:00.000Z",
  "notification_id": "670e8400-e29b-41d4-a716-446655440001"
}
```

### **✅ What Just Happened:**

1. You made a prediction with low stock (25) and low predicted value (4.9)
2. Both condition in the rule matched:
   - `prediction` (4.9) <= 6 ✅
3. Alert engine created an alert
4. WebSocket instantly received it

---

## 🔥 **STEP 3A: Trigger Alert via Manual Test-Fire**

### **Request:**

```http
POST http://127.0.0.1:8000/alerts/test-fire/
Content-Type: application/json
```

### **Body (JSON):**

```json
{
  "title": "Critical O- shortage risk",
  "message": "Hospital H005 may run out of O- within 2 days. Current stock: 6 units, Predicted days until stockout: 2",
  "severity": "critical",
  "status": "open",
  "entity_type": "hospital",
  "entity_id": "H005",
  "blood_type": "O-",
  "context": {
    "current_stock_units": 30,
    "predicted_days_until_stockout": 2,
    "prediction": 4.9
  }
}
```

### **Expected Response (201 Created):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "rule_id": null,
  "source_type": "manual",
  "source_ref": "test-fire",
  "event_key": "test-fire-1711657200",
  "severity": "critical",
  "status": "open",
  "title": "Critical O- shortage risk",
  "message": "Hospital H005 may run out of O- within 2 days...",
  "context": {
    "current_stock_units": 300,
    "predicted_days_until_stockout": 2,
    "prediction": 4.9
  },
  "entity_type": "hospital",
  "entity_id": "H005",
  "blood_type": "O-",
  "prediction": 1.5,
  "actual_value": null,
  "threshold_value": null,
  "opened_at": "2026-03-28T20:10:00.000Z",
  "last_evaluated_at": "2026-03-28T20:10:00.000Z",
  "acknowledged_at": null,
  "escalated_at": null,
  "resolved_at": null,
  "notification_id": "650e8400-e29b-41d4-a716-446655440001"
}
```

### **🔔 WebSocket Receives (in your WebSocket tab):**

```json
{
  "type": "alert_event",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "rule_id": null,
  "source_type": "manual",
  "source_ref": "test-fire",
  "event_key": "test-fire-1711657200",
  "severity": "critical",
  "status": "open",
  "title": "Critical O- shortage risk",
  "message": "Hospital H005 may run out of O- within 2 days...",
  "context": {
    "current_stock_units": 20,
    "predicted_days_until_stockout": 2,
    "predicted_value": 1.5
  },
  "entity_type": "hospital",
  "entity_id": "H005",
  "blood_type": "O-",
  "predicted_value": 1.5,
  "opened_at": "2026-03-28T20:10:00.000Z",
  "notification_id": "650e8400-e29b-41d4-a716-446655440001"
}
```

### **✅ What Just Happened:**

1. You sent a POST request to `/alerts/test-fire/`
2. Backend created an `AlertEvent` in the database
3. Backend broadcasted to WebSocket group `"alerts"`
4. Your connected WebSocket received the alert **instantly**
5. Backend also mirrored to `/ws/notifications/` group
