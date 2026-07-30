import requests
import random
import string

BASE_URL = "http://localhost:8000/dashboard/add/"

def random_label():
    return ''.join(random.choices(string.ascii_uppercase, k=6))

# --- 10 "ideal" requests ---
print("=== Sending 'ideal' requests ===")
for i in range(10):
    payload = {
        "typeModel": "ideal",
        "jsonResponse": {
            "label": random_label(),
            "value": random.randint(1, 200)
        }
    }
    r = requests.post(BASE_URL, json=payload)
    print(f"[ideal #{i+1}] {payload['jsonResponse']} → {r.status_code} {r.text[:80]}")

# --- 10 "blood" requests ---
# print("\n=== Sending 'blood' requests ===")
# BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
# for i in range(10):
#     payload = {
#         "typeModel": "blood",
#         "jsonResponse": {
#             "bloodType": random.choice(BLOOD_TYPES),
#             "isNormal": random.choice([True, False]),
#             "isCritical": random.choice([True, False]),
#             "progress": random.randint(0, 100),
#             "value": random.randint(1, 200)
#         }
#     }
#     r = requests.post(BASE_URL, json=payload)
#     print(f"[blood #{i+1}] {payload['jsonResponse']} → {r.status_code} {r.text[:80]}")

# --- 10 "stats" requests ---
print("\n=== Sending 'stats' requests ===")
UNITS = ["par microlitre", "g/dL", "mg/L", "mmol/L", "%"]
LABELS = ["Hémoglobine", "Ferritine", "Protéines", "Plaquettes", "Hématocrite"]
for i in range(10):
    payload = {
        "typeModel": "stats",
        "jsonResponse": {
            "label": random.choice(LABELS),
            "value": random.randint(1, 200),
            "unit": random.choice(UNITS)
        }
    }
    r = requests.post(BASE_URL, json=payload)
    print(f"[stats #{i+1}] {payload['jsonResponse']} → {r.status_code} {r.text[:80]}")