# File: docs/conventions/naming-conventions.md

# Conventions de Nommage

## 1) Généralités
- `camelCase` : variables, fonctions
- `PascalCase` : classes, types, composants
- `kebab-case` : dossiers/fichiers (sauf si standard `PascalCase.tsx` côté UI)
- `SNAKE_CASE` : constantes, enums
- Préfixes recommandés : `is`, `has`, `can`, `should`, `fetch`, `build`, `map`, `to`

---

## 2) Web (Frontend)

### Composants
- Composants : `PascalCase`
- Props/state : `camelCase`
- Hooks : `useSomething`

Exemple :
```ts
// src/components/UserProfileCard.tsx
export function UserProfileCard() {
  return null;
}

// Hook
export const useUserData = () => {};

Services / API clients
// src/services/user-api.service.ts
export class UserApiService {
  async fetchUserProfile(userId: string) {}
}
```
## 3) Mobile (React Native)
Components / Screens

Components : PascalCase.tsx

Screens : SomethingScreen.tsx

Structure recommandée :

screens/
├── auth/
│   ├── LoginScreen.tsx
│   └── RegisterScreen.tsx
└── main/
    └── HomeScreen.tsx

## 4) IA / Data Science
Scripts / pipelines

Python : snake_case

# model_training_pipeline.py
```
class CustomerChurnPredictor:
    def train(self, data):
        pass
```
Notebooks
```
notebooks/
├── 01_data_exploration.ipynb
├── 02_feature_engineering.ipynb
└── 03_model_training.ipynb
```
Datasets
```
- Préfixes : raw_, processed_, cleaned_
- raw_customer_data.csv
- processed_training_data.parquet
- cleaned_features.parquet
```
## 5) API
REST endpoints

Ressources au pluriel

IDs en path

Exemples :

- GET    /api/v1/users
- POST   /api/v1/users
- GET    /api/v1/users/{id}
- PUT    /api/v1/users/{id}
- DELETE /api/v1/users/{id}

WebSocket events

Format : namespace:event

- user:connected
- user:disconnected
- chat:message
- notification:new
