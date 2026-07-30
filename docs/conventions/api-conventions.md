# File: docs/conventions/api-conventions.md

# Conventions d'API

## 1) REST API

### 1.1 Versioning
```text
/api/v1/...
```
Breaking change → nouvelle version (/api/v2/...).
### 1.2 Nommage des routes

Ressources au pluriel : /users, /orders

ID en path : /users/{id}

Actions rares : /users/{id}/activate

### 1.3 Exemples
GET    /api/v1/users
POST   /api/v1/users
GET    /api/v1/users/{id}
PUT    /api/v1/users/{id}
DELETE /api/v1/users/{id}

## 2) Standards de réponse

### 2.1 Succès
{
  "data": {},
  "meta": {
    "requestId": "string",
    "timestamp": "2026-01-18T10:30:00Z"
  },
  "errors": []
}

### 2.2 Erreur (standard)
{
  "data": null,
  "meta": {
    "requestId": "string",
    "timestamp": "2026-01-18T10:30:00Z"
  },
  "errors": [
    {
      "code": "VALIDATION_ERROR",
      "message": "Invalid input",
      "details": [
        { "field": "email", "reason": "Invalid format" }
      ]
    }
  ]
}

## 3) Pagination / tri / filtres
?page=1&pageSize=25
?sort=createdAt:desc
?status=active&country=CA

## 4) Codes HTTP

200 OK

201 Created

204 No Content

400 Bad Request

401 Unauthorized

403 Forbidden

404 Not Found

409 Conflict

422 Validation Error

500 Internal Server Error

## 5) Auth / sécurité

Bearer token :

Authorization: Bearer <token>

Validation input systématique (payload, query, params)

Secrets jamais en repo ni dans les logs

Rate limiting sur endpoints sensibles (login, OTP)

## 6) GraphQL (si applicable)
type Query {
  user(id: ID!): User
  users(filter: UserFilter): [User]!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
}

## 7) WebSocket / Events

Format : namespace:event

user:connected
user:disconnected
chat:message
notification:new

## 8) Documentation

OpenAPI/Swagger obligatoire pour REST

Inclure exemples success + error

Endpoint recommandé :

/api/docs
