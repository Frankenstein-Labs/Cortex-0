# API HTTP

Lancer le serveur :

```bash
cortex serve --model-dir checkpoints/tiny-cpu --port 8000
# ou
uvicorn cortex.api.app:app --host 127.0.0.1 --port 8000
```

Documentation interactive : <http://127.0.0.1:8000/docs>.

## Authentification

Si `CORTEX_API_KEY` est défini, toutes les routes `/v1/*` exigent l'en-tête :

```http
x-api-key: <clé>
```

`/health` reste public. Sans clé configurée, le serveur doit rester sur
`127.0.0.1`.

## Erreurs

Format uniforme :

```json
{"detail": "message"}
```

Codes : `400` entrée invalide, `401` clé absente ou fausse, `404` ressource
inconnue, `409` conflit d'état, `503` modèle non chargé.

## Santé et modèle

```bash
curl localhost:8000/health
curl localhost:8000/v1/model/info
```

```json
{
  "loaded": true,
  "parameters": 2557632,
  "context_length": 256,
  "vocab_size": 4096,
  "device": "cpu",
  "model_dir": "checkpoints/tiny-cpu"
}
```

## Génération

```bash
curl -X POST localhost:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"CORTEX est","max_new_tokens":40,"temperature":0.8,"top_p":0.95}'
```

```json
{
  "text": "...",
  "prompt_tokens": 4,
  "generated_tokens": 40,
  "model": {"loaded": true, "parameters": 2557632, "context_length": 256}
}
```

Streaming SSE :

```bash
curl -N -X POST localhost:8000/v1/generate/stream \
  -H 'content-type: application/json' \
  -d '{"prompt":"CORTEX est","max_new_tokens":20}'
```

Chaque événement : `data: {"token": "...", "index": 0}` puis
`data: {"done": true}`.

## Conversation

```bash
curl -X POST localhost:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo","message":"Bonjour CORTEX"}'
```

```json
{"session_id":"demo","text":"...","used_model":true}
```

`used_model: false` signifie que le serveur a répondu avec un fallback
déterministe parce qu'aucun modèle n'était chargé. Ce n'est pas une réponse du
réseau.

`POST /v1/chat/stream` renvoie des deltas SSE.

## Raisonnement

```bash
curl -X POST localhost:8000/v1/reason \
  -H 'content-type: application/json' \
  -d '{"question":"Combien font 12 * 7 ?","expected":"84","num_samples":4}'
```

La réponse contient `answer`, `verified`, `confidence`, `candidates` et
`revisions`. `verified` vient d'un vérificateur programmatique, pas d'une
auto-déclaration du modèle.

## Mémoire

```bash
curl -X POST localhost:8000/v1/memory \
  -H 'content-type: application/json' \
  -d '{"content":"Décision : utiliser SQLite","kind":"project"}'

curl 'localhost:8000/v1/memory/search?q=SQLite&limit=5'
curl 'localhost:8000/v1/memory/recent?kind=project&limit=10'
curl -X POST localhost:8000/v1/memory/consolidate \
  -H 'content-type: application/json' -d '{"session_id":"demo"}'
```

Types : `working`, `short_term`, `long_term`, `episodic`, `semantic`,
`project`, `preference`.

## Objectifs

```bash
curl -X POST localhost:8000/v1/goals \
  -H 'content-type: application/json' \
  -d '{"title":"Entraîner le modèle 50M","priority":0.9}'

curl localhost:8000/v1/goals
curl -X PATCH localhost:8000/v1/goals/1 \
  -H 'content-type: application/json' -d '{"progress":0.4}'
```

## Missions

```bash
curl -X POST localhost:8000/v1/missions \
  -H 'content-type: application/json' \
  -d '{"title":"Auth","instruction":"Ajouter une authentification"}'

curl -X POST localhost:8000/v1/missions/<id>/run \
  -H 'content-type: application/json' -d '{"max_steps":3}'

curl -X POST localhost:8000/v1/missions/<id>/instruction \
  -H 'content-type: application/json' -d '{"text":"Utiliser OAuth"}'

curl -X POST localhost:8000/v1/missions/<id>/approve
curl -X POST localhost:8000/v1/missions/<id>/reject
```

Une mission sans modèle renvoie `503`.

## File de tâches

```bash
curl -X POST localhost:8000/v1/tasks \
  -H 'content-type: application/json' \
  -d '{"kind":"echo","payload":{"x":1},"priority":5}'

curl localhost:8000/v1/tasks/stats
curl -X POST localhost:8000/v1/tasks/<id>/cancel
```

## Initiatives

```bash
curl -X POST localhost:8000/v1/initiatives/tick
curl 'localhost:8000/v1/initiatives?status=pending'
curl -X POST localhost:8000/v1/initiatives/<id>/dismiss
curl -X POST localhost:8000/v1/initiatives/<id>/send
curl -N localhost:8000/v1/events
```

`/v1/events` est un flux SSE des initiatives en attente.

## Outils

```bash
curl localhost:8000/v1/tools
```

Renvoie le registre d'outils avec leur niveau de risque.
