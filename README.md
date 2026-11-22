# CORTEX-0

**Small Autonomous Language Model — Frankenstein-Labs**

CORTEX-0 est une petite LLM construite de zéro : Transformer décodeur,
tokenizer BPE byte-level, préentraînement autoregressif, SFT, évaluation,
raisonnement vérifié, mémoire, initiative et agent de codage. Le code est réel
et testé sur CPU. Les poids ne sont pas publiés dans ce dépôt.

> CORTEX-0 est un projet de recherche précoce, pas un assistant généraliste.
> Le modèle de référence fait ~2,56 M de paramètres. La reconstruction 2026
> utilise un corpus français v2 de 10,3 M de tokens, un tokenizer 8192 et un
> témoin de 3,34 M de paramètres. Toutes les métriques sont mesurées.

## État vérifié

| Élément | État |
| --- | --- |
| Architecture Llama-compatible | testée (causalité, KV-cache, GQA, RoPE) |
| Tokenizer BPE byte-level | aller-retour exact, 4096 et 8192 |
| Préentraînement CPU | 3 000 steps (ancien), 1 500 steps (v2) |
| Export Hugging Face | logits identiques (`max_logit_diff = 0.0`) |
| Tests automatisés | 117 tests (`pytest -q`) |
| Mémoire SQLite + FTS5 | stockage, recherche, consolidation |
| Raisonnement + vérificateur | best-of-N, révision, GRPO |
| Initiative proactive | policy, scheduler, déduplication |
| Agent de codage | sandbox, approbations, missions persistantes |
| API FastAPI | chat, streaming SSE, mémoire, missions |

## Reconstruction 2026 — corpus v2

| Élément | Résultat mesuré |
|---|---|
| Corpus v2 | 10 333 724 tokens train, 574 473 val, 611 376 test |
| Sources | 7 livres FR domaine public, Wikipédia FR CC BY-SA, textes originaux, code |
| Tokenizer | BPE byte-level 8192, entraîné sur le train seulement |
| Modèle témoin | 3 344 064 paramètres, 1 500 steps, 6,14 M tokens vus |
| Bits/caractère sur texte test commun | 1,951 (ancien : 2,020) |
| Complétion `cortex-100-fr` | 88 % (ancien : 19 %) |
| Meilleur SFT | 12/100 (10 refus, 1 math, 1 anti-répétition) |

Le résultat est **négatif pour les compétences** : le modèle termine mieux ses
réponses et modélise mieux le français, mais n'apprend ni les instructions, ni
le code, ni l'arithmétique. Les traces arithmétiques de SFT v8 donnent une
validation trompeuse (0,848) et **0/80** en généralisation. La prochaine étape
demande un tokenizer avec chiffres séparés, plus de données et un modèle 50 M
sur GPU.

Détails :
[`docs/PRETRAIN_AUDIT.md`](docs/PRETRAIN_AUDIT.md),
[`docs/CORPUS_V2.md`](docs/CORPUS_V2.md),
[`docs/SFT_V6.md`](docs/SFT_V6.md),
[`docs/EXPERIMENTS_V2.md`](docs/EXPERIMENTS_V2.md).

## Évaluation `cortex-100-fr`

100 questions françaises inédites, génération gloutonne, 128 tokens maximum.

| Modèle | Correct / 100 | Refus | Python | Tronqué |
|---|---:|---:|---:|---:|
| base 2,56 M | 0 | 0/10 | 0/15 | 66 % |
| SFT v3 plain 2,56 M | 9 | 9/10 | 0/15 | 0 % |
| base 15 M (1 500 steps) | 1 | 0/10 | 0/15 | 18 % |
| base v2 3,34 M (1 500 steps) | 0 | 0/10 | 0/15 | 12 % |
| SFT v6 3,34 M | 10 | 9/10 | 0/15 | 0 % |
| SFT v7 3,34 M | 12 | 10/10 | 0/15 | 9 % |
| SFT v8 3,34 M + traces | 11 | 10/10 | 0/15 | 0 % |

Aucune compétence en compréhension, grammaire, vocabulaire, résumé,
mathématiques, Python, sciences ou informatique n'est mesurable de façon
fiable. Rapport complet : [`docs/EVAL_100.md`](docs/EVAL_100.md).

## Démarrage rapide

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"

python scripts/prepare_data.py --out-dir datasets/processed \
  --vocab-size 4096 --val-ratio 0.02 --test-ratio 0.01
python scripts/train.py --config configs/tiny-cpu.yaml
python scripts/evaluate.py --model-dir checkpoints/tiny-cpu \
  --val-data datasets/processed/val.bin --n 20
python scripts/generate.py --model-dir checkpoints/tiny-cpu \
  --prompt "CORTEX est" --max-new-tokens 60
python scripts/serve.py --model-dir checkpoints/tiny-cpu --port 8000
```

CLI unifiée :

```bash
cortex train --config configs/tiny-cpu.yaml
cortex chat --model-dir checkpoints/tiny-cpu
cortex serve --model-dir checkpoints/tiny-cpu --scheduler
cortex evaluate --model-dir checkpoints/tiny-cpu
cortex export-hf --model-dir checkpoints/tiny-cpu --out export/Cortex-0 --verify
```

## Architecture

```text
cortex-0/
├── cortex/
│   ├── config.py
│   ├── model/          # RMSNorm, RoPE, GQA, SwiGLU, Transformer, sampling
│   ├── tokenizer/      # BPE byte-level + template de chat CORTEX
│   ├── training/       # données, trainer, SFT, corpus
│   ├── evaluation/     # perplexité, cortex-100-fr
│   ├── reasoning/      # vérificateur AST, best-of-N, révision, GRPO
│   ├── memory/         # SQLite + FTS5
│   ├── planning/       # objectifs et missions
│   ├── initiative/     # curiosité, idées, auto-évaluation, scheduler
│   ├── conversation/   # compagnon conversationnel
│   ├── agents/         # sandbox, workspace, orchestrateur
│   ├── coding/         # agent de codage + outils
│   ├── tools/          # registre d'outils typés
│   ├── execution/      # file de tâches durable + workers
│   └── api/            # FastAPI
├── configs/
├── scripts/
├── tests/
├── docs/
└── pyproject.toml
```

## Modèle

- Decoder-only Transformer, pré-norm RMSNorm.
- RoPE, attention causale, GQA (`num_key_value_heads`).
- SwiGLU, embeddings liés (`tie_word_embeddings`).
- KV-cache, génération autoregressive, température, top-k, top-p,
  répétition penalty, streaming token par token.
- Noms de tenseurs identiques à `transformers.LlamaForCausalLM`.

Le preset `tiny-cpu` (2,56 M, contexte 256) est le modèle de référence. Le
preset `cortex-15m` (14,56 M, contexte 512) est entraînable sur CPU ou GPU
gratuit ; `cortex-50m` et `cortex-150m` sont configurés mais non entraînés ici.

## Entraînement historique

- Corpus : Les Misérables, Vingt Mille Lieues sous les mers, Le Comte de
  Monte-Cristo, Frankenstein, TinyShakespeare, Wikipédia FR.
- 5,92 M tokens train, 133 K validation, 12,29 M tokens vus en 3 000 steps.
- Tokenizer BPE 4096, batch 16, contexte 256, AdamW, cosine, warmup 150.
- Meilleure validation : loss 3,634 ; perplexité 37,44.

Détails : [`docs/TRAINING.md`](docs/TRAINING.md).

## Audit SFT et dataset v3

Le premier SFT (1 080 exemples, 184 questions, 896 doublons, 27 % d'identité)
répétait une réponse canned et sa validation fuyait. Le dataset v3 contient
789 questions uniques, zéro doublon, 0,63 % d'identité et un split groupé sans
fuite. Le modèle de 2,56 M reste limité : 1/17 sur le benchmark demandé. Les
ablations v4 et v5 ne dépassent pas 1/17. Résultat négatif documenté dans
[`docs/SFT_AUDIT_AND_REBUILD.md`](docs/SFT_AUDIT_AND_REBUILD.md).

## Raisonnement

- `ArithmeticVerifier` : parse et vérifie l'arithmétique via un évaluateur AST
  restreint (pas un second LLM).
- `ReasoningEngine` : best-of-N, choix par vérification puis log-probabilité,
  révision optionnelle.
- `grpo.py` : avantages de groupe, objectif clippé, KL optionnel.
- `scripts/train_grpo.py` : boucle expérimentale avec récompense
  programmatique, journalisée step par step.

## Mémoire

SQLite + FTS5, sept types : `working`, `short_term`, `long_term`, `episodic`,
`semantic`, `project`, `preference`. Recherche hybride FTS5 + recouvrement
lexical + embeddings hashés. Consolidation extractive. La mémoire externe est
distincte des connaissances intégrées dans les poids ; aucune mise à jour de
poids n'est automatique.

## Autonomie et initiative

Le moteur propose au plus un message à la fois, et seulement si la policy
l'autorise : heures calmes, intervalle minimum, plafond quotidien, inactivité
utilisateur, déduplication, score minimum. Sources : objectifs bloqués, mémoire
projet, questions sans réponse, recombinaison d'idées, auto-évaluation.

```bash
cortex serve --model-dir checkpoints/tiny-cpu --scheduler
curl -X POST localhost:8000/v1/initiatives/tick -H 'content-type: application/json' -d '{}'
```

## Agent de codage

Lecture de dépôt, fichiers, écriture, commandes autorisées, pytest, diff et
préparation de commit. Garde-fous : jail de chemins, liste blanche
d'exécutables, `git push`/`git remote`/`git config` interdits, variables
d'environnement suspectes refusées, limites CPU/mémoire, approbation explicite
pour les écritures, commandes et commits, missions persistantes avec reprise.

Limite : c'est un sandbox **processus**, pas un conteneur. Voir
[`docs/SECURITY.md`](docs/SECURITY.md).

## API

`uvicorn cortex.api.app:app` ou `cortex serve`. Documentation sur `/docs`.
Authentification optionnelle par `x-api-key` (`CORTEX_API_KEY`).

| Méthode | Route | Rôle |
| --- | --- | --- |
| GET | `/health` | état public |
| GET | `/v1/model/info` | paramètres, contexte, device |
| POST | `/v1/generate` | génération |
| POST | `/v1/generate/stream` | génération SSE |
| POST | `/v1/chat` | conversation |
| POST | `/v1/chat/stream` | conversation SSE |
| POST | `/v1/reason` | raisonnement vérifié |
| GET/POST | `/v1/memory*` | mémoire |
| GET/POST/PATCH | `/v1/goals*` | objectifs |
| GET/POST | `/v1/missions*` | missions et approbations |
| GET/POST | `/v1/tasks*` | file de tâches |
| GET/POST | `/v1/initiatives*` | initiatives |
| GET | `/v1/events` | SSE des initiatives en attente |

## Hugging Face

`export/Cortex-0` est généré par :

```bash
python scripts/export_hf.py --model-dir checkpoints/tiny-cpu \
  --out export/Cortex-0 --verify
```

La vérification charge le modèle avec `transformers.LlamaForCausalLM` et
compare les logits : écart maximal 0.0. Les poids de l'ancien modèle sont dans
le dépôt privé `Frankenstein-Labs/Cortex-0-tiny`. Le nouveau corpus v2 et ses
checkpoints ne sont pas publiés sur Hugging Face.
