# CORTEX-0

**Small Autonomous Language Model — Frankenstein-Labs**

CORTEX-0 est une petite LLM construite **de zéro** : architecture Transformer
décodeur, tokenizer BPE, pipeline de préentraînement, évaluation, raisonnement
vérifié, mémoire persistante, moteur d'initiative et agent de codage. Le code
est réel, testé et exécutable sur CPU. Les poids sont exportables au format
Hugging Face `LlamaForCausalLM`.

> CORTEX-0 est un projet de recherche précoce, pas un assistant généraliste.
> Le modèle entraîné dans ce dépôt fait ~2,56 M de paramètres, sur un corpus de
> 5,9 M de tokens (12,3 M de tokens vus, ~2 époques). Il produit du texte court,
> souvent imparfait. Toutes les métriques publiées ici sont mesurées, jamais simulées.

---

## État vérifié

| Élément | État |
| --- | --- |
| Architecture Llama-compatible | ✅ testée (causalité, KV-cache, GQA, RoPE) |
| Tokenizer BPE 4096 | ✅ entraîné, aller-retour octet-exact |
| Préentraînement CPU | ✅ 3 000 steps, ~5 700 tokens/s |
| Export Hugging Face | ✅ logits identiques (`max_logit_diff = 0.0`) |
| Tests automatisés | ✅ 89 tests (`pytest -q`) |
| Perplexité validation | ✅ 37,44 (loss 3,623) |
| Mémoire SQLite + FTS5 | ✅ stockage, recherche, consolidation |
| Raisonnement + vérificateur | ✅ best-of-N, révision, GRPO |
| Initiative proactive | ✅ policy, scheduler, déduplication |
| Agent de codage | ✅ sandbox, approbations, missions persistantes |
| API FastAPI | ✅ chat, streaming SSE, mémoire, missions |
| Publication Hugging Face | ⛔ bloquée : aucun token `HF_TOKEN` disponible |

---

## Démarrage rapide

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"

# 1. Construire le tokenizer et les fichiers de tokens
python scripts/prepare_data.py --config configs/tiny-cpu.yaml

# 2. Préentraîner (CPU, ~36 min pour 3 000 steps)
python scripts/train.py --config configs/tiny-cpu.yaml

# 3. Évaluer
python scripts/evaluate.py --model-dir checkpoints/tiny-cpu \
  --val-data datasets/processed/val.bin --n 20

# 4. Générer
python scripts/generate.py --model-dir checkpoints/tiny-cpu \
  --prompt "CORTEX est" --max-new-tokens 60

# 5. Discuter
python scripts/serve.py --model-dir checkpoints/tiny-cpu --port 8000
# puis http://127.0.0.1:8000/docs
```

CLI unifiée :

```bash
cortex train --config configs/tiny-cpu.yaml
cortex chat --model-dir checkpoints/tiny-cpu
cortex serve --model-dir checkpoints/tiny-cpu --scheduler
cortex evaluate --model-dir checkpoints/tiny-cpu
cortex export-hf --model-dir checkpoints/tiny-cpu --out export/Cortex-0 --verify
```

---

## Architecture

```text
cortex-0/
├── cortex/
│   ├── config.py            # CortexConfig (miroir de Llama)
│   ├── model/               # RMSNorm, RoPE, GQA, SwiGLU, Transformer, sampling
│   ├── tokenizer/           # BPE byte-level + template de chat CORTEX
│   ├── training/            # données, trainer, SFT, corpus synthétique
│   ├── evaluation/          # perplexité, suite arithmétique/logique
│   ├── reasoning/           # vérificateur AST, best-of-N, révision, GRPO
│   ├── memory/              # SQLite + FTS5, embeddings hashés, consolidation
│   ├── planning/            # objectifs et missions persistantes
│   ├── initiative/          # curiosité, idées, auto-évaluation, scheduler
│   ├── conversation/        # compagnon conversationnel proactif
│   ├── agents/              # sandbox, workspace, orchestrateur
│   ├── coding/              # agent de codage + outils
│   ├── tools/               # registre d'outils typés
│   ├── execution/           # file de tâches durable + workers
│   └── api/                 # FastAPI
├── configs/                 # tiny-cpu (2,6 M), cortex-50m, cortex-150m
├── scripts/                 # prepare_data, train, train_sft, train_grpo, ...
├── tests/                   # 89 tests
├── docs/                    # architecture, training, sécurité, roadmap
└── pyproject.toml
```

Détails : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Modèle

- Decoder-only Transformer, pré-norm RMSNorm.
- RoPE, attention causale, GQA (`num_key_value_heads`).
- SwiGLU, embeddings liés (`tie_word_embeddings`).
- KV-cache, génération autoregressive, température, top-k, top-p,
  répétition penalty, streaming token par token.
- Noms de tenseurs identiques à `transformers.LlamaForCausalLM`.

Le preset `tiny-cpu` (2,56 M de paramètres, contexte 256) est celui qui est
entraîné ici. Les presets `cortex-50m` et `cortex-150m` sont fournis et
configurés, mais **ne sont pas entraînés** dans ce dépôt (CPU uniquement).

---

## Entraînement

- Corpus : Les Misérables, Vingt Mille Lieues sous les mers, Le Comte de
  Monte-Cristo, Frankenstein, TinyShakespeare, Wikipédia FR.
- 5,92 M tokens d'entraînement, 133 K tokens de validation.
- 12,29 M tokens vus en 3 000 steps (~2 époques), loss finale 3,683.
- Tokenizer BPE byte-level, 4096 tokens, 10 tokens spéciaux.
- 3 000 steps, batch 16, contexte 256, AdamW, cosine schedule, warmup 150.
- Débit mesuré : ~5 700 tokens/s sur CPU.
- Meilleure validation : loss 3,634 ; perplexité 37,44 sur 102 400 tokens.
- Checkpoints : `checkpoints/tiny-cpu/checkpoint-*`, meilleur modèle dans
  `checkpoints/tiny-cpu/best`.

Détails et journal : [`docs/TRAINING.md`](docs/TRAINING.md).

---

## Évaluation

Deux niveaux, volontairement séparés :

1. **Perplexité** sur la validation (loss, perplexité, bits/token).
2. **Suite interne** `cortex-internal-v1` : arithmétique et logique, en
   français et en anglais, avec vérité terrain calculée.

Cette suite **n'est ni MMLU, ni GSM8K, ni un benchmark externe**. Les résultats
sont écrits dans `experiments/eval-tiny-cpu.json` avec le commit Git, la date,
le device et le nombre de paramètres. Voir [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Raisonnement

- `ArithmeticVerifier` : parse la réponse et vérifie l'arithmétique via un
  évaluateur AST restreint (pas un second LLM).
- `ReasoningEngine` : best-of-N, choix par vérification puis log-probabilité,
  révision optionnelle, enregistrement des exécutions.
- `grpo.py` : avantages de groupe, objectif clippé, KL optionnel.
- `scripts/train_grpo.py` : boucle expérimentale réelle avec récompense
  programmatique. Ce n'est pas une preuve de raisonnement fort : les résultats
  sont journalisés step par step.

```bash
python scripts/train_grpo.py --model-dir checkpoints/tiny-cpu \
  --steps 50 --group-size 4 --out checkpoints/tiny-cpu-grpo
```

---

## Mémoire

SQLite + FTS5, sept types : `working`, `short_term`, `long_term`, `episodic`,
`semantic`, `project`, `preference`.

- Recherche hybride : FTS5 + recouvrement lexical + embeddings hashés.
- Consolidation extractive des conversations en mémoire sémantique.
- Mémoire projet, préférences utilisateur, journal d'événements.
- La mémoire externe est distincte des connaissances intégrées dans les poids.
  Aucune mise à jour de poids n'est faite automatiquement.

---

## Autonomie et initiative

Le moteur d'initiative propose **au plus un message à la fois**, et seulement si
la policy l'autorise :

- heures calmes (22h–7h par défaut) ;
- intervalle minimum (15 min) et plafond quotidien (6) ;
- l'utilisateur doit être inactif depuis 5 min ;
- nombre d'initiatives en attente limité ;
- déduplication lexicale et sémantique ;
- score minimum.

Sources : objectifs bloqués, mémoire projet, questions sans réponse,
recombinaison d'idées, auto-évaluation des raisonnements. Chaque proposition
est persistée avec son score et sa source ; elle peut être ignorée.

```bash
cortex serve --model-dir checkpoints/tiny-cpu --scheduler
curl -X POST localhost:8000/v1/initiatives/tick -H 'content-type: application/json' -d '{}'
```

---

## Agent de codage

Capacités : lire un dépôt, lister les fichiers, lire/écrire, exécuter des
commandes autorisées, lancer pytest, voir le diff, préparer un commit.

Garde-fous :

- jail de chemins dans le workspace ;
- liste blanche d'exécutables (`python3`, `pytest`, `git`, `ls`, ...) ;
- `git push`, `git remote`, `git config` interdits ;
- variables d'environnement suspectes refusées ;
- limites CPU/mémoire/taille de fichier ;
- écritures, commandes et commits soumis à **approbation explicite** ;
- missions persistantes avec plan, checkpoints et reprise ;
- une nouvelle instruction met à jour le plan sans rejouer le travail terminé.

Limite importante : c'est un sandbox **processus**, pas un conteneur. Il ne
protège pas contre du code Python malveillant qui lirait `/etc` ou ouvrirait
une socket. Voir [`docs/SECURITY.md`](docs/SECURITY.md).

---

## API

`uvicorn cortex.api.app:app` ou `cortex serve`. Documentation interactive sur
`/docs`. Authentification optionnelle par `x-api-key` (`CORTEX_API_KEY`).

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

---

## Hugging Face

Le dossier `export/Cortex-0` est généré par :

```bash
python scripts/export_hf.py --model-dir checkpoints/tiny-cpu \
  --out export/Cortex-0 --verify
```

La vérification charge le modèle avec `transformers.LlamaForCausalLM` et
compare les logits : écart maximal mesuré **0.0**, tokenizer identique.

La publication est prête mais **bloquée** : aucun `HF_TOKEN` n'est disponible
dans cet environnement. Voir [`docs/HF_PUBLISHING.md`](docs/HF_PUBLISHING.md).

---

## Sécurité

- Aucun secret dans le code, les logs ou les commits.
- Aucun benchmark inventé, aucun poids non entraîné présenté comme final.
- Licences et sources des datasets vérifiées.
- Opérations sensibles soumises à autorisation.
- Journaux d'exécution et checkpoints conservés.

Voir [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Feuille de route

Phases 1–11 largement couvertes ; phase 12 (publication) partielle à cause du
token HF manquant. Prochaines étapes : entraîner 50M/150M sur GPU, ajouter des
données africaines (soussou, malinké, bambara), renforcer le raisonnement,
brancher un vrai runtime isolé (conteneur/K8s) pour l'agent.

Voir [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Licence

Apache-2.0. Voir [`LICENSE`](LICENSE).

Frankenstein-Labs — **CORTEX-0, building autonomous intelligence.**
