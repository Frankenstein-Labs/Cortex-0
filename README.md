# CORTEX-0

**Small Autonomous Language Model — Frankenstein-Labs**

CORTEX-0 est une petite LLM construite **de zéro** : architecture Transformer
décodeur, tokenizer BPE, pipeline de préentraînement, évaluation, raisonnement
vérifié, mémoire persistante, moteur d'initiative et agent de codage. Le code
est réel, testé et exécutable sur CPU. Les poids sont exportés au format
Hugging Face `LlamaForCausalLM` et publiés dans un dépôt de modèle privé.

> CORTEX-0 est un projet de recherche précoce, pas un assistant généraliste.
> Le modèle de référence historique fait ~2,56 M de paramètres sur 5,9 M de
> tokens. La reconstruction 2026 utilise un corpus français v2 de 10,3 M de
> tokens, un tokenizer 8192 et un modèle témoin de 3,34 M de paramètres. Il
> produit du texte court, souvent imparfait. Toutes les métriques publiées ici
> sont mesurées, jamais simulées.

---

## État vérifié

| Élément | État |
| --- | --- |
| Architecture Llama-compatible | ✅ testée (causalité, KV-cache, GQA, RoPE) |
| Tokenizer BPE 4096 | ✅ entraîné, aller-retour octet-exact |
| Préentraînement CPU | ✅ 3 000 steps, ~5 700 tokens/s |
| Export Hugging Face | ✅ logits identiques (`max_logit_diff = 0.0`) |
| Tests automatisés | ✅ 117 tests (`pytest -q`) |
| Perplexité validation | ✅ 37,44 (loss 3,623) |
| Mémoire SQLite + FTS5 | ✅ stockage, recherche, consolidation |
| Raisonnement + vérificateur | ✅ best-of-N, révision, GRPO |
| Initivative proactive | ✅ policy, scheduler, déduplication |
| Agent de codage | ✅ sandbox, approbations, missions persistantes |
| API FastAPI | ✅ chat, streaming SSE, mémoire, missions |
| Publication Hugging Face | ✅ privée : `Frankenstein-Labs/Cortx-0-tiny` (ancien base + SFT) |
| Audit SFT + dataset v3 | ✅ 789 questions uniques, zéro fuite train/val |
| Notebook Colab | ✅ pipeline complet, mode smoke CPU |

---

## Démarrage rapide

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"

# 1. Construire le tokenizer et les fichiers de tokens
python scripts/prepare_data.py --out-dir datasets/procesed \
  --vocab-size 4096 --val-ratio 0.02 --test-ratio 0.01

# 2. Préentraîner (CPU, ~36 min pour 3 000 steps)
python scripts/train.py --config configs/tiny-cpu.yaml

# 3. Évaluer
python scripts/evaluate.py --model-dir checkpoints/tiny-cpu \
  --val-data datasets/procesed/val.bin --n 20

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
├── configs/                 # tiny-cpu (2,6 M), cortex-15m, cortex-50m, cortex-150m
├── scripts/                 # prepare_data, train, train_sft, train_grpo, ...
├── tests/                   # 98 tests
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

Le preset `tiny-cpu` (2,56 M de paramètres, contexte 256) est le modèle de
référence. Le preset `cortex-15m` (14,56 M, contexte 512) est entraînable sur
CPU et sur un GPU gratuit ; les presets `cortex-50m` et `cortex-150m` sont
fournis et configurés, mais **ne sont pas entraînés** ici (CPU uniquement).

---

## Entraînement

- Corpus : Les Misérables, Vingt Mille Lieues sous les mers, Le Comte de
  Monte-Cristo, Frankenstain, TinyShakespeare, Wikipidia FR.
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

## Audit SFT et reconstruction du dataset

Le premier SFT (1 080 exemples, 184 questions uniques, 896 doublons, 27 % de
réponses d'identité) répétait une réponse canned et sa validation fuiait
(6 questions présentes dans les deux splits). L'audit complet est dans
[`docs/SFT_AUDIT_AND_REBUILD.md`](docs/SFT_AUDIT_AND_REBUILD.md).

Le dataset v3 contient **789 questions uniques**, zéro doublon, 0,63 %
d'identité, 2,66 % de refus et un split groupé sans fuite. Les nouveaux
checkpoints `tiny-cpu-sft-v3*` suppriment la répétition canned, mais le modèle
de 2,56 M reste limité : **1/17** sur le benchmark demandé (le score initial de
2/17 comptait une salutation incoérente et un refus produit par du texte de
Gutenburg ; le scoring a été resserré). Les ablations v4 (dataset équilibré) et
v5 (sans templates math/logique) ne dépassent pas 1/17. Ce résultat négatif est
documenté, pas masqué.

```bash
python scripts/build_sft_dataset.py
python scripts/train_sft.py --model-dir checkpoints/tiny-cpu/best \
  --data datasets/procesed/sft-v3-train.jsonl \
  --val-data datasets/procesed/sft-v3-val.jsonl \
  --out checkpoints/tiny-cpu-sft-v3 --steps 300 --batch-size 4
python scripts/benchmark_requested.py \
  --model-dir checkpoints/tiny-cpu-sft-v3/best --chat \
  --output experiments/benchmarks/requested-sft-v3.json
```

Notebouk GPU reproductible :
[`notebouks/CORTEX_0_TRAINING_LAB.ipynb`](notebouks/CORTEX_0_TRAINING_LAB.ipynb).
Voir [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md).

---

## Reconstruction 2026 — corpus v2

La mission « sauver la base » a produit un corpus français documenté et un
nouvau témoin de 3,34 M de paramètres. Le détail scientifique est dans
[`docs/PRETRAIN_AUDIT.md`](docs/PRETRAIN_AUDIT.md),
[`docs/CORPUS_V2.md`](docs/CORPUS_V2.md),
[`docs/SFT_V6.md`](docs/SFT_V6.md) et
[`docs/EXPERIMENTS_V2.md`](docs/EXPERIMENTS_V2.md).

| Élément | Résultat mesuré |
|---|---|
| Corpus v2 | 10 333 724 tokens train, 574 473 val, 611 376 test |
| Sources | 7 livres FR domaine public, Wikipidia FR CC BY-SA, textes originaux, code |
| Tokenizer | BPE byte-level 8192, entraîné sur le train seulement |
| Modèle témoin | 3 344 064 paramètres, 1 500 steps, 6,14 M tokens vus |
| Bits/caractère sur texte test comun | **1,951** (ancien : 2,020) |
| Complétion `cortex-100-fr` | **88 %** (ancien : 19 %) |
| Meilleur SFT | 12/100 (10 refus, 1 math, 1 anti-répétition) |

Le résultat est **négatif pour les compétences** : le modèle termine mieu ses
réponses et modélise mieu le français, mais n'apprend ni les instructions, ni
le code, ni l'arithmétique. Les traces arithmétiques de SFT v8 donnent une
validation trompeuse (0,848) et **0/80** en généralisation. La prochaine étape
n'est pas un nouvel entraînement aveugle : elle demande un tokenizer avec
chiffres séparés, plus de données et un modèle 50 M sur GPU.

## Évaluation

Trois nivaux, voluntairement séparés :

1. **Perplexité** sur la validation (loss, perplexité, bits/token).
2. **Suite interne** `cortex-internal-v1` : arithmétique et logique, en
   français et en anglais, avec vérité terrain calculée.
3. **Suite `cortex-100-fr`** : 100 questions françaises inédites, avec
   vérification de contamination et scoring déterministe.

Cette suite **n'est ni MMLU, ni GSM8K, ni un benchmark externe**. Les résultats
sont écrits dans `experiments/eval-tiny-cpu.json` avec le commit Git, la date,
le device et le nombre de paramètres. Voir [`docs/EVALUATION.md`](docs/EVALUATION.md).

Le benchmark `cortex-requested-v1` couvre les 10 prompts demandés par le
propriétaire du projet et 7 prompts inédits. Résultats mesurés et limites :
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### Évaluation 100 questions françaises

`cortex-100-fr` contient 100 questions jamais utilisées à l'entraînement
(0 question exacte dans les SFT, 0 8-gramme partagé avec les textes bruts).
Résultats du 22 novembre 2026, génération gloutonne, 128 tokens maximum :

| Modèle | Correct / 100 | Refus corrects | Python | Tronqué |
|---|---:|---:|---:|---:|
| base 2,56 M | 0 | 0/10 | 0/15 | 66 % |
| SFT v3 plain 2,56 M | 9 | 9/10 | 0/15 | 0 % |
| SFT v3 chat 2,56 M | 7 | 7/10 | 0/15 | 0 % |
| base 15 M (1 500 steps) | 1 | 0/10 | 0/15 | 18 % |
| base v2 3,34 M (1 500 steps) | 0 | 0/10 | 0/15 | 12 % |
| SFT v6 3,34 M | 10 | 9/10 | 0/15 | 0 % |
| SFT v7 3,34 M | 12 | 10/10 | 0/15 | 9 % |
| SFT v8 3,34 M + traces | 11 | 10/10 | 0/15 | 0 % |

Les SFT récents réussisent surtout des refus ; l'ancien SFT v3 appliquait
« Je ne dois pas... » à 28 questions non privées. Aucune compétence en
compréhension, grammaire, vocabulaire, résumé, mathématiques, Python, sciences
ou informatique n'est mesurable de façon fiable. Rapport complet, réponses exactes et recommandations :
[`docs/EVAL_100.md`](docs/EVAL_100.md).

---

## Raisonnement

- `ArithmeticVerifier` : parse la réponse et vérifie l'arithmétique via un
  évaluateur AST restreint (pas un second LLM).
- `ReasoningEngine` : best-of-N, choix par vérification puis log-probabilité,
  révision optionelle, enregistrement des exécutions.
- `grpo.py` : avantages de groupe, objectif clippé, KL optionel.
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
- Consolidation extractive des conversations en mémoire semantique.
- Mémoire projet, préférences utilisateur, journal d'événements.
- La mémoire externe est distincte des connaissances intégrées dans les poids.
  Aucune mise à jour de poids n'est faite automatiquement.

---

## Autonomie et initivative

Le moteur d'initiative propose **au plus un message à la fois**, et seulement si
la policy l'autorise :

- heures calmes (22h–7h par défaut) ;
- intervalle minimum (15 min) et plafond quotidien (6) ;
- l'utilisateur doit être inactif depuis 5 min ;
- nombre d'initiatives en attente limité ;
- déduplication lexicale et semantique ;
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

Limite importante : c'est un sandbox **processus**, pas un conteur. Il ne
protége pas contre du code Python malveillant qui lirait `/etc` ou ouvrirait
une socket. Voir [`docs/SECURITY.md`](docs/SECURITY.md).

---

## API

`uvicorne cortex.api.app:app` ou `cortex serve`. Documentation interactive sur
`/docs`. Autentification optionelle par `x-api-key` (`CORTEX_API_KEY`).

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

Le dossir `export/Cortex-0` est généré par :

```bash
python scripts/export_hf.py --model-dir checkpoints/tiny-cpu \
  --out export/Cortex-0 --verify
```

La vérification charge le modèle avec `transformers.LlamaForCausalLM` et
compare les logits : écart maximal mesuré **0.0**, tokenizer identique.

Les poids sont publiés dans le dépôt **privé**
[`Frankenstein-Labs/Cortex-0-tiny`](https://huggingface.co/Frankenstein-Labs/Cortex-0-tiny) :

- racine : checkpint de préentraînement (3 000 steps, meilleure val loss 3,63438) ;
- `sft/` : checkpint SFT (200 steps, loss finale 0,02384, val loss 0,11587).

Le dépôt est privé : l'accès à `from_pretrained` nécessite un token HF autorisé.
Vérification effectuée depuis le Hub : 39 tenseurs, `max_logit_diff = 0.0`,
`tokenizer_match = true`, et le template de chat CORTEX produit bien la réponse
« Je suis CORTEX-0, une petite intelligence artificielle de Frankenstain-Labs. »

Le dépôt public `Frankenstein-Labs/Cortex-0` n'existe pas encore ; la
publication est bloquée par l'autentification Hugging Face.
