# Entraînement de CORTEX-0

## Environnement réel

| Élément | Valeur |
| --- | --- |
| Machine | conteneur Linux, CPU uniquement |
| Python | 3.11 |
| PyTorch | 2.x, CPU |
| Device | `cpu` |
| Durée du run | ~36 min pour 3 000 steps |
| Débit | ~5 700 tokens/s |

Aucun GPU n'était disponible. Les presets 50M et 150M sont fournis mais non
entraînés ici.

## Données

Le pipeline est défini dans `scripts/prepare_data.py`. Le manifeste complet
(`datasets/processed/manifest.json`) contient l'URL, la licence, le nombre de
caractères, le nombre de documents et le SHA-256 de chaque source.

| Source | Langue | Licence | Documents |
| --- | --- | --- | --- |
| Les Misérables (Gutenberg 17489) | fr | domaine public | 180 |
| Vingt Mille Lieues sous les mers (Gutenberg 5097) | fr | domaine public | 238 |
| Le Comte de Monte-Cristo (Gutenberg 17989) | fr | domaine public | 191 |
| Frankenstein (Gutenberg 84) | en | domaine public | 118 |
| TinyShakespeare | en | domaine public | 404 |
| Wikipédia FR (20231101.fr) | fr | CC BY-SA 4.0 | 2 147 |
| Code du projet | — | Apache-2.0 | 15 |

Résultat :

- 5 923 113 tokens d'entraînement ;
- 132 854 tokens de validation ;
- dtype `uint16` (vocabulaire 4096 < 65536).

Nettoyage : normalisation des espaces, suppression des en-têtes Gutenberg,
découpage en paragraphes, déduplication, filtrage des lignes trop courtes.
Les textes sont tokenisés par lots et écrits dans un fichier binaire pour
garder une empreinte mémoire bornée.

## Tokenizer

```bash
python scripts/prepare_data.py --config configs/tiny-cpu.yaml
```

- BPE byte-level, 4 096 tokens ;
- 10 tokens spéciaux réservés (IDs 0–9) ;
- alphabet initial = 256 octets, donc aucun UNK ;
- entraîné sur un échantillon du corpus (`_tokenizer_corpus.txt`).

Vérifications : aller-retour octet-exact sur français accentué, bambara,
malinké, soussou, emoji et code (`tests/test_tokenizer.py`).

## Préentraînement

Commande :

```bash
python scripts/train.py --config configs/tiny-cpu.yaml
```

Hyperparamètres (`configs/tiny-cpu.yaml`) :

| Paramètre | Valeur |
| --- | --- |
| Paramètres | 2 557 632 |
| Couches | 4 |
| Hidden | 192 |
| Têtes / têtes KV | 6 / 6 |
| Contexte | 256 |
| Vocabulaire | 4 096 |
| Batch | 16 |
| Steps | 3 000 |
| LR max | 6e-4 |
| Warmup | 150 |
| Schedule | cosine, min ratio 0.1 |
| Optimiseur | AdamW (0.9, 0.95), weight decay 0.1 |
| Gradient clipping | 1.0 |
| Seed | 1337 |

Le run couvre environ 2 époques. Les checkpoints sont écrits tous les 750
steps ; le meilleur modèle selon la validation est copié dans
`checkpoints/tiny-cpu/best`.

### Courbe mesurée

Extrait de `experiments/tiny-cpu-train.jsonl` :

| Step | Loss | Grad norm | Tokens/s |
| --- | --- | --- | --- |
| 50 | 7.543 | 0.910 | 6 020 |
| 500 | 4.841 | 0.898 | 5 840 |
| 1 000 | 4.278 | 1.047 | 5 789 |
| 1 500 | 3.851 | 1.190 | 5 795 |
| 2 000 | 3.770 | 1.281 | 5 747 |
| 2 250 | 3.801 | 1.292 | 5 711 |
| 2 400 | 3.420 | 1.243 | 5 693 |
| 2 750 | 3.605 | 1.291 | 5 684 |

Validation au step 2 400 : loss 3.690, perplexité 40.05. Validation finale au
step 3 000 : loss 3.63438, perplexité 37.878.

Les valeurs finales exactes sont dans `experiments/eval-tiny-cpu.json`, généré
par `scripts/evaluate.py` avec le commit Git et l'horodatage.

## Supervised fine-tuning

```bash
python scripts/train_sft.py \
  --model-dir checkpoints/tiny-cpu \
  --data datasets/processed/sft.jsonl \
  --out checkpoints/tiny-cpu-sft \
  --steps 300
```

Le dataset SFT est généré par `prepare_data.py` : conversations
`system`/`user`/`assistant` courtes en français (présentations, arithmétique
simple, questions sur le projet). `encode_supervised` masque tout sauf les
tours assistant.

Run mesuré sur le checkpoint de base :

| Paramètre | Valeur |
| --- | --- |
| Exemples | 1 080 |
| Steps | 200 |
| Batch | 8 |
| LR | 1e-4 |
| Durée | 41 s |
| Loss finale | 0,024 |
| Val loss | 0,116 |
| Val perplexité | 1,123 |

Le modèle reproduit alors exactement les réponses du jeu SFT (présentation,
capacités, limites). C'est de la **mémorisation** d'un petit jeu, pas une
généralisation. Le run valide la chaîne SFT et le template de chat ; il ne
prouve pas une amélioration conversationnelle générale.

Point d'implémentation important : la perte du modèle est **non décalée**
(`logits[t]` contre `labels[t]`). Le dataset de préentraînement fournit déjà
`y = x[t+1]`, tandis que `collate_sft` décale les labels d'une position vers la
gauche. Sans ce décalage, le SFT apprend à recopier l'entrée et produit
immédiatement `<|end|>` à l'inférence. Un test couvre ce comportement.

## GRPO expérimental

```bash
python scripts/train_grpo.py \
  --model-dir checkpoints/tiny-cpu \
  --steps 50 \
  --group-size 4 \
  --out checkpoints/tiny-cpu-grpo
```

- récompense binaire : réponse vérifiée par `ArithmeticVerifier` ;
- avantages normalisés dans chaque groupe ;
- objectif clippé (`clip_eps=0.2`), KL optionnel ;
- les groupes à variance nulle sont ignorés (aucun signal) ;
- chaque step est journalisé dans `grpo_log.jsonl`.

Ce run est une démonstration d'optimisation, pas une preuve de raisonnement.
Avec 2,6 M de paramètres et 50 steps, l'accuracy attendue reste faible.

## Reproductibilité

- Seeds : 1337 (données, modèle, trainer), 0 (évaluation).
- `manifest.json` contient les SHA-256 des sources.
- `config.json` est sauvegardé avec chaque checkpoint.
- Le tokenizer est sauvegardé avec chaque checkpoint.
- `evaluate.py` enregistre le commit Git, la date, le device et les paramètres.

Pour reproduire :

```bash
python scripts/prepare_data.py --config configs/tiny-cpu.yaml
python scripts/train.py --config configs/tiny-cpu.yaml
python scripts/evaluate.py --model-dir checkpoints/tiny-cpu
```

## Checkpoints

```text
checkpoints/tiny-cpu/
├── checkpoint-750/
├── checkpoint-1500/
├── checkpoint-2250/
├── best/
├── config.json
├── model.safetensors
├── tokenizer/
└── logs/
```

Les poids ne sont pas versionnés dans Git (`.gitignore`). Ils sont destinés à
Hugging Face ; voir `docs/HF_PUBLISHING.md`.
