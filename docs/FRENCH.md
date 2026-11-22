# CORTEX-0 français — rapport technique

Ce rapport ne contient que des valeurs **mesurées**. Chaque chiffre provient
d'une commande réellement exécutée sur un runtime Google Colab et d'un artefact
sous `experiments/`. Ce qui n'a pas été fait est listé explicitement dans la
section « Ce qui n'est pas fait ».

## 1. Environnement de calcul

| Élément | Valeur mesurée |
| --- | --- |
| Outil | `google-colab-cli` 0.8.0 (CLI officiel) |
| Authentification | OAuth2 « remote copy-paste » |
| GPU | Tesla T4, 15 360 MiB |
| PyTorch / CUDA | 2.14.1+cu130 / CUDA 13.0 |
| dtype retenu | `bfloat16` (`bf16_supported = true`) |
| Débit mesuré | 66 000–72 000 tokens/s |

Détail de la configuration et des limitations : `docs/colab.md`.

## 2. Corpus

Construit par `scripts/build_french_corpus.py`, décrit dans `docs/CORPUS.md`.

- 2 539 documents, 35 905 076 caractères.
- 2 537 articles de Wikipédia français (CC BY-SA 4.0), 2 fichiers de code du
dépôt (Apache-2.0).
- `gutenberg_fr` a échoué (réseau) et l'échec est enregistré, pas masqué.
- Après tokenisation : 11 324 207 tokens d'entraînement, 95 983 de validation,
  123 676 de test.

## 3. Audit du tokenizer 4096

`scripts/tokenizer_audit.py`, artefact
`experiments/tokenizer-audit-french-4096.json`.

| Mesure | Valeur |
| --- | --- |
| Tokens par mot (français) | **2,121** |
| Mots découpés en plusieurs tokens | **61,24 %** |
| Mots accentués découpés | **83,58 %** |
| Mots rares (fréquence 1) découpés | 99,15 % |
| Caractères par token | 3,117 |
| Caractères par token (code Python) | 1,583 |
| Caractères français couverts par un seul token | 44 / 46 (`ë`, `æ` absents) |

**Verdict.** Un vocabulaire de 4 096 tokens est trop petit pour le français :
deux tiers des mots exigent plusieurs tokens, et plus de quatre mots accentués
sur cinq sont découpés. Conserver ce vocabulaire coûte de la longueur de
contexte et de la qualité. **Aucune expérience comparative avec un vocabulaire
plus grand n'a été menée** : le remplacer maintenant serait une décision non
mesurée, donc elle n'a pas été prise.

## 4. Préentraînement

Configuration `configs/french-tiny.yaml` — architecture **identique** à
`tiny-cpu` (2 557 632 paramètres, 4 couches, 192 de dimension, contexte 256),
seules les données changent.

| Métrique | Valeur mesurée |
| --- | --- |
| Steps | 12 000 |
| Tokens vus | 49 152 000 (~4,3 époques) |
| `best_val_loss` | **3,17706** |
| Perplexité de validation | **23,976** |
| Perplexité de validation initiale (step 500) | 131,095 |

Trajectoire de la perplexité de validation : 131,10 (500) → 64,26 (1 000) →
48,95 (1 500) → 29,23 (5 500) → 24,07 (11 000) → **23,98 (12 000)**.

Le run a été interrompu une première fois par l'expiration de la session Colab
et **repris** depuis `checkpoint-*` sans perte, ce qui valide le mécanisme de
reprise.

## 5. Fine-tuning supervisé (identité et conversation)

361 exemples de chat (`datasets/sft/train.jsonl`), issus du jeu d'identité
contrôlé et des exemples synthétiques français.

Un run naïf (400 steps, lr 2e-4) atteint `val_perplexity` **1,43** — mais c'est
de la mémorisation : la perplexité sur le français général s'effondre de 23,97
à **79,43**. Sélectionner ce checkpoint sur la seule perte d'entraînement aurait
été une erreur. Quatre variantes ont donc été comparées
(`experiments/sft-sweep.json`) :

| Variante | Perplexité française | Identité | Complétion |
| --- | --- | --- | --- |
| préentraînement seul | **23,97** | 0,0 % | 0,0 % |
| 400 steps, lr 2e-4 | 79,43 | 12,5 % | 0,0 % |
| **150 steps, lr 1e-4** | **40,66** | **12,5 %** | 0,0 % |
| 300 steps, lr 5e-5 | 38,64 | 0,0 % | 0,0 % |
| 100 steps, lr 2e-4 | 51,27 | 0,0 % | 0,0 % |

Retenu : **150 steps, lr 1e-4**, seul compromis qui améliore l'identité sans
doubler la perplexité.

## 6. Exemples réels, avant et après SFT

`experiments/samples-before-after.json`. Génération gloutonne avec la même
graine, 30 tokens.

| Invite | Préentraînement seul | Après SFT |
| --- | --- | --- |
| Quel est ton nom ? | « Le nom de nombres premiers degré d'accès (souvent : c'est le nom d'un objet « arbre »), » | « u. Je n'ai pas de conscience et je ne prends pas d'initiative sans autorisation. » |
| Qui t'a créé ? | « , est un langage de programmation. En français, les langages de programmations d'IET étaient d'abord » | « , une petite IA autonome de Frankenstein-Labs. Je suis un modèle de langage entraîné localement, » |
| Le chat mange une | « sorte de voie (dans la cuisine française, mais aussi les tours de mai à octobre) qui se déroulent. » | « sorte de , est limitée à mes données d'entraînement et je peux me tromper. » |
| Bonjour, je m'appelle | « le cessez-le-feu est l'un des premiers auteurs de la famille des hautes bourgeoises. » | « CORTEX-0. Je n'ai pas de conscience et je ne prends pas d'initiative sans » |

**Lecture honnête.** Le préentraînement produit un français localement
plausible mais hors sujet : le modèle ne répond pas, il continue du texte
encyclopédique. Après SFT, les faits d'identité apparaissent (« CORTEX-0 »,
« Frankenstein-Labs », « petite IA autonome »), mais ils **débordent sur des
invites sans rapport** : « Le chat mange une » reçoit une phrase de limites. Le
modèle a mémorisé des formules, il ne les a pas comprises. À 2,5 M de
paramètres et avec 361 exemples, c'est le comportement attendu, et c'est
pourquoi la précision d'identité reste à 12,5 % (1 question sur 8).

## 7. Ce qui n'est pas fait

- **Aucun entraînement GRPO** n'a été lancé.
- **Aucune publication Hugging Face** : `HF_TOKEN` n'est pas présent dans
  l'environnement. `scripts/publish_hf.py --private` est prêt et refuse de
  continuer sans jeton, par conception.
- **Aucun benchmark externe** n'est revendiqué. Les scores internes
  (identité, complétion) mesurent un petit modèle, rien de plus.
- **Aucune comparaison de vocabulaires** (4k contre 8k/16k) n'a été mesurée.
- **`gutenberg_fr` n'a pas produit de données** sur ce run.
- Les configurations `cortex-50m` et `cortex-150m` n'ont pas été entraînées.

## 8. Prochaines étapes, par ordre de valeur

1. **Tokenizer 8 192 ou 16 384** entraîné sur le corpus français, puis audit
   comparatif avec le même script. C'est le levier le plus rentable : 61 % des
   mots sont actuellement découpés.
2. **Corpus plus large et plus varié** : viser 100 M+ tokens, ajouter du
   dialogue réel, du code et des textes pédagogiques, pas seulement Wikipédia.
3. **SFT plus riche** : plusieurs milliers d'exemples, avec des réponses
   variées par question, pour éviter la mémorisation d'une formule unique.
4. **Modèle plus grand** : `cortex-50m` sur T4 avec accumulation de gradient,
   en conservant le contexte de 256 tokens ou en le portant à 512.
5. **GRPO** sur l'arithmétique et le raisonnement, une fois le SFT stable.

## 9. Reproduction

```bash
# corpus + tokenizer + shards
python scripts/build_french_corpus.py --out-dir datasets/french \
    --wikipedia-docs 3000 --tokenize --sft-dir datasets/sft

# audit du tokenizer
python scripts/tokenizer_audit.py --tokenizer-dir datasets/french/tokenizer \
    --corpus datasets/french/corpus.jsonl \
    --out experiments/tokenizer-audit-french-4096.json

# préentraînement (Colab T4)
python scripts/train.py --config configs/french-tiny.yaml

# SFT
python scripts/train_sft.py --model-dir checkpoints/french-tiny/best \
    --data datasets/sft/train.jsonl --out checkpoints/sft-150-lr1e-4 \
    --steps 150 --batch-size 8 --lr 0.0001 --max-length 256

# évaluation
python scripts/evaluate_french.py --model-dir checkpoints/sft-150-lr1e-4 \
    --val-data datasets/french/val.bin --output experiments/french-eval-sft.json
```
