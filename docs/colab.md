# Entraînement sur Google Colab

Ce document décrit la connexion vérifiée entre l'agent (OpenHands) et un runtime
Google Colab via le CLI officiel [`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli).

## Environnement réellement mesuré

Toutes les valeurs ci-dessous proviennent d'exécutions réelles sur un runtime
Colab, pas d'estimations.

| Élément | Valeur mesurée |
| --- | --- |
| CLI | `google-colab-cli` 0.8.0 |
| Authentification | OAuth2 « remote copy-paste » (identique à `gcloud auth application-default login`) |
| Runtime GPU | Tesla T4, 15 360 MiB (14,56 GiB utilisables), compute capability 7.5 |
| Runtime CPU | 2 vCPU, 112,6 Gio de disque |
| Python distant | 3.13.16 (GPU), 3.13.18 (CPU) |
| PyTorch distant | 2.14.1+cu130, CUDA 13.0 |
| `bf16_supported` | `true` → `dtype: auto` sélectionne `torch.bfloat16` |
| Débit `tiny-cpu` sur T4 | ~65 000–72 000 tokens/s |
| Débit `tiny-cpu` sur CPU local | ~5 700 tokens/s |
| Run de contrôle | 300 steps, 1 228 800 tokens, 17,7 s, `val_loss` 5,54365, `val_ppl` 255,608 |

Le checkpoint de contrôle produit (`checkpoints/colab-smoke/best`) fait 9,8 Mio,
soit exactement 2 557 632 paramètres en fp32. Il s'agit d'un **run de fumée de
300 steps**, pas d'un modèle entraîné : les générations qu'il produit n'ont pas
de sens. Il ne doit jamais être présenté comme un modèle.

## Installation et authentification

```bash
uv tool install google-colab-cli
colab sessions          # déclenche le flux OAuth : URL + code à recopier
```

Le flux affiche une URL, l'utilisateur l'ouvre dans n'importe quel navigateur,
approuve, puis colle le code affiché. Aucun serveur local ni navigateur n'est
nécessaire côté machine distante : le flux est conçu pour les conteneurs.

Le jeton est stocké dans le trousseau système. **Dans un conteneur sans D-Bus
(notre cas), il faut installer `keyrings.alt`**, sinon chaque commande redemande
une autorisation :

```bash
uv pip install --python "$(dirname "$(command -v colab)")/../bin/python" keyrings.alt
```

## Commandes utiles

```bash
colab new -s cortex0 --gpu T4      # alloue un runtime T4
colab status -s cortex0            # matériel, forme, état
colab exec -s cortex0 -f script.py # exécute un fichier Python local à distance
colab exec -s cortex0 --timeout 0  # désactive la limite de silence
colab upload/download -s cortex0 LOCAL REMOTE
colab stop -s cortex0              # libère le runtime
```

## Pilote automatisé

`scripts/colab.sh` encapsule le flux complet vérifié :

```bash
scripts/colab.sh up
scripts/colab.sh deps
scripts/colab.sh data
scripts/colab.sh train --config configs/tiny-cpu.yaml --max-steps 300
scripts/colab.sh logs
scripts/colab.sh fetch ./colab-checkpoints
scripts/colab.sh down
```

## Limitations vérifiées

1. **Session GPU limitée à 30 minutes.** La réponse d'allocation renvoie
`"fit": 1800`. Au-delà, le runtime est libéré : tout entraînement doit
sauvegarder et reprendre (`resume: true` est déjà actif dans les configs).
2. **`colab exec` abandonne après 30 s sans sortie** (`--timeout`, défaut 30).
Toute commande silencieuse (pip, entraînement, préparation de données) doit
soit passer `--timeout 0`, soit être lancée détachée avec un fichier de log.
C'est la raison pour laquelle `scripts/colab.sh` détache systématiquement.
3. **Aucun stockage persistant entre sessions.** Chaque runtime possède son
propre `/content`. Les artefacts doivent être téléchargés (ou poussés sur
Google Drive via `colab drivemount`) avant l'expiration.
4. **Quota.** Le compte utilisé affichait `0.00` unité de calcul ; le T4 reste
accessible sur l'offre gratuite mais l'allocation peut rester en file
d'attente (première tentative : > 180 s ; seconde : immédiate). Aucune
opération payante n'a été déclenchée.
5. **`bf16` sur T4.** Turing ne dispose pas de cœurs bf16 natifs ;
`torch.cuda.is_bf16_supported()` renvoie pourtant `true` et l'autocast bf16
fonctionne. Comme le dtype retenu est bf16, l'absence de `GradScaler` dans
`cortex/training/trainer.py` n'est pas bloquante. Forcer `dtype: float16`
exposerait en revanche à un sous-dépassement de gradient, faute de *loss
scaling*.
6. **Écart de vocabulaire des presets 50M/150M.** `cortex-50m.yaml` déclare
`vocab_size: 16384` et `cortex-150m.yaml` `32768`, alors que le tokenizer
produit a `vocab_size = 4096`. `scripts/train.py` aligne silencieusement
`vocab_size` sur le tokenizer : les comptes de paramètres réels sont donc
48 100 480 (et non 55 964 800) et 139 486 208 (et non 168 846 336). Il faut
soit ré-entraîner un tokenizer plus grand, soit assumer l'écart.
7. **`AGENTS.md` documente une commande invalide.** `prepare_data.py` n'accepte
pas `--config` ; l'invocation correcte est
`python scripts/prepare_data.py --vocab-size 4096`.
