# Expériences de reconstruction — novembre 2026

Objectif : vérifier si un corpus français plus grand, un tokenizer 8192 et un
SFT équilibré suffisent à sauver la base CORTEX-0, **sans augmenter la taille
du modèle**. Tous les entraînements sont sur CPU ; aucun résultat n'est simulé.

## Dispositif

| Élément | Ancien | Nouveau |
|---|---|---|
| Corpus | 5,9 M tokens (3 livres FR + 2 EN + wiki) | 10,3 M tokens (7 livres FR + Wikipédia + textes originaux + code) |
| Tokenizer | BPE 4096 | BPE 8192 |
| Modèle témoin | 2,56 M | 3,34 M |
| Contexte | 256 | 256 |
| Steps | 3 000 (12,3 M tokens) | 1 500 (6,1 M tokens) |
| SFT | v3 789 exemples, 56 % templates | v6 542, v7 1 553, v8 4 162 avec traces |

## E1 — Préentraînement contrôlé

Comparaison à **steps et tokens vus égaux** (1 500 steps, batch 16, contexte
256, 6 144 000 tokens), sur le **même texte de test français v2** tokenizé par
le tokenizer de chaque modèle.

| Métrique | Ancien 1500 | Nouveau 1500 |
|---|---:|---:|
| Bits/caractère | 2,020 | **1,951** |
| Perte (même texte) | 3,964 | 4,326 |
| Tokens du texte | 42 412 | 37 532 |
| Perplexité (même texte) | 52,7 | 75,6 |
| Complétion `cortex-100-fr` | 19 % | **88 %** |
| Score `cortex-100-fr` | 0/100 | 0/100 |

Le nouveau tokenizer réduit le nombre de tokens de 11,5 % sur le même texte
(3,20 contre 2,83 caractères/token). En bits par caractère, qui neutralise la
taille du vocabulaire, le nouveau modèle est meilleur de **3,4 %**. Il apprend
aussi à produire EOS : 88 % de réponses terminées contre 19 %.

La perte de validation absolue du nouveau modèle (4,575, perplexité 97) n'est
pas comparable à l'ancienne (3,906, perplexité 49,7) : le corpus v2 contient
Wikipédia, plus difficile que trois romans. La baseline unigramme v2 est à
6,922 (perplexité 1 014), donc le modèle apprend bien (facteur 11), mais reste
loin d'un modèle de langage utile.

## E2 — SFT : quatre tentatives

| Run | Base | Données | Steps | Val loss | `cortex-100-fr` |
|---|---|---|---:|---:|---:|
| SFT v3 plain | tiny 2,56 M | 789 ex. | 300 | 1,61 | 9/100 |
| SFT v6 | tiny 2,56 M | 542 ex. | 300 | 3,29 | 11/100 |
| SFT v6 | v2 3,34 M | 542 ex. | 400 | 4,22 | 10/100 |
| SFT v7 | v2 3,34 M | 1 553 ex. | 700 | 3,75 | **12/100** |
| SFT v8 | v2 3,34 M | 4 162 ex. + traces | 1 200 | **0,848** | 11/100 |

Détail des scores :

| Run | Compréhension | Grammaire | Vocabulaire | Résumé | Instructions | Maths | Python | Sciences | Informatique | Refus | Anti-rép. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT v6 (2,56 M) | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/15 | 0/15 | 0/7 | 0/7 | 10/10 | 1/6 |
| SFT v7 (3,34 M) | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 1/15 | 0/15 | 0/7 | 0/7 | 10/10 | 1/6 |
| SFT v8 (3,34 M) | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/15 | 0/15 | 0/7 | 0/7 | 10/10 | 1/6 |

Les refus sont la seule compétence stable (10/10). Le reste n'est pas appris.
Le score de 12/100 de v7 est obtenu avec 10 refus, 1 math et 1 anti-répétition.

## E3 — Le piège des traces arithmétiques

SFT v8 ajoute 2 981 exemples de traces de décomposition (centaines, dizaines,
unités, retenue). La perte de validation tombe à **0,848** (perplexité 2,34),
ce qui pourrait faire croire à un succès. Deux contrôles montrent le contraire :

1. Sur les 15 questions mathématiques de `cortex-100-fr` : **0/15**.
2. Sur 80 additions tenues à l'écart, même format, mêmes templates :
   **0/80 exactes**.

Exemple de sortie :

```text
Question: Combien font 456 + 622 ?
Réponse: ... Centaines : 400 + 600 = 1100. Dizaines : 0 + 10 = 10.
Unités : 9 + 0 = 0 avec une retenue de 1. Total : ... = 1301.
```

Le modèle a appris la **forme** de la trace, pas l'algorithme : il copie mal
les chiffres (456 devient 9 + 0), invente des retenues et produit un total
incohérent. La faible perte de validation mesure la reproduction des
templates, pas la généralisation arithmétique. C'est exactement le piège que
l'audit voulait éviter.

## Diagnostic

1. **Capacité** : 3,34 M paramètres est trop petit pour combiner langue,
   instructions, code et arithmétique. Les 15 M et 50 M n'ont pas encore été
   entraînés sur ce corpus ; le CPU seul rend l'expérience longue.
2. **Tokenisation des nombres** : le BPE 8192 regroupe les chiffres
   (`456`, `622`), ce qui rend l'arithmétique positionnelle difficile. Un
   tokenizer avec chiffres séparés est la prochaine hypothèse à tester.
3. **Données SFT** : 542 à 4 162 exemples restent peu pour un modèle de cette
   taille ; les templates produisent une fausse validation basse.
4. **Préentraînement** : 6,1 M tokens vus, soit 1,8 token/paramètre. Le modèle
   est sous-entraîné même pour le français.
5. **Évaluation** : la suite 100 est plus dure que les templates SFT ; c'est
   voulu. Un score de 12/100 est un résultat négatif honnête, pas un benchmark
   gonflé.

## Décision

- **Ne pas lancer** de nouveaux entraînements aveugles sur 3,34 M.
- Prochaine étape scientifique : tokenizer avec chiffres séparés + 50 M de
  tokens français + 50 M de paramètres, sur GPU gratuit (Colab/Kaggle), avec
  la même évaluation tenue à l'écart.
- Conserver `cortex-100-fr` comme test de non-régression ; ne jamais
  l'utiliser pour le SFT.
- Le corpus v2 et le SFT v6/v7/v8 sont conservés comme base reproductible,
  pas comme preuve de performance.

## Reproductibilité

```bash
python scripts/build_corpus_v2.py --wiki-rows 6000
python scripts/build_sft_v6.py
python scripts/build_sft_v7.py
python scripts/build_arithmetic_sft.py
python scripts/train.py --config configs/experiments/tiny-v2.yaml \
  --max-steps 1500 --out-dir checkpoints/exp-tiny-v2-pretrain --no-resume
python scripts/train_sft.py \
  --model-dir checkpoints/exp-tiny-v2-pretrain/checkpoint-1500 \
  --data datasets/cortex-v2/sft-v7-train.jsonl \
  --val-data datasets/cortex-v2/sft-v7-val.jsonl \
  --out checkpoints/exp-v2-sft-v7 --steps 700 --batch-size 8 \
  --lr 0.0002 --format plain --no-resume
python scripts/eval_candidates.py \
  --model new-sft-v7=checkpoints/exp-v2-sft-v7/best
```

Les fichiers JSON complets sont dans `experiments/eval-v2/`.
