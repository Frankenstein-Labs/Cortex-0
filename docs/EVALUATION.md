# Évaluation

## Principe

Deux mesures, séparées et reproductibles :

1. **Perplexité** : qualité de la prédiction du token suivant sur un jeu de
   validation jamais utilisé pour l'entraînement.
2. **Suite interne** `cortex-internal-v1` : arithmétique et syllogismes, en
   français et en anglais.

Cette suite **n'est pas MMLU, ni GSM8K, ni un benchmark externe**. Elle est
petite, générée localement, avec une vérité terrain calculée. Elle sert à
détecter les régressions, pas à revendiquer une performance générale.

## Perplexité

- 132 854 tokens de validation tokenisés.
- Fenêtres de 256 tokens, stride 256.
- 50 batches de 8 fenêtres = 102 400 tokens évalués.
- Loss moyenne, perplexité `exp(loss)`, bits par token `loss / ln(2)`.

Résultat mesuré sur `checkpoints/tiny-cpu/best` :

| Métrique | Valeur |
| --- | --- |
| Loss | 3.6228 |
| Perplexité | 37.443 |
| Bits / token | 5.2266 |
| Tokens évalués | 102 400 |

## Suite de raisonnement

Chaque exemple est présenté comme :

```text
Question: <énoncé>
```

La réponse est extraite après `Réponse:` / `Answer:`, sinon par le dernier
nombre, sinon par `oui`/`non`. La comparaison normalise la casse, les espaces,
les virgules décimales et les synonymes `yes/no`.

### Arithmétique

20 problèmes par langue, graine fixe. Le modèle doit produire le résultat.

| Langue | Correct | Accuracy | Baseline majoritaire |
| --- | --- | --- | --- |
| Français | 0 / 20 | 0.00 | 0.05 |
| Anglais | 0 / 20 | 0.00 | 0.10 |

Interprétation : le modèle de 2,6 M de paramètres **n'a pas appris le format
de réponse arithmétique**. Il produit des nombres plausibles mais faux. Ce
n'est pas un échec du pipeline : c'est la limite attendue d'un modèle aussi
petit, entraîné sur un corpus littéraire.

### Logique

20 syllogismes par langue, réponse `oui`/`non`.

| Langue | Correct | Accuracy | Baseline majoritaire | Écart |
| --- | --- | --- | --- | --- |
| Français | 11 / 20 | 0.55 | 0.55 | 0.00 |
| Anglais | 14 / 20 | 0.70 | 0.60 | +0.10 |

Interprétation : en français, le modèle répond **toujours `non`** (20/20). Son
score est exactement celui de la classe majoritaire : aucun raisonnement
mesurable. En anglais, il alterne `oui` (8) et `non` (12) et dépasse la baseline
de 10 points, mais avec 20 exemples l'intervalle de confiance est large. Ce
résultat n'est **pas** une preuve de raisonnement ; il peut refléter un biais
de données ou du hasard.

Les distributions prédites et attendues sont enregistrées dans
`experiments/eval-tiny-cpu.json` pour permettre cette vérification.

## Ce que ces chiffres ne disent pas

- Ils ne comparent pas CORTEX-0 à GPT, Llama, Mistral ou DeepSeek.
- Ils ne mesurent pas la conversation, la mémoire ou l'autonomie.
- Ils ne mesurent pas les compétences en code, sciences, business ou langues
  africaines.
- Une accuracy de 70 % sur 20 exemples n'est pas un résultat statistiquement
  solide.

## Reproductibilité

```bash
python scripts/evaluate.py \
  --model-dir checkpoints/tiny-cpu/best \
  --val-data datasets/processed/val.bin \
  --output experiments/eval-tiny-cpu.json \
  --n 20 \
  --max-new-tokens 96
```

Le fichier de sortie contient :

- le commit Git (`git_commit`) ;
- l'horodatage UTC ;
- le device ;
- la configuration complète du modèle ;
- le nombre de paramètres total et hors embeddings ;
- les résultats par suite, avec baseline et distributions.

## Évaluations futures

Pour un modèle 50M/150M, ajouter :

- un harnais MMLU/GSM8K reproductible ;
- des évaluations de code (HumanEval ou équivalent) ;
- des évaluations de langues africaines avec locuteurs natifs ;
- des mesures de mémoire (rappel à long terme) ;
- des mesures d'initiative (pertinence, taux de rejet, doublons).

Tant que ces évaluations ne sont pas exécutées, elles ne doivent pas figurer
comme résultats.
