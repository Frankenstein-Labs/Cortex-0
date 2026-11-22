# SFT v6 — dataset équilibré

Fichier principal : `datasets/cortex-v2/sft-v6.jsonl`.
Script reproductible : `scripts/build_sft_v6.py`.
Format : `messages` (system/user/assistant), entraînement **plain** avec
`PlainSFTDataset` et `collate_sft`.

## Composition

| Catégorie | Exemples | Part |
|---|---:|---:|
| Instructions (grammaire, vocabulaire, transformations, résumés) | 215 | 40 % |
| Mathématiques vérifiées | 181 | 34 % |
| Connaissances (sciences, informatique, histoire) | 55 | 10 % |
| Conversation / persona | 30 | 6 % |
| Code Python exécutable | 25 | 5 % |
| Refus de sécurité | 16 | 3 % |
| Incertitude (météo, avenir, privé) | 10 | 2 % |
| Anglais | 10 | 2 % |
| **Total** | **542** | **100 %** |

Les refus ne dominent plus : 16 exemples sur 542 (3 %), avec des formulations
variées. Les réponses sont courtes (42,5 caractères en moyenne) et les
questions sont uniques.

## Vérifications

- **Contamination** : 23 questions générées ont été retirées parce qu'elles
reproduisaient exactement un prompt de `cortex-100-fr`. Aucun prompt du
benchmark n'est présent dans le SFT final.
- **Déduplication** : 51 doublons de prompt retirés ; 542 prompts uniques.
- **Code** : les 25 exemples sont exécutés dans un sous-processus isolé
(`python -I`) avec des assertions ; un exemple qui échoue fait échouer la
construction du dataset.
- **Mathématiques** : les réponses sont calculées par le générateur, pas
recopiées à la main.
- **Masquage** : seuls les tokens de la réponse et l'EOS sont supervisés ; la
question et le prompt système sont à `-100`.

## Découpage

| Split | Exemples |
|---|---:|
| Train | 461 |
| Validation | 58 |
| Test | 23 |

Le découpage est groupé par prompt : une question ne peut pas être dans le
train et la validation.

## Limites

- 542 exemples restent petits pour un modèle de 3,3 M paramètres. Le SFT peut
encore surapprendre ; la validation et l'arrêt précoce sont obligatoires.
- Les instructions générées couvrent des transformations simples ; elles ne
remplacent pas un corpus d'instructions humaines.
- Le code couvre 25 fonctions courtes ; ce n'est pas un benchmark de
programmation.
- Les langues africaines ne sont pas encore un axe SFT quantitatif.

## Reproduction

```bash
python scripts/build_sft_v6.py
python -m pytest -q tests/test_pretraining_audit.py
```
