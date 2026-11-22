# Corpus français de CORTEX-0

Ce document décrit le corpus **réellement construit et utilisé**, avec ses
sources, ses licences, ses volumes mesurés et ses limites. Les chiffres
proviennent de `datasets/french/manifest.json` et de `datasets/french/corpus.jsonl`.

## Sources et licences

| Source | Licence | Documents retenus | Caractères |
| --- | --- | --- | --- |
| `wikipedia_fr` (`wikimedia/wikipedia`, config `20231101.fr`) | CC BY-SA 4.0 | 2 537 | 35 899 736 |
| `project_code` (sources Python du dépôt) | Apache-2.0 | 2 | 5 340 |
| `gutenberg_fr` (livres du domaine public) | domaine public | **0** | 0 |
| **Total** | | **2 539** | **35 905 076** |

`gutenberg_fr` a échoué sur ce run : les URL `gutenberg.pglaf.org` n'ont pas
répondu depuis le runtime. L'échec est enregistré dans
`manifest.json → failed_sources`, il n'est pas silencieux.

## Volumes après tokenisation

Tokenizer BPE 4096 entraîné sur le corpus français
(`datasets/french/tokenizer`).

| Split | Documents | Tokens |
| --- | --- | --- |
| train | 2 483 | 11 324 207 |
| val | 27 | 95 983 |
| test | 29 | 123 676 |

## Nettoyage appliqué

1. Normalisation Unicode **NFC** — les accents sont conservés, jamais retirés.
2. Compactage des espaces et des sauts de ligne.
3. Rejet si `< 200` caractères, `> 40 000` caractères, ou `< 40` mots.
4. Rejet si la part de mots-outils français est `< 0,08` (heuristique, pas un
détecteur de langue validé).
5. Rejet si le ratio de mots uniques est `< 0,20` (texte répétitif).
6. Rejet si le texte est saturé de balisage (`<` dans plus d'un caractère sur 50).
7. **Déduplication exacte** par SHA-256 du texte normalisé.
8. **Déduplication approchée** par Jaccard sur des 5-shingles de mots
(seuil 0,85).

## Séparation des données

Le split est attribué par hachage de l'identifiant du document
(`split_for()`), donc un même document ne peut jamais apparaître dans deux
splits. Les 27 documents de validation et les 29 de test sont disjoints des
2 483 d'entraînement.

## Catégories

L'affectation est une **heuristique par mots-clés**, documentée comme telle :

| Catégorie | Documents |
| --- | --- |
| C professionnel | 1 330 |
| A fondamental | 796 |
| D scientifique | 269 |
| B conversationnel | 135 |
| E technique | 9 |

Aucune catégorie `F pédagogique` n'apparaît dans le préentraînement : les
données pédagogiques et conversationnelles sont fournies à l'étape **SFT**
(`datasets/sft/train.jsonl`), pas au préentraînement.

## Limites connues

1. **Le corpus est encyclopédique.** Wikipédia couvre bien l'explication et le
registre soutenu, mal le dialogue. La conversation vient du SFT, pas d'ici.
2. **Un seul domaine réel.** En dehors de Wikipédia, seuls 2 fichiers de code
ont passé les filtres. La couverture « informatique » du préentraînement est
donc faible ; elle repose sur le SFT et sur les connaissances du modèle.
3. **Le tokenizer 4096 est trop petit pour le français.** Mesuré par
`scripts/tokenizer_audit.py` : 2,12 tokens par mot, 61,2 % des mots découpés,
**83,6 % des mots accentués découpés**, 99,15 % des mots rares découpés. Voir
`experiments/tokenizer-audit-french-4096.json`.
4. **Les filtres sont heuristiques**, pas des classifieurs entraînés.

## Reproduction

```bash
python scripts/build_french_corpus.py \
    --out-dir datasets/french \
    --wikipedia-docs 3000 \
    --tokenize \
    --sft-dir datasets/sft
```

Le manifeste contient l'horodatage, les volumes par source, les rejets par
motif, les sources en échec, les licences et le SHA-256 du corpus.
