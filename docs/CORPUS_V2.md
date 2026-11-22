# Corpus français CORTEX-0 v2

Profil : `cortex-v2` — construit le 22 novembre 2026.
Répertoire : `datasets/cortex-v2/`.
Script reproductible : `scripts/build_corpus_v2.py`.

## Sources et licences

| Source | Langue | Documents gardés | Caractères | Licence |
|---|---|---:|---:|---|
| Candide (Gutenberg 4650) | fr | 58 | 202 161 | domaine public |
| L'Île mystérieuse (Gutenberg 14287) | fr | 321 | 1 209 533 | domaine public |
| Monte-Cristo tome 2 (17990) | fr | 181 | 707 579 | domaine public |
| Monte-Cristo tome 3 (17991) | fr | 174 | 673 858 | domaine public |
| Monte-Cristo tome 4 (17992) | fr | 186 | 716 617 | domaine public |
| Les Misérables tome 1 (17489) | fr | 179 | 658 399 | domaine public |
| Vingt mille lieues (5097) | fr | 238 | 877 509 | domaine public |
| Wikipédia FR (échantillon) | fr | 3 958 | 34 090 702 | CC BY-SA 4.0 |
| Textes éducatifs originaux | fr | 162 | 28 803 | Apache-2.0 |
| Code du projet | code | 90 | 520 090 | Apache-2.0 |

Attribution Wikipédia : texte extrait de Wikipédia en français, Wikimedia
Foundation, sous licence CC BY-SA 4.0. L'échantillon est un sous-ensemble
d'articles, pas un dump complet ; il ne doit pas être présenté comme la
totalité de Wikipédia.

Les textes éducatifs (grammaire, vocabulaire, sciences, informatique,
mathématiques, histoire, conversations et phrases soussou/malinké/bambara)
ont été écrits pour CORTEX-0. Ils sont volontairement courts et ne remplacent
pas un corpus natif de locuteurs pour les langues africaines.

## Nettoyage et déduplication

1. Normalisation des sauts de ligne, espaces et UTF-8.
2. Suppression des en-têtes/pieds Gutenberg.
3. Wikipédia : retrait des sections « Notes et références », « Liens
   externes », « Voir aussi », « Bibliographie », des tableaux et des URL.
4. Filtre qualité : 400 à 12 000 caractères, 95 % de caractères imprimables,
   pas de caractère de remplacement, ratio de mots français ≥ 0,08.
5. Déduplication exacte (SHA-256) et normalisée (minuscules, ponctuation et
   espaces retirés).
6. Aucun doublon exact ou normalisé conservé.

| Étape | Documents |
|---|---:|
| Documents bruts | 5 552 |
| Après filtre qualité | 5 386 |
| Doublons retirés | 0 |
| Documents finaux | 5 386 |

## Découpage

Le découpage est déterministe et indépendant : `sha256(document)[:8] mod 10000`.

| Split | Documents | Tokens | Usage |
|---|---:|---:|---|
| Train | 4 834 | 10 333 724 | entraînement + tokenizer |
| Validation | 264 | 574 473 | choix de modèle, perplexité |
| Test | 288 | 611 376 | évaluation finale, jamais utilisée pour le tokenizer |

Le tokenizer est entraîné **uniquement sur le train**. Aucun document du test
n'entre dans le tokenizer ni dans l'ajustement des hyperparamètres.

## Tokenizer

- Byte-level BPE, vocabulaire **8 192**.
- Tokens spéciaux : PAD=0, BOS=1, EOS=2, END=7.
- Entraîné sur le train uniquement, `min_frequency=2`.
- Tout texte UTF-8 est représentable ; aucun UNK.

## Limites honnêtes

- 10,3 M tokens restent peu pour un modèle de 3,3 M paramètres (≈3 tokens par
  paramètre) et très peu pour 15 M.
- Les livres Gutenberg sont du français littéraire du XIXe siècle, pas du
  français courant contemporain.
- Wikipédia est encyclopédique et parfois listé ; le nettoyage retire une
  partie des listes mais pas toute la structure.
- Les langues africaines ne sont représentées que par 15 phrases de
  conversation ; ce n'est pas un corpus linguistique suffisant.
- Le corpus ne remplace pas une validation par des locuteurs natifs.

## Reproduction

```bash
python scripts/build_corpus_v2.py --wiki-rows 6000
```

Le manifeste complet, avec SHA-256 des fichiers binaires, est dans
`datasets/cortex-v2/manifest.json`.
