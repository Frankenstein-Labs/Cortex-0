# Publication Hugging Face

## État

**Bloquée.** Aucun `HF_TOKEN` n'est disponible dans l'environnement de build.
Le dépôt de modèle `Frankenstein-Labs/Cortex-0` n'a donc **pas** été créé ni
alimenté. Le script de publication existe, est testé, et refuse d'inventer un
succès.

Dernière vérification :

```text
$ python scripts/publish_hf.py --folder export/Cortex-0
{"status": "blocked", "reason": "HF_TOKEN is not set", ...}
exit=2
```

## Ce qui est prêt

- `scripts/export_hf.py` produit un dossier complet :
  - `config.json` (`LlamaForCausalLM`) ;
  - `model.safetensors` (noms de tenseurs Llama) ;
  - `tokenizer.json` + `tokenizer_config.json` (`PreTrainedTokenizerFast`) ;
  - `README.md` (carte de modèle) ;
  - `generation_config.json` si présent dans le checkpoint.
- Vérification numérique : chargement avec `transformers.LlamaForCausalLM` et
  comparaison des logits. Résultat mesuré : `max_logit_diff = 0.0`,
  `tokenizer_match = true`.
- `scripts/publish_hf.py` crée le dépôt et téléverse le dossier.

## Procédure de déblocage

1. Créer un token **write** sur <https://huggingface.co/settings/tokens>.
2. Vérifier l'appartenance à l'organisation `Frankenstein-Labs` et les droits
   de création de dépôt.
3. Exporter :

   ```bash
   python scripts/export_hf.py \
     --model-dir checkpoints/tiny-cpu \
     --out export/Cortex-0 \
     --verify
   ```

4. Exporter le token dans l'environnement, sans l'écrire dans un fichier :

   ```bash
   export HF_TOKEN=hf_...
   ```

5. Publier :

   ```bash
   python scripts/publish_hf.py \
     --folder export/Cortex-0 \
     --repo-id Frankenstein-Labs/Cortex-0
   ```

6. Vérifier la page du dépôt et télécharger le modèle dans un environnement
   neuf :

   ```python
   from transformers import AutoTokenizer, LlamaForCausalLM
   tok = AutoTokenizer.from_pretrained("Frankenstein-Labs/Cortex-0")
   model = LlamaForCausalLM.from_pretrained("Frankenstein-Labs/Cortex-0")
   ```

## Contenu attendu du dépôt

```text
Frankenstein-Labs/Cortex-0/
├── README.md
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## Règles

- Ne jamais publier un checkpoint non entraîné comme modèle final.
- La carte de modèle doit indiquer la taille réelle, les données, les limites
  et l'absence de benchmarks externes.
- Ne jamais committer `HF_TOKEN`.
- Versionner les poids sur Hugging Face, pas dans Git.
- Un checkpoint par run, avec le commit Git correspondant dans la carte.

## Alternative si le token reste indisponible

- Publier uniquement le code et les scripts (GitHub).
- Joindre les petits poids (`model.safetensors`, ~10 Mo pour 2,6 M de
  paramètres) en release GitHub si la politique du dépôt l'autorise.
- Documenter clairement que le dépôt Hugging Face n'existe pas encore.
