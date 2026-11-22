# Publication Hugging Face

## État

**Publiée (dépôt privé).** Le 2026-11-22, les poids ont été téléversés dans
[`Frankenstein-Labs/Cortex-0-tiny`](https://huggingface.co/Frankenstein-Labs/Cortex-0-tiny).

Le dépôt `Frankenstein-Labs/Cortex-0` existait déjà et contient un autre projet
multimodal (`opendevin-multimodal`, ~15 M de paramètres, tokenizer 32k). Il n'a
**pas** été modifié. Le modèle Llama-compatible de ce dépôt est donc publié sous
le nom distinct `Cortex-0-tiny`.

Dernier commit du dépôt de modèle : `aa98076c8700f347547cfccabcfa65cb6e88d5d5`.
Le commit GitHub source est `da3108808579750d2dca891941eb7cd08a79c30e`.

## Contenu publié

```text
Frankenstein-Labs/Cortex-0-tiny/
├── README.md                 carte de modèle (FR/EN, limites, licences)
├── LICENSE                   Apache-2.0
├── config.json               LlamaForCausalLM, 2 557 632 paramètres
├── model.safetensors         checkpoint de préentraînement (39 tenseurs)
├── tokenizer.json            BPE 4096
├── tokenizer_config.json     template de chat CORTEX
├── generation_config.json
├── training_metrics.json     métriques mesurées et provenance
└── sft/
    ├── README.md
    ├── config.json
    ├── model.safetensors     checkpoint SFT (200 steps)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── generation_config.json
```

## Vérifications réellement exécutées

- Téléversement : 14 fichiers, puis mise à jour de la carte et du template.
- `AutoTokenizer` + `LlamaForCausalLM` chargés **depuis le Hub** : 39/39 tenseurs.
- Logits finis, forme `[1, 12, 4096]`, 2 557 632 paramètres.
- `apply_chat_template` avec le template CORTEX : la réponse SFT contient bien
  « Je suis CORTEX-0… ».
- Export local contre l'implémentation PyTorch d'origine :
  `max_logit_diff = 0.0`, `tokenizer_match = true`.

## Utilisation

```python
from transformers import AutoTokenizer, LlamaForCausalLM

repo = "Frankenstein-Labs/Cortex-0-tiny"
tokenizer = AutoTokenizer.from_pretrained(repo, token="hf_...")
model = LlamaForCausalLM.from_pretrained(repo, token="hf_...")
```

Pour le checkpoint SFT :

```python
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="sft", token="hf_...")
model = LlamaForCausalLM.from_pretrained(repo, subfolder="sft", token="hf_...")

messages = [
    {"role": "system", "content": "Tu es CORTEX-0, une petite intelligence artificielle autonome de Frankenstein-Labs."},
    {"role": "user", "content": "Qui es-tu ?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Le dépôt étant privé, `token=` est obligatoire. Pour le rendre public, une
autorisation explicite est requise ; l'opération se fait depuis les réglages du
dépôt Hugging Face ou avec `HfApi.update_repo_settings(..., private=False)`.

## Reproduire l'export et la publication

```bash
python scripts/export_hf.py \
  --model-dir checkpoints/tiny-cpu \
  --out export/Cortex-0 --verify

python scripts/export_hf.py \
  --model-dir checkpoints/tiny-cpu-sft \
  --out export/Cortex-0-SFT --verify

export HF_TOKEN=hf_...   # jamais écrit dans un fichier

python scripts/publish_hf.py \
  --folder export/Cortex-0 \
  --repo-id Frankenstein-Labs/Cortex-0-tiny \
  --private
```

Le script `publish_hf.py` crée le dépôt si nécessaire et téléverse le dossier.
Il refuse de continuer sans token et n'invente jamais un succès.

## Règles

- Ne jamais publier un checkpoint non entraîné comme modèle final.
- La carte de modèle indique la taille réelle, les données, les licences et les
  limites ; aucun benchmark externe n'est revendiqué.
- Ne jamais committer `HF_TOKEN`.
- Versionner les poids sur Hugging Face, pas dans Git.
- Un checkpoint par run, avec le commit Git correspondant dans la carte.
