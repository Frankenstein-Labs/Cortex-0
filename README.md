# CORTEX-0

**Small Autonomous Language Model — Frankenstein-Labs**

CORTEX-0 commence par une base honnête et mesurable : un Transformer decoder-only implémenté avec PyTorch, entraînable par prédiction autoregressive du prochain token et exécutable localement.

## État du premier jalon

Déjà implémenté :

- architecture decoder-only Transformer ;
- RMSNorm, connexions résiduelles et causal self-attention ;
- embeddings de tokens et de positions ;
- poids liés entre embeddings et tête de langage ;
- calcul des logits et de la cross-entropy next-token ;
- génération avec température et top-k ;
- tokenizer UTF-8 byte-level sans API de modèle externe ;
- dataset causal à fenêtres ;
- configuration `configs/tiny.json` ;
- tests de forme, perte, génération et aller-retour tokenizer.

Ce jalon **n’est pas encore un modèle entraîné** et ne prétend pas comprendre, raisonner ou être conscient. Les capacités de mémoire, conversation, initiative, outils et autonomie seront ajoutées après la validation du noyau d’apprentissage.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Exemple minimal

```python
import torch
from cortex.model import CortexConfig, CortexLM
from cortex.tokenizer import ByteTokenizer

tokenizer = ByteTokenizer()
model = CortexLM(CortexConfig(d_model=128, n_layers=4, n_heads=4, d_ff=512))
prompt = torch.tensor([tokenizer.encode("Bonjour")])
output = model.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=40)
print(tokenizer.decode(output[0].tolist()))
```

## Feuille de route

1. Valider ce noyau sur CPU et GPU.
2. Ajouter un script d’entraînement reproductible et des checkpoints safetensors.
3. Remplacer le tokenizer de baseline par BPE/SentencePiece évalué sur un corpus contrôlé.
4. Ajouter des évaluations de perplexité et de génération.
5. Ajouter fine-tuning conversationnel, mémoire externe et API.
6. Ajouter raisonnement, planification et autonomie sous permissions explicites.

Les poids, datasets et résultats expérimentaux seront versionnés séparément du code source.
