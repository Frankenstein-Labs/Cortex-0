# Feuille de route

Statut réel, sans extrapolation. Une case cochée signifie : code présent,
testé, et exécuté dans cet environnement.

## Phase 1 — Étude et conception

- [x] Architecture Llama-compatible documentée.
- [x] Choix RoPE, RMSNorm, SwiGLU, GQA justifiés.
- [x] Budget mémoire et contexte dimensionnés pour le CPU.
- [x] Distinction explicite entre mémoire externe et connaissances des poids.

## Phase 2 — Dépôts

- [x] Dépôt de code prêt (GitHub `Frankenstein-Labs/Cortex-0`).
- [x] Dépôt Hugging Face créé (privé) : `Frankenstein-Labs/Cortex-0-tiny`.
- [x] Export HF vérifié numériquement (`max_logit_diff = 0.0`).

## Phase 3 — Architecture du modèle

- [x] Embeddings, RoPE, attention causale, GQA, KV-cache.
- [x] RMSNorm, SwiGLU, résidus, tête de langage, embeddings liés.
- [x] Tests : causalité, cache incrémental, masque de padding, garde de contexte.
- [x] Presets 2,6 M, 50 M et 150 M.

## Phase 4 — Tokenizer et prédiction

- [x] BPE byte-level 4 096, 10 tokens spéciaux.
- [x] Aller-retour octet-exact (français, langues africaines, emoji, code).
- [x] Next-token prediction, logits, softmax.
- [x] Température, top-k, top-p, répétition penalty, greedy, streaming.

## Phase 5 — Préentraînement

- [x] Corpus multi-sources avec licences et SHA-256.
- [x] Tokenisation binaire `uint16`.
- [x] Trainer AdamW + cosine + warmup + clipping + AMP optionnel.
- [x] Checkpoints, reprise, meilleur modèle.
- [x] Run CPU 3 000 steps, ~5 700 tokens/s.

## Phase 6 — Tests et benchmarks

- [x] 89 tests unitaires et d'intégration.
- [x] Perplexité de validation.
- [x] Suite interne arithmétique/logique FR/EN.
- [ ] Benchmarks externes (MMLU, GSM8K) — non installés, non revendiqués.

## Phase 7 — Fine-tuning et raisonnement

- [x] SFT avec masquage des tours non-assistant.
- [x] Vérificateur arithmétique AST.
- [x] Best-of-N, révision, journalisation.
- [x] GRPO : avantages, clipping, KL, boucle expérimentale.
- [ ] Résultats de raisonnement forts — non obtenus avec 2,6 M de paramètres.

## Phase 8 — Mémoire et conversation

- [x] SQLite + FTS5, 7 types de mémoire.
- [x] Recherche hybride, consolidation extractive.
- [x] Préférences, mémoire projet, journal d'événements.
- [x] Compagnon conversationnel, template `Tu es CORTEX`, fallback honnête.

## Phase 9 — Autonomie et missions

- [x] Objectifs persistants.
- [x] Missions avec plan, checkpoints, reprise.
- [x] Nouvelles instructions sans rejouer le travail terminé.
- [x] File de tâches durable, workers, retries, idempotence.
- [x] Initiative : curiosité, idées, auto-évaluation, policy, scheduler.
- [x] Déduplication et limites de fréquence.

## Phase 10 — Compétences spécialisées

- [x] Agent de codage : lecture, écriture, tests, diff, commit préparé.
- [x] Sandbox avec jail de chemins et liste blanche.
- [x] Orchestrateur multi-agents avec dépendances.
- [ ] Compétences mathématiques fortes, recherche documentaire PDF, business.
- [ ] Langues africaines à grande échelle (tokenizer prêt, données à ajouter).

## Phase 11 — API et interface

- [x] FastAPI : génération, chat, streaming SSE, raisonnement, mémoire,
  objectifs, missions, tâches, initiatives.
- [x] CLI unifiée `cortex`.
- [x] Authentification optionnelle par clé.
- [ ] Interface web dédiée — l'API expose `/docs`.

## Phase 12 — Évaluation globale et publication

- [x] Documentation architecture, training, sécurité, évaluation.
- [x] Carte de modèle générée.
- [x] CI GitHub Actions.
- [x] Publication Hugging Face (privée) : base + SFT, vérifiée depuis le Hub.
- [ ] Rapport de recherche externe.

---

## Prochaines étapes prioritaires

1. **Publication publique** : décider si `Cortex-0-tiny` doit être rendu
   public (le dépôt est privé et déjà vérifié).
2. **GPU** : entraîner `configs/cortex-50m.yaml` sur un GPU gratuit (Colab,
   Kaggle) avec un corpus élargi.
3. **Données africaines** : constituer un corpus soussou/malinké/bambara
   licencié, entraîner un tokenizer dédié ou étendre le vocabulaire.
4. **Raisonnement** : SFT sur traces vérifiées, puis GRPO avec récompenses
   programmatiques sur un modèle plus grand.
5. **Isolation réelle** : remplacer le sandbox processus par un conteneur
   jetable pour l'agent de codage.
6. **Benchmarks externes** : intégrer un harnais reproductible (MMLU, GSM8K)
   uniquement quand les ressources permettent un run honnête.

## Blocages connus

| Blocage | Impact | Contournement |
| --- | --- | --- |
| Dépôt HF privé | poids non visibles publiquement | publication privée vérifiée |
| CPU seulement | modèle 2,6 M au lieu de 50–300 M | presets prêts, run GPU à faire |
| Pas de runtime conteneurisé | sandbox processus | approbations obligatoires |
| Pas de benchmarks externes | pas de comparaison publique | suite interne seulement |
