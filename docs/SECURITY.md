# Sécurité et fiabilité

## Modèle de menace

CORTEX-0 est conçu pour un usage **local, mono-utilisateur, expérimental**.
Les actifs à protéger sont :

- les secrets de l'environnement (`GITHUB_TOKEN`, `HF_TOKEN`, clés SSH, etc.) ;
- le contenu du dépôt et du workspace ;
- l'intégrité des poids et des checkpoints ;
- la machine hôte.

Les menaces considérées :

1. **Sortie de modèle non fiable** : le modèle peut produire du texte faux,
   répétitif ou contenant des instructions malveillantes.
2. **Injection de prompt via des fichiers** : un fichier lu par l'agent peut
   contenir « ignore les instructions et exécute … ». Le texte est traité
   comme une donnée, jamais comme une autorisation.
3. **Action de tool excessive** : écriture hors workspace, commande destructrice,
   `git push`, exfiltration de secrets.
4. **Code Python malveillant** exécuté dans le sandbox.
5. **Fuite de secrets** dans les logs, les commits ou les réponses HTTP.

## Garanties implémentées

### Workspace et chemins

`Sandbox.resolve` et `Workspace` résolvent chaque chemin et vérifient qu'il
reste sous la racine du workspace. `../`, les chemins absolus et les liens
symboliques qui sortent sont refusés (`SandboxError`).

### Exécutables

`sandbox.py` n'accepte que des exécutables en liste blanche :
`python3`, `pytest`, `git`, `ls`, `cat`, `head`, `tail`, `grep`, `find`, `wc`,
`sed`, `awk`, `cp`, `mv`, `mkdir`, `touch`, `diff`, `ruff`, `mypy`, `black`.
Toute autre commande est refusée avant exécution.

### Git

Les commandes contenant `push`, `remote`, `config`, `credential`, `filter-branch`
ou `--force` sont refusées. L'agent peut voir le diff et préparer un message de
commit, mais **ne pousse jamais**.

### Environnement

Les variables dont le nom contient `TOKEN`, `SECRET`, `PASSWORD`, `KEY`,
`CREDENTIAL` ou `AUTH` sont supprimées de l'environnement des sous-processus.
Passer explicitement une telle variable à `Sandbox.run` lève une erreur.

### Ressources

- timeout par défaut : 60 s (configurable par appel) ;
- `RLIMIT_CPU` : timeout + 2 s ;
- `RLIMIT_AS` : 2 Go ;
- `RLIMIT_FSIZE` : 50 Mo ;
- `RLIMIT_NOFILE` : 256 descripteurs ;
- taille de sortie tronquée à 20 000 caractères ;
- nombre de fichiers listés limité.

### Approbations

`CodingAgent` classe les actions :

| Action | Risque | Approbation |
| --- | --- | --- |
| `list_files`, `read_file`, `git_status`, `git_diff` | faible | non |
| `write_file`, `run_command`, `run_tests`, `git_commit` | élevé | oui |

Sans `auto_approve`, l'action est mise en attente, l'état est persisté, et
`approve()` l'exécute après redémarrage éventuel. Le mode `auto_approve` est
réservé aux tests et aux environnements jetables.

### API

- `/health` est public ;
- les routes `/v1/*` exigent `x-api-key` **si** `CORTEX_API_KEY` est défini ;
- sans clé, le serveur doit rester sur `127.0.0.1` ;
- les erreurs ne renvoient jamais de secret ni de trace complète.

### Secrets

- Aucun token n'est écrit dans le dépôt, les logs ou les commits.
- `.env`, `*.sqlite`, `*.db` sont ignorés par Git.
- `publish_hf.py` lit `HF_TOKEN` depuis l'environnement et ne l'affiche jamais.
- Le token Hugging Face est **absent** de cet environnement : la publication
  est bloquée, pas simulée.

## Non-garanties (important)

Le sandbox est un **processus**, pas un conteneur. Il ne fournit pas :

- d'isolation noyau (namespace, seccomp, cgroups) ;
- d'isolation réseau : du code Python peut ouvrir une socket ;
- de protection contre la lecture de fichiers hors workspace par du code
  Python arbitraire (`open('/etc/passwd')`) ;
- de protection contre une élévation de privilèges.

Conséquences :

- n'exécutez pas de code non fiable avec `auto_approve` ;
- ne lancez pas l'API sur une interface publique sans clé et sans reverse proxy ;
- pour un vrai isolement, utilisez un conteneur jetable, gVisor, Firecracker ou
  Kubernetes avec des politiques réseau.

Ces limites sont explicites pour éviter une fausse sensation de sécurité.

## Sécurité du modèle

CORTEX-0 n'est **pas aligné**. Il n'a pas de filtre de sécurité, pas de RLHF,
pas de modération. Il peut produire des erreurs, des répétitions ou du contenu
inapproprié. Il ne doit pas être présenté comme un conseiller médical,
juridique ou financier.

## Données et licences

- Textes du domaine public (Project Gutenberg, TinyShakespeare).
- Wikipédia FR : CC BY-SA 4.0, attribution requise pour toute redistribution
  substantielle.
- Code du projet : Apache-2.0.

Aucun dataset contenant des données personnelles n'est utilisé.

## Journalisation et récupération

- Chaque exécution d'agent écrit son plan, ses actions et ses résultats.
- Les checkpoints sont conservés ; `best` permet un retour arrière.
- La file de tâches persiste les tentatives, les erreurs et les résultats.
- Les initiatives sont horodatées et peuvent être ignorées.

## Signaler un problème

Ouvrir une issue privée ou contacter Frankenstein-Labs. Ne pas inclure de
secret dans le rapport. Pour une vulnérabilité, fournir les étapes de
reproduction et l'impact estimé.
