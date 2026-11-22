"""CORTEX-0 - a small, from-scratch autonomous language model.

The package is organised in layers:

- ``cortex.model``        decoder-only Transformer (Llama-compatible)
- ``cortex.tokenizer``    byte-level BPE tokenizer and chat template
- ``cortex.training``     reproducible pretraining and supervised fine-tuning
- ``cortex.evaluation``   perplexity and reasoning benchmarks
- ``cortex.reasoning``    multi-sample reasoning, verification, GRPO
- ``cortex.memory``       persistent multi-level memory (SQLite)
- ``cortex.initiative``   proactive initiative engine and scheduler
- ``cortex.conversation`` conversational companion
- ``cortex.agents``       coding agent and isolated execution sandbox
- ``cortex.planning``     persistent missions, goals and checkpoints
- ``cortex.execution``    durable task queue and background workers
- ``cortex.api``          FastAPI service
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
