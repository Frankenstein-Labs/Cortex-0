"""Proactive conversational companion.

The companion owns the conversational loop: retrieve memory, build a bounded
prompt, generate, persist the turn, and decide whether to consolidate.  It can
also produce a proactive message through the initiative engine.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..initiative.engine import Initiative, InitiativeEngine
from ..memory.store import MemoryStore
from ..model.generation import generate
from ..model.transformer import CortexForCausalLM
from ..planning.goals import GoalStore
from ..tokenizer.bpe import CortexTokenizer

DEFAULT_PERSONA_FR = (
    "Tu es CORTEX-0, une petite intelligence artificielle autonome développée par "
    "Frankenstein-Labs. Tu réponds en français, avec précision et honnêteté. Tu admets "
    "tes limites, tu ne prétends jamais avoir exécuté une action sans preuve, et tu "
    "n'inventes pas de souvenirs."
)


@dataclass
class ChatReply:
    text: str
    session_id: str
    used_model: bool
    memory_hits: List[dict]
    initiative: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "used_model": self.used_model,
            "memory_hits": self.memory_hits,
            "initiative": self.initiative,
        }


class Companion:
    def __init__(
        self,
        store: MemoryStore,
        model: Optional[CortexForCausalLM] = None,
        tokenizer: Optional[CortexTokenizer] = None,
        device: str | torch.device = "cpu",
        persona: str = DEFAULT_PERSONA_FR,
        initiative_engine: Optional[InitiativeEngine] = None,
        max_context_turns: int = 8,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 0,
    ) -> None:
        self.store = store
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        if model is not None:
            self.model.to(self.device)
        self.persona = persona
        self.initiative_engine = initiative_engine or InitiativeEngine(store)
        self.goals = GoalStore(store)
        self.max_context_turns = max_context_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    # ------------------------------------------------------------------- chat
    def chat(self, session_id: str, message: str, now: Optional[float] = None) -> ChatReply:
        now = now or time.time()
        message = message.strip()
        if not message:
            raise ValueError("message cannot be empty")

        self.store.add_event(session_id, "user", message)
        self.store.add(
            content=message,
            kind="working",
            importance=0.3,
            source="chat",
            session_id=session_id,
        )
        memory_hits = self.store.search(message, limit=6)
        goals = [goal.to_dict() for goal in self.goals.list(status="active", limit=5)]
        prompt = self._build_prompt(session_id, message, memory_hits, goals)

        if self.model is not None and self.tokenizer is not None:
            text = self._generate(prompt)
            used_model = True
        else:
            text = self._fallback_reply(message, memory_hits)
            used_model = False

        self._persist_turn(session_id, message, text, now)
        return ChatReply(
            text=text,
            session_id=session_id,
            used_model=used_model,
            memory_hits=[hit.to_dict() for hit in memory_hits],
        )

    def stream_chat(self, session_id: str, message: str, now: Optional[float] = None):
        """Yield incremental text deltas while persisting the final turn."""
        now = now or time.time()
        message = message.strip()
        if not message:
            raise ValueError("message cannot be empty")
        self.store.add_event(session_id, "user", message)
        self.store.add(
            content=message, kind="working", importance=0.3, source="chat", session_id=session_id
        )
        memory_hits = self.store.search(message, limit=6)
        goals = [goal.to_dict() for goal in self.goals.list(status="active", limit=5)]
        prompt = self._build_prompt(session_id, message, memory_hits, goals)

        if self.model is None or self.tokenizer is None:
            text = self._fallback_reply(message, memory_hits)
            yield text
            self._persist_turn(session_id, message, text, now)
            return

        ids = self.tokenizer.encode(prompt)
        max_context = self.model.config.max_position_embeddings
        max_new = min(self.max_new_tokens, max(1, max_context - 1))
        ids = ids[-max(1, max_context - max_new) :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        from ..model.generation import generate_stream

        produced: List[int] = []
        emitted = 0
        for token in generate_stream(
            self.model,
            input_ids,
            max_new_tokens=max_new,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=40,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_token_ids=[self.tokenizer.end_token_id],
            generator=generator,
            pad_token_id=self.tokenizer.pad_token_id,
        ):
            produced.append(int(token[0]))
            text = self.tokenizer.decode(produced, skip_special_tokens=True)
            if "\ufffd" in text:
                continue
            if len(text) > emitted:
                yield text[emitted:]
                emitted = len(text)
        final_text = self.tokenizer.decode(produced, skip_special_tokens=True).strip()
        self._persist_turn(session_id, message, final_text, now)

    def _persist_turn(self, session_id: str, message: str, text: str, now: float) -> None:
        self.store.add_event(session_id, "assistant", text)
        if re.search(r"(je préfère|j'aime|rappelle-toi|i prefer|remember that)", message, re.I):
            self.store.add(
                content=message,
                kind="preference",
                importance=0.9,
                source="chat",
                session_id=session_id,
            )
            self.store.set_preference(f"session:{session_id}:{int(now)}", message)
        turns = len(self.store.recent_events(session_id, limit=20))
        if turns and turns % 10 == 0:
            self.store.consolidate(session_id=session_id, min_events=6)

    def _build_prompt(
        self, session_id: str, message: str, memory_hits: List, goals: List[dict]
    ) -> str:
        if self.tokenizer is None:
            return message
        context_lines = [f"- {hit.content}" for hit in memory_hits[:5]]
        goal_lines = [f"- {goal['title']} ({goal['progress']:.0%})" for goal in goals]
        system = self.persona
        if context_lines:
            system += "\n\nMémoire pertinente :\n" + "\n".join(context_lines)
        if goal_lines:
            system += "\n\nObjectifs actifs :\n" + "\n".join(goal_lines)
        history = self.store.recent_events(session_id, limit=self.max_context_turns * 2)
        # Drop the just-stored user message; it is added explicitly below.
        if history and history[-1]["role"] == "user" and history[-1]["content"] == message:
            history = history[:-1]
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for turn in history:
            if turn["role"] in {"user", "assistant"}:
                messages.append({"role": turn["role"], "content": turn["content"][:1200]})
        messages.append({"role": "user", "content": message})
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def _generate(self, prompt: str) -> str:
        ids = self.tokenizer.encode(prompt)
        max_context = self.model.config.max_position_embeddings
        max_new = min(self.max_new_tokens, max(1, max_context - 1))
        ids = ids[-max(1, max_context - max_new) :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        output = generate(
            self.model,
            input_ids,
            max_new_tokens=max_new,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=40,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_token_ids=[self.tokenizer.end_token_id],
            generator=generator,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(output[0, len(ids) :].tolist(), skip_special_tokens=True)
        return text.strip() or self._fallback_reply("", [])

    @staticmethod
    def _fallback_reply(message: str, memory_hits: List) -> str:
        if memory_hits:
            return (
                "Je n'ai pas de modèle chargé pour générer une réponse. "
                f"J'ai trouvé en mémoire : « {memory_hits[0].summary} »."
            )
        return (
            "Je n'ai pas de modèle chargé pour générer une réponse. "
            "Je peux tout de même enregistrer ton message et le retrouver plus tard."
        )

    # ------------------------------------------------------------- proactivity
    def proactive(
        self,
        session_id: str,
        now: Optional[float] = None,
        user_last_active: Optional[float] = None,
    ) -> Optional[Initiative]:
        now = now or time.time()
        initiatives = self.initiative_engine.tick(
            session_id=session_id, now=now, user_last_active=user_last_active
        )
        if not initiatives:
            return None
        initiative = initiatives[0]
        self.initiative_engine.mark_sent(initiative.id, now=now)
        self.store.add_event(session_id, "assistant", initiative.message, meta={"proactive": True})
        return initiative
