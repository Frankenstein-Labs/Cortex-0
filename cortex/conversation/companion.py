"""Proactive conversational companion around the CORTEX-0 language model."""

from __future__ import annotations

import time
from dataclasses import dataclass

from cortex.generation.sampler import GenerationConfig, generate_text
from cortex.memory.store import MemoryStore, estimate_tokens

PERSONA = (
    "Tu es CORTEX-0, un petit modèle de langage développé par Frankenstein-Labs. "
    "Tu réponds de façon concise, honnête et curieuse. Tu ne prétends jamais avoir "
    "fait une action que tu n'as pas faite. Si tu ne sais pas, tu le dis."
)


@dataclass
class ChatReply:
    text: str
    session_id: str
    used_memories: list[str]
    context_tokens: int
    instruction_tuned: bool

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "used_memories": self.used_memories,
            "context_tokens": self.context_tokens,
            "instruction_tuned": self.instruction_tuned,
        }


class Companion:
    def __init__(
        self,
        model,
        tokenizer,
        memory: MemoryStore,
        persona: str = PERSONA,
        context_tokens: int = 900,
        instruction_tuned: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.persona = persona
        self.context_tokens = context_tokens
        self.instruction_tuned = instruction_tuned

    def _context(self, session_id: str, user_text: str) -> tuple[list[dict[str, str]], list[str]]:
        relevant = self.memory.search_episodes(user_text, limit=4)
        memories = [f"{ep.role}: {ep.content[:200]}" for ep in relevant]
        history = self.memory.recent_episodes(session_id, limit=12)

        system = self.persona
        if memories:
            system += "\nSouvenirs pertinents :\n" + "\n".join(f"- {m}" for m in memories)

        messages: list[dict[str, str]] = [{"role": "user", "content": "Instructions: " + system}]
        budget = self.context_tokens - estimate_tokens(system) - estimate_tokens(user_text)
        kept: list[dict[str, str]] = []
        for ep in reversed(history):
            cost = estimate_tokens(ep.content)
            if cost > budget:
                break
            budget -= cost
            kept.append({"role": "assistant" if ep.role == "assistant" else "user", "content": ep.content})
        messages.extend(reversed(kept))
        messages.append({"role": "user", "content": user_text})
        return messages, memories

    def reply(
        self,
        session_id: str,
        user_text: str,
        gen: GenerationConfig | None = None,
    ) -> ChatReply:
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("empty message")
        messages, memories = self._context(session_id, user_text)
        self.memory.add_episode(session_id, "user", user_text)
        prompt = self.tokenizer.chat_prompt(messages, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) > self.tokenizer_max_context():
            prompt_ids = prompt_ids[-self.tokenizer_max_context() :]

        gen = gen or GenerationConfig(max_new_tokens=80, temperature=0.8, top_k=50, top_p=0.95)
        text, _ = generate_text(self.model, self.tokenizer, prompt_ids, gen)
        if not text:
            text = "(pas de réponse générée)"
        self.memory.add_episode(session_id, "assistant", text)
        return ChatReply(
            text=text,
            session_id=session_id,
            used_memories=memories,
            context_tokens=len(prompt_ids),
            instruction_tuned=self.instruction_tuned,
        )

    def tokenizer_max_context(self) -> int:
        return getattr(getattr(self.model, "cfg", None), "max_seq_len", 512)

    def deliver_proactive(self, session_id: str, text: str) -> ChatReply:
        self.memory.add_episode(session_id, "assistant", text, importance=0.7, tags="proactive")
        return ChatReply(
            text=text,
            session_id=session_id,
            used_memories=[],
            context_tokens=estimate_tokens(text),
            instruction_tuned=self.instruction_tuned,
        )

    def summary(self, session_id: str) -> str:
        return self.memory.summarize_session(session_id)

    def status(self) -> dict:
        return {
            "persona": self.persona,
            "instruction_tuned": self.instruction_tuned,
            "context_tokens": self.context_tokens,
            "memory": self.memory.stats(),
            "time": time.time(),
        }
