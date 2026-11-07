"""Immutable, request-local user/system messages for the original tokenizer.

No tokenizer facade, mutable engine-wide prompt or hidden conversation state is
used. Readiness, capacity calculations and actual generation must all call the
same ``chat_messages`` helper with the same PromptText instance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


MAX_USER_PROMPT_CHARACTERS = 8000
MAX_SYSTEM_PROMPT_CHARACTERS = 4000
DEFAULT_DESIGN_PROMPT = (
    "You build responsive browser apps. Return one complete HTML document in a single html code block, "
    "with inline CSS and JavaScript. Use no external libraries, fonts, images, APIs or network calls. "
    "Make visible controls work. Use box-sizing:border-box, responsive layouts, width:100% and max-width "
    "constraints; avoid fixed minimum widths and horizontal overflow. For a new app, build the requested "
    "app from scratch. For an edit, return the complete updated document. Keep code concise."
)


def _validate_user(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a prompt first.")
    if len(value) > MAX_USER_PROMPT_CHARACTERS:
        raise ValueError("Use a prompt shorter than 8,000 characters.")


def _validate_system(value):
    if value is not None and not isinstance(value, str):
        raise ValueError("System instructions must be text.")
    if value is not None and len(value) > MAX_SYSTEM_PROMPT_CHARACTERS:
        raise ValueError("Use system instructions shorter than 4,000 characters.")
    return value if value and value.strip() else None


class PromptText(str):
    """A user string carrying immutable system instructions for this request.

    This remains compatible with the existing parent string validation. JSON
    serializes its user text, so callers must record ``system_prompt`` separately.
    Ordinary ``strip``/concatenation creates plain strings; construct PromptText
    after building the complete visible user/update context.
    """
    __slots__ = ("_system_prompt",)

    def __new__(cls, user_text, system_prompt=None):
        _validate_user(user_text)
        system_prompt = _validate_system(system_prompt)
        value = super().__new__(cls, user_text)
        object.__setattr__(value, "_system_prompt", system_prompt)
        return value

    @property
    def system_prompt(self):
        return self._system_prompt

    def __setattr__(self, name, value):
        raise AttributeError("PromptText instructions are immutable.")

    def __delattr__(self, name):
        raise AttributeError("PromptText instructions are immutable.")

    def __reduce__(self):
        return type(self), (str(self), self.system_prompt)


def chat_messages(prompt):
    """Return actual chat roles without rewriting either instruction string."""
    _validate_user(prompt)
    system = _validate_system(prompt.system_prompt) if isinstance(prompt, PromptText) else None
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": str(prompt)})
    return messages


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
