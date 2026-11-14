"""Recognize plain conversation that does not need to become a Dream note."""

import re

from .chat import _asks_for_inventory, _wiki_lookup_query


_SMALL_TALK = re.compile(
    r"(?:hi|hey|hello)(?: there| again| vicena)?|good (?:morning|afternoon|evening|night)|"
    r"(?:okay|ok|sure|sounds good)(?: thanks| thank you)?|"
    r"(?:many |no |okay |ok )?thanks(?: a lot| so much| for the help| vicena)?|"
    r"thank you(?: so much| very much| for the help| vicena)?|"
    r"got it|understood|makes sense|great|perfect|yes|no|bye|goodbye"
)
_QUESTION_HEAD = re.compile(
    r"(?:(?:please )?(?:can|could|would|will) you\b|"
    r"(?:did|do|does|is|are|was|were|should|can|could|would|will|has|have|had) "
    r"(?:i|it|we|you|they|he|she|this|that|these|those|the)\b|"
    r"(?:what|which|who|whom|whose|where|when|why|how)\b)"
)
_PLAIN_QUESTION = re.compile(
    r"(?:(?:please )?(?:can|could|would|will) you|"
    r"(?:what|which|who|where|when|why|how) "
    r"(?:is|are|was|were|do|does|did|can|could|should|would|will|has|have)|"
    r"(?:what|who) (?:happened|changed))\b"
)
_SHORT_QUESTION = re.compile(r"(?:any (?:ideas|questions|suggestions)|thoughts|how are you|what's up)\?")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_MIXED_CLAUSE = re.compile(
    r"[:;]|[,–—]\s*(?:(?:but |because )?(?:i|we|our|my)\b|(?:because|since|given)\b)"
)


def _conversation_sentence(text):
    normalized = " ".join(text.casefold().split())
    if not normalized:
        return True
    if _asks_for_inventory(normalized) or _wiki_lookup_query(normalized):
        return True
    words = normalized.rstrip(".!? ").replace(", ", " ")
    if _SMALL_TALK.fullmatch(words) or _SHORT_QUESTION.fullmatch(normalized):
        return True
    # A question can still convey a new observation or instruction.
    if _MIXED_CLAUSE.search(normalized):
        return False
    if normalized.endswith("?"):
        return _QUESTION_HEAD.match(normalized) is not None
    return _PLAIN_QUESTION.match(words) is not None


def is_conversation_only(text):
    """Conservatively recognize turns with no evident content to preserve as a note."""
    return bool(text.strip()) and all(
        _conversation_sentence(sentence)
        for line in text.splitlines()
        for sentence in _SENTENCE_BREAK.split(line.strip())
    )
