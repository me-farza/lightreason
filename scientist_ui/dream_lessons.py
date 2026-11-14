"""Check proposed Dream lessons against the words in a captured note."""

import re


_QUESTION = {
    "process": "What is the usual action for {key}?",
    "preference": "What do I prefer for {key}?",
    "experiment": "What is the status of experiment {key}?",
}
_SIGNAL = {
    "process": re.compile(r"\b(?:always|must|routinely|standard|usual|first (?:move|step|action)|we start)\b", re.I),
    "preference": re.compile(r"\b(?:prefer|preference|usually (?:choose|use|pick)|go-to)\b", re.I),
    "experiment": re.compile(r"\b(?:experiment|run|study|trial|test)\b", re.I),
}
_STATUS = re.compile(
    r"(?:complete(?:d)?|done|finished|failed|running|paused|blocked|pending|cancelled|"
    r"canceled|scheduled|in progress|on hold|awaiting [\w -]+)", re.I,
)
_UNSAFE_CONTEXT = re.compile(
    r"[?;]|\b(?:maybe|perhaps|might|could|would|should|probably|tentative(?:ly)?|example|hypothetical|"
    r"used to|previously|formerly|only|unless|except|depending|whether|if|until|then|instead|"
    r"never|avoid|not|don't|do not|cannot|can't|ignore|pretend|imagine|otherwise|but|or)\b", re.I,
)
_ACTION = re.compile(
    r"(?:apply|blind|check|choose|clean|collect|compare|confirm|discard|inspect|keep|label|log|"
    r"measure|prepare|record|repeat|replace|rinse|run|select|set|start|stop|test|use|verify|wait|weigh)\s+\S", re.I,
)


def grounded_lesson(capture, proposal):
    """Return one compact lesson if the proposed topic and answer are explicitly stated."""
    if not isinstance(proposal, dict) or proposal.get("capture_id") != capture["id"]:
        return None
    category = proposal.get("category")
    if not isinstance(category, str) or category not in _QUESTION:
        return None
    key, answer, evidence = (proposal.get(field) for field in ("key", "answer", "evidence"))
    if not all(isinstance(value, str) and value == value.strip() and "\n" not in value and value
               for value in (key, answer, evidence)):
        return None
    if (len(key) > 60 or not 2 <= len(key.split()) <= 8 or len(answer) > 100 or len(answer.split()) > 12
            or len(evidence) > 320 or not re.fullmatch(r"[\w][\w -]*", key)):
        return None
    text = capture["text"].strip()
    # Extractive validation cannot infer which qualifiers survive a summary of a multi-step note.
    if text != evidence or "`" in text or _UNSAFE_CONTEXT.search(text) or not _SIGNAL[category].search(evidence):
        return None

    def span(value):
        return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", evidence, re.I)

    topic, value = span(key), span(answer)
    if not topic or not value or value.end() < len(evidence.rstrip(".! ")) or topic.end() > value.start():
        return None
    if re.search(r"[,!]|\s(?:and|with the exception of)\s|\.\s", answer, re.I):
        return None
    if category == "process" and not _ACTION.match(answer):
        return None
    if category == "experiment" and not _STATUS.fullmatch(answer):
        return None
    return {
        "capture_id": capture["id"], "source": capture["source"], "category": category,
        "question": _QUESTION[category].format(key=key), "answer": answer, "key": key,
        "observation": evidence, "neural_eligible": True, "training_kind": category,
        "origin": "gemini_extraction",
        "reward": "Exact stated value earns +1; a different value earns -1.",
    }
