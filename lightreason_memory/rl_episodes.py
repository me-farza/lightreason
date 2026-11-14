"""Synthetic chronological episodes for the experimental neural memory policy."""

import random


MEMORY_KEYS = ("lantern", "bicycle", "jacket", "mug", "backpack", "notebook", "scarf", "suitcase")
VALUES = ("red", "blue", "green", "yellow", "purple", "orange", "black", "white", "brown", "pink", "gray", "gold")
UNKNOWN = "UNKNOWN"
UNKNOWN_TARGET = len(VALUES)
_SPLITS = {"train": {0, 1, 2}, "validation": {3}, "test": {4, 5}}

_WRITES = {
    "train": (
        "My {key} is {value}.",
        "I remember that my {key} was {value}.",
        "For the record, the {key} is {value}.",
        "The {key} I own is {value}.",
        "This morning I found that the {key} is {value}.",
        "I verified the color: the {key} is {value}.",
        "The color of the {key} turned out to be {value}.",
        "I took a look just now. My {key} is {value}.",
    ),
    "validation": (
        "Please remember the {key} color is {value}.",
        "The color of my {key} is {value}.",
    ),
    "test": (
        "I checked and my {key} is {value}.",
        "As a fact, the {key} has the color {value}.",
        "I can confirm the {key} is {value}.",
        "About my {key}, its color is {value}.",
    ),
}
_CORRECTIONS = {
    "train": (
        "Correction: my {key} is actually {value}.",
        "I was mistaken earlier; the {key} is {value}.",
        "After checking again, my {key} is definitely {value}.",
        "Here is a revision: my {key} is {value}.",
    ),
    "validation": (
        "Update the color of my {key} to {value}.",
        "My earlier description of the {key} was wrong; its color is {value}.",
    ),
    "test": (
        "To correct my note about the {key}, its real color is {value}.",
        "I need to revise what I said: the {key} is {value}.",
        "I looked again; the correct color of my {key} is {value}.",
    ),
}
_HYPOTHETICALS = {
    "train": (
        "I wondered if the {key} might be {value}, but I do not know.",
        "Is it possible that the {key} is {value}?",
        "Only as a guess, my {key} might be {value}.",
        "In a fictional world my {key} could be {value}.",
        "Picture a version of the {key} that is {value}; I made this up.",
    ),
    "validation": ("Imagine that the {key} were {value}.",),
    "test": (
        "Maybe the {key} could be {value}; that is only a guess.",
        "Suppose for a story that the {key} is {value}.",
    ),
}
_QUESTIONS = {
    "train": (
        "What color is my {key}?",
        "Do you remember the color of my {key}?",
        "Which color did I say my {key} was?",
        "What hue do you recall for my {key}?",
        "Can you remind me which color the {key} has?",
    ),
    "validation": (
        "What did I record as the color of my {key}?",
        "What is the recorded color for my {key}?",
    ),
    "test": (
        "Can you tell me the hue of my {key}?",
        "Remind me what color I said the {key} was.",
        "Looking back at my notes, which color is the {key}?",
        "What color did I mention for the {key}?",
    ),
}
_UNRELATED = {
    "train": ("It rained all afternoon.", "The bus arrived early.", "Someone was playing piano."),
    "validation": ("I heard a bird at sunrise.", "We finished washing the dishes."),
    "test": ("I liked the music at dinner.", "The door opened without a sound.", "We walked back together."),
}
_TRANSIENT = {
    "train": (
        "For this exercise only, my {key} is {value}. Do not keep that as a lasting fact.",
        "As a temporary example, the {key} is {value}. This expires after the exercise.",
        "Only for today's exercise, my {key} is {value}. This is a short-lived label.",
    ),
    "validation": (
        "For this exercise only, the {key} is {value}. This label expires at the end.",
        "My {key} is {value} as a temporary example. This is for this exercise only.",
    ),
    "test": (
        "For this exercise only, my {key} is {value}. Do not treat this as a lasting fact.",
        "As a temporary example, my {key} is {value}. Use it only for the current exercise.",
        "Only for today's exercise, the {key} is {value}. Keep lasting facts separately.",
    ),
}


def _span(template, key, value=None):
    text = template.format(key=key, value=value)
    key_start = text.index(key)
    value_start = text.index(value, key_start + len(key)) if value is not None else None
    return text, key_start, value_start


def _values_for(slot, split):
    # Every slot and value occurs in training, but their pairings vary by split.
    return tuple(value for target, value in enumerate(VALUES) if (slot + target) % 6 in _SPLITS[split])


def _episode(rng, split, index, transient):
    observations, queries = [], []
    visible = {}
    original, retained, corrected, learned_late, unavailable = rng.sample(MEMORY_KEYS, 5)
    transient_key = rng.choice(tuple(key for key in MEMORY_KEYS if key not in (
        original, retained, corrected, learned_late, unavailable,
    ))) if transient else None

    def write(key, correction=False):
        choices = _values_for(MEMORY_KEYS.index(key), split)
        value = rng.choice(tuple(value for value in choices if value != visible.get(key)))
        templates = _CORRECTIONS if correction else _WRITES
        text, key_start, value_start = _span(rng.choice(templates[split]), key, value)
        observations.append({
            "text": text, "key": key, "value": value, "key_start": key_start, "value_start": value_start,
            "slot": MEMORY_KEYS.index(key), "target": VALUES.index(value),
            "is_distractor": False, "is_correction": correction,
        })
        visible[key] = value

    def distract(key=None):
        if key is None:
            text, key_start, value_start = rng.choice(_UNRELATED[split]), None, None
        else:
            choices = _values_for(MEMORY_KEYS.index(key), split)
            invented = rng.choice(tuple(value for value in choices if value != visible.get(key)))
            text, key_start, value_start = _span(rng.choice(_HYPOTHETICALS[split]), key, invented)
        observations.append({
            "text": text, "key": None, "value": None, "key_start": key_start, "value_start": value_start,
            "slot": None, "target": None, "is_distractor": True, "is_correction": False,
        })

    def read(key, kind):
        value = visible.get(key)
        text, key_start, _ = _span(rng.choice(_QUESTIONS[split]), key)
        queries.append({
            "text": text, "key": key, "key_start": key_start, "value": value, "kind": kind,
            "slot": MEMORY_KEYS.index(key), "target": UNKNOWN_TARGET if value is None else VALUES.index(value),
            "after_write": len(observations),
        })

    initial_keys = [original, retained, corrected]
    first = initial_keys.pop(rng.randrange(len(initial_keys)))
    write(first)
    read(first, "initial")
    read(learned_late, "unknown")
    read(unavailable, "unknown")

    # Vary chronology so a writer has to tell observations from distractions
    # from their contents. A correction always follows the original fact and
    # the related hypothetical, so those queries have the intended semantics.
    pending = [*initial_keys, "unrelated", "hypothetical", "guess", "correction", "new"]
    if transient:
        pending.append("transient")
    while pending:
        options = list(pending)
        if corrected not in visible and "hypothetical" in options:
            options.remove("hypothetical")
        if ("hypothetical" in pending or original not in visible) and "correction" in options:
            options.remove("correction")
        event = rng.choice(options)
        pending.remove(event)
        if event in initial_keys:
            write(event)
            read(event, "initial")
        elif event == "unrelated":
            distract()
            read(retained if retained in visible else first, "retention")
        elif event == "hypothetical":
            distract(corrected)
            read(corrected, "distractor")
        elif event == "guess":
            distract(unavailable)
            read(unavailable, "unknown")
        elif event == "correction":
            write(corrected, correction=True)
            read(corrected, "correction")
            read(original, "retention")
        elif event == "transient":
            key = transient_key
            value = rng.choice(_values_for(MEMORY_KEYS.index(key), split))
            text, key_start, value_start = _span(rng.choice(_TRANSIENT[split]), key, value)
            observations.append({
                "text": text, "key": key, "value": value, "key_start": key_start, "value_start": value_start,
                "slot": MEMORY_KEYS.index(key), "target": VALUES.index(value),
                "is_distractor": False, "is_correction": False, "is_transient": True,
            })
            read(key, "transient")
        elif event == "new":
            write(learned_late)
            read(learned_late, "initial")
    read(corrected, "correction")
    read(retained, "retention")
    read(unavailable, "unknown")
    if transient:
        read(transient_key, "transient")
    return {"episode_id": f"{split}-{index:05d}", "observations": observations, "queries": queries}


def make_episodes(seed, count, split="train", *, transient=False):
    """Return deterministic examples with chronological metadata and disjoint factual pairs.

    The policy receives only natural-language observation or query text. The
    slot, target, and span fields label actions and permit evaluation. A query's
    `after_write` counts all observations already processed, including distractors.
    """
    if split == "valid":
        split = "validation"
    if split not in _SPLITS:
        raise ValueError("split must be train, validation, or test")
    if type(seed) is not int or type(count) is not int or count < 1:
        raise ValueError("seed must be an integer and count must be a positive integer")
    rng = random.Random(f"{seed}:{split}")
    return [_episode(rng, split, index, transient) for index in range(count)]
