"""Two-day chat rehearsal for a synthetic laboratory decision workflow."""

DAYS = (
    (
        "For this synthetic rehearsal, when a reference signal drifts, replace the reference first "
        "to check whether the instrument is drifting.",
        "For this synthetic rehearsal, when a person scores subjective sample features, hide the "
        "sample groups from the scorer to prevent expectation bias.",
    ),
    (
        "Correction for the synthetic rehearsal: if the reference signal drifts, use a sealed blank "
        "first. Our reference replacement didn't fix the drift. The sealed blank isolates ambient drift.",
    ),
)

_ACTIONS = {
    "fresh_reference": "Replace the reference; check instrument drift.",
    "sealed_blank": "Use a sealed blank; isolate ambient drift.",
    "blind_scorer": "Blind the scorer; prevent expectation bias.",
}

_DRIFT = (
    ("drift-thermal", "A thermal camera's reference signal drifts slowly despite identical readings at "
     "the start of each run. What check should we do first?"),
    ("drift-optical", "Our optical detector's baseline drifts between successive standard measurements. "
     "How should we investigate the changing reference signal?"),
    ("drift-pressure", "The reference signal on a pressure monitor gradually shifts during repeated "
     "measurements. Which control would help identify the source, and why?"),
)

_SCORING = (
    ("score-surface", "An operator will score the roughness of coated surfaces by eye. What procedure "
     "reduces subjective scoring bias across coating groups?"),
    ("score-film", "A technician will assign visual clarity scores to coated film samples from two "
     "groups. What precaution should we take before subjective scoring?"),
    ("score-dye", "An evaluator will judge whether dye is distributed uniformly across perforated "
     "test coupons. How should we organize this subjective color assessment, and why?"),
    ("score-panel", "Two people will compare how uniformly the pigment spreads across fired glaze "
     "panels. How should we prepare the samples for this human scoring, and why?"),
    ("score-adhesive", "A person has to judge the appearance of dried adhesive from competing "
     "formulations. What would help make their subjective ratings fair?"),
)

_OTHER = (
    ("other-slide", "How should I order three slides in a short team update?"),
    ("other-export", "Which file format should I use to share a table of results with a colleague?"),
    ("other-calendar", "How should I arrange this week's work blocks in my calendar?"),
)

CHECKS = (
    {"id": "transfer", "question": "On a new humidity rig, the reference signal creeps upward with "
     "each measurement. What would you do first, and why?",
     "day_one": "fresh_reference", "day_two": "sealed_blank"},
    {"id": "retention", "question": "In a different trial, a person has to judge how evenly each "
     "sample's color is spread. How should we set up this subjective scoring, and why?",
     "day_one": "blind_scorer", "day_two": "blind_scorer"},
    {"id": "abstention", "question": "How do I choose a color palette for my conference poster?",
     "day_one": None, "day_two": None},
)


def lesson_key(text):
    """Normalize the scripted lesson for matching after chat capture."""
    return " ".join(text.casefold().split())


def training_batch(captures):
    """Build training and held-out examples from completed scripted chat lessons."""
    original_lessons = {lesson_key(lesson) for lesson in DAYS[0]}
    correction = lesson_key(DAYS[1][0])
    taught = set()
    corrected = False
    for row in captures:
        for line in row["text"].splitlines():
            key = lesson_key(line)
            if key in original_lessons:
                taught.add(key)
            elif key == correction and taught == original_lessons:
                corrected = True
    if taught != original_lessons:
        return None

    drift_action = "sealed_blank" if corrected else "fresh_reference"
    expected_field = "day_two" if corrected else "day_one"
    return {
        "labels": dict(_ACTIONS),
        "supersedes": {"sealed_blank": "fresh_reference"} if corrected else {},
        "training": [
            {"id": key, "question": question, "action": action}
            for scenarios, action in ((_DRIFT, drift_action), (_SCORING, "blind_scorer"), (_OTHER, None))
            for key, question in scenarios
        ],
        "checks": [
            {
                "id": case["id"],
                "question": case["question"],
                "action": case[expected_field],
                "retain": corrected and case["id"] == "retention",
            }
            for case in CHECKS
        ],
    }
