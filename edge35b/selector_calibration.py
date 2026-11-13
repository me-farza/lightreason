"""Fit a small DFlash2 score adjustment from verified teacher prefixes."""

import json
import sys


TRAINING_PROMPTS = (
    "Solve for x: 4(x - 3) + 2x = 30. Show the steps, then give x.",
    "Write a Python function that tests whether a string is a palindrome after ignoring case and whitespace. Include two examples.",
    "Write a friendly reply accepting a coworker's invitation to lunch on Wednesday. Mention a vegetarian option.",
    "Build a compact HTML page for a neighborhood bookstore with three featured books, a search field, and responsive cards.",
    "A tank fills at 9 liters per minute and drains at 4. It starts with 18 liters. How long until it has 78 liters?",
    "Implement a Python function that merges two sorted lists without mutating either. State the complexity.",
    "Draft a short apology for missing a project meeting and propose a concrete way to catch up tomorrow.",
    "Build a simple one-page pottery workshop signup with two session cards and an accessible booking form.",
    "Explain why the square of an odd integer is odd. Give an algebraic proof and one example.",
    "Write a JavaScript function that groups an array of objects by a string-valued key. Show the result for a three-item example.",
    "Politely decline a neighbor's invitation to a weekend barbecue and suggest a different day.",
    "Create a responsive HTML landing page for a bicycle repair shop. Include a price list and contact details.",
    "A recipe for six people uses 420 grams of flour. How much flour is needed for nine people? Explain briefly.",
    "Write Python code to return the most frequent integer in a nonempty list, breaking ties in favor of the smaller integer.",
    "Draft a two-sentence thank-you note for someone who helped you move. Offer to help with their next project.",
    "Create one HTML file for a museum exhibit page with an image placeholder, opening times, and a responsive layout.",
)
VALIDATION_COUNT = 4
TOKEN_LIMIT = 96


def prompts():
    return {f"selector-dev-{i:02d}": {"prompt": prompt, "system_prompt": None}
            for i, prompt in enumerate(TRAINING_PROMPTS)}


def examples(records):
    import torch

    unary, transition, position, labels, chosen = [], [], [], [], []
    for record in records:
        for p in record["passes"]:
            for j in range(min(p["committed"], p["accepted_drafts"] + 1, len(p["candidate_ids"]))):
                candidates = p["candidate_ids"][j]
                token = p["target_ids"][j]
                if token not in candidates:
                    continue
                unary.append(p["scores"][0][j])
                transition.append(p["scores"][1][j])
                position.append(j)
                labels.append(candidates.index(token))
                chosen.append(candidates.index(p["draft_ids"][j]))
    if not labels:
        raise ValueError("No verified candidate labels")
    u, t = torch.tensor(unary), torch.tensor(transition)
    pos, y, old = map(torch.tensor, (position, labels, chosen))
    if not u.isfinite().all() or not t.isfinite().all():
        raise ValueError("Nonfinite score data")
    u, t = u - u.amax(1, keepdim=True), t - t.amax(1, keepdim=True)
    return u, t, pos, y, old


def fit(records):
    import torch
    import torch.nn.functional as F

    if len(records) != len(TRAINING_PROMPTS):
        raise ValueError("Unexpected calibration corpus")
    torch.set_num_threads(1)
    train = examples(records[:-VALIDATION_COUNT])
    valid = examples(records[-VALIDATION_COUNT:])
    positions, k = len(records[0]["passes"][0]["candidate_ids"]), train[0].shape[1]

    def score(samples, scale, bias):
        u, t, pos, y, old = samples
        logits = u * scale[pos, None] + t + bias[pos]
        pred = logits.argmax(1)
        return {
            "rows": y.numel(),
            "loss": float(F.cross_entropy(logits, y)),
            "accuracy": float((pred == y).float().mean()),
            "original_accuracy": float((old == y).float().mean()),
            "changed_correct": int(((old == y) & (pred != y)).sum()),
            "fixed_incorrect": int(((old != y) & (pred == y)).sum()),
            "agreement_with_native": float((old == pred).float().mean()),
        }

    results = {}
    for mode in ("identity", "scale", "scale-and-rank"):
        alpha = torch.zeros(positions, requires_grad=mode != "identity")
        bias = torch.zeros(positions, k, requires_grad=mode == "scale-and-rank")
        params = [p for p in (alpha, bias) if p.requires_grad]
        if params:
            optimizer = torch.optim.LBFGS(params, max_iter=45, line_search_fn="strong_wolfe")
            u, t, pos, y, _ = train

            def closure():
                optimizer.zero_grad()
                a = alpha.clamp(-0.69, 0.69).exp()
                b = bias.clamp(-2, 2)
                loss = F.cross_entropy(u * a[pos, None] + t + b[pos], y)
                loss = loss + 0.03 * alpha.square().mean() + 0.03 * bias.square().mean()
                loss.backward()
                return loss

            optimizer.step(closure)
        with torch.no_grad():
            scale = alpha.clamp(-0.69, 0.69).exp()
            ranks = bias.clamp(-2, 2)
            results[mode] = {"scale": scale.tolist(), "bias": ranks.tolist(),
                             "train": score(train, scale, ranks), "validation": score(valid, scale, ranks)}
    if results["identity"]["validation"]["agreement_with_native"] < 0.995:
        raise ValueError("Recorded teacher-prefix scores do not reproduce the native selector")
    winner = min(("scale", "scale-and-rank"), key=lambda mode: (
        -results[mode]["validation"]["accuracy"], results[mode]["validation"]["loss"]
    ))
    selected = {name: results[winner][name] for name in ("scale", "bias")}
    return {"selection": winner, "selected": selected, "models": results,
            "training_prompts": len(TRAINING_PROMPTS) - VALIDATION_COUNT,
            "validation_prompts": VALIDATION_COUNT}


if __name__ == "__main__":
    json.dump(fit(json.load(sys.stdin)), sys.stdout)
