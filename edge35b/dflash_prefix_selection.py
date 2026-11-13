"""Score DFlash2 candidate paths by expected verified prefix length."""


STEPS = 7
TOP_K = 16


def select_prefix(ids, unary, state, anchor, predecessors, successors):
    """Select a greedy request's proposal and report conditional path scores.

    The transition model is the existing DFlash2 selector. A backward pass
    maximizes the expected number of accepted tokens under its normalized
    scores. A low-scoring early token discounts all its later rewards.
    """
    import torch

    if ids.shape != (1, STEPS, TOP_K) or unary.shape != ids.shape or anchor.shape != (1,):
        raise ValueError("Expected one seven-proposal DFlash2 candidate lattice")
    previous = torch.cat((anchor.expand(TOP_K).reshape(1, TOP_K), ids[0, :-1]), dim=0)
    query = predecessors[previous].float() * state.reshape(STEPS, 1, -1).float()
    keys = successors[ids[0]].float()
    scores = torch.bmm(query, keys.transpose(1, 2)) + unary[0, :, None, :]
    probabilities = scores.softmax(dim=-1)
    value = torch.zeros_like(probabilities[0, 0])
    backtrack = []
    for slot in range(STEPS - 1, -1, -1):
        value, choice = (probabilities[slot] * (1 + value)).max(dim=-1)
        backtrack.append(choice)
    path, confidence = [], []
    prev = torch.zeros(1, dtype=torch.int64, device=ids.device)
    for slot, choice in enumerate(reversed(backtrack)):
        nxt = choice.gather(0, prev)
        confidence.append(probabilities[slot].index_select(0, prev)[0].gather(0, nxt))
        path.append(ids[0, slot].gather(0, nxt))
        prev = nxt
    return torch.cat(path).view(1, STEPS), torch.cat(confidence).view(1, STEPS)
