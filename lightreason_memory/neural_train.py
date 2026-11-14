"""Train the neural-memory writer and reader on synthetic episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
from tempfile import NamedTemporaryFile
from time import perf_counter


TRAIN_KEYS = (
    "cabinet", "locker", "chest", "portal", "wallet", "safe", "archive", "cupboard",
    "notebook", "toolbox", "shed", "attic", "garage", "gate", "studio", "closet",
    "valise", "suitcase", "bookcase", "briefcase", "drawer", "trunk", "luggage",
    "shelter", "crate", "basket", "tent", "vest", "jacket", "parcel", "case",
    "ledger", "bookplate", "keepsake", "mosaic", "sculpture", "calendar", "podium",
    "banner", "pouch", "vase", "mask", "pendant", "ring", "locket", "journal",
    "leather pouch", "green folder", "garden shed", "metal locker", "desk drawer",
    "wooden trunk", "ceramic bowl", "camping tent", "phone case", "field journal",
)
DEV_KEYS = ("trophy", "canteen", "harp", "violin", "helmet", "mirror", "crystal sphere", "travel pack")
TRAIN_VALUES = (
    "silver", "lemon", "mint", "brick", "cream", "salmon", "moss", "steel", "marble",
    "spruce", "pine", "cherry", "granite", "slate", "buttercup", "indigo", "crimson",
    "azure", "plum", "rose", "teal", "aqua", "olive", "emerald", "silverfish", "copper",
    "sandstone", "bamboo", "silk", "wool", "paper", "clay", "ecru", "bronze", "cinnamon",
    "cardamom", "sage", "river", "cloud", "quartz", "mustard", "charcoal", "orchid",
    "marigold", "rainbow", "tangerine", "lavender", "daffodil", "honey", "yarrow",
    "almond", "ginger", "vanilla", "sand", "hickory", "maple", "waterlily", "flax",
)
DEV_VALUES = ("fig", "anise", "cocoa", "oat", "sunbeam", "sunflower", "dandelion", "aspen")
ACCEPT_VALUES = (
    "coral", "amber", "saffron", "zircon", "hazel", "cedar", "opal", "mango", "velvet",
    "basalt", "walnut", "onyx", "jasper", "ochre", "cobalt", "ivy", "satin", "iris",
    "cotton", "jade", "rattan", "pewter", "rust", "birch", "violet", "ivory", "lilac",
    "mahogany", "pearl", "linen",
)
WRITE = (
    "I found that the {field} for the {key} is {value}.",
    "I set the {field} for the {key} to {value}.",
    "My {field} for the {key} is now {value}.",
    "The {key}'s {field} is {value}.",
    "Just a note: I picked {value} as the {field} for the {key}.",
    "Please remember that {value} is the {field} for the {key}.",
    "When I need to open the {key}, I enter {value}.",
    "Today I recorded {value} as the {key}'s {field}.",
    "During setup, I entered {value} as the {field} for the {key}.",
    "Here is an update: the {field} for the {key} has changed to {value}.",
    "I changed the {field} for the {key} to {value}.",
    "The new {field} for the {key} is {value}.",
    "I am using {value} for the {key}'s {field} now.",
    "I put {value} on file for the {key}'s {field}.",
    "I made the {field} of the {key} {value}.",
    "I have switched the {field} for the {key} over to {value}.",
    "The {field} that I use to unlock the {key} is {value}.",
    "For the {key}, put down {value} as the {field}.",
    "The {key} opens with {value}.",
    "The {key} can be unlocked by entering {value}.",
    "On the {key}, the {field} is written as {value}.",
    "I saved the {field} {value} for the {key}.",
    "Use {value} when opening the {key}.",
    "The current {field} for the {key} should be {value}.",
    "I have the {field} for the {key} written as {value}.",
)
READ = (
    "What is the {field} for the {key}?",
    "Which {field} should I use for the {key}?",
    "Can you remind me of the {key}'s {field}?",
    "What {field} do I need to open the {key}?",
    "What passphrase opens the {key}?",
    "Could you tell me how to unlock the {key}?",
    "Which {field} does the {key} require now?",
    "What was the {field} that I picked for the {key}?",
    "What do I type to open the {key}?",
    "What {field} should I enter to unlock the {key}?",
    "Tell me the {field} I used for the {key}.",
    "Could you recall the {key}'s new {field}?",
    "What did I name the {key}'s {field}?",
    "What do I need to unlock the {key}?",
    "How can I open the {key} with its {field}?",
    "Can you recall the {field} that I chose for the {key}?",
    "Which words will open the {key}?",
    "Which phrase should I use to unlock the {key}?",
    "Which {field} do I use to get into the {key}?",
    "What do I enter for the {key}?",
    "Do you remember what opens the {key}?",
)
DEV_WRITE = (
    "I learned the {field} for the {key} is {value}.",
    "For future use, the {key}'s {field} will be {value}.",
    "I updated the {key}'s {field} to {value}.",
    "In the morning, I wrote down {value} for opening the {key}.",
    "The {field} I typed into the {key} this time was {value}.",
    "I told a friend to enter {value} to open the {key}.",
    "My most recent {field} for the {key} became {value}.",
    "After a quick check, I recorded the {field} on the {key} as {value}.",
)
DEV_READ = (
    "Remind me which {field} is used for the {key}.",
    "Do you know the current {field} for the {key}?",
    "I forget what I type when I open the {key}.",
    "Which {field} did I write down for my {key}?",
    "Can you remember how I get into the {key}?",
    "Which words let me unlock my {key} again?",
    "Do you know the {key}'s current {field}?",
)
FIELDS = ("access phrase", "entry code", "passphrase", "unlock phrase")
WRITE_CONTEXT = (
    "For my records, ", "Quick update: ", "Last night, ", "This afternoon, ", "While setting up, ",
    "At the end of the session, ", "After checking twice, ", "I should remember this. ",
    "Before I sign off, ", "This is just a reminder. ", "When I got home, ", "While I was testing it, ",
)
READ_CONTEXT = (
    "One more thing: ", "I need a reminder. ", "Just checking: ", "Before I start, ",
    "I wrote something down yesterday. ", "I don't have my notes. ",
)
WRITE_NEGATIVES = (
    "We had a nice walk this morning.",
    "Can you help me organize my notes?",
    "The forecast predicts snow tomorrow.",
    "My notebook is already open.",
)
READ_NEGATIVES = (
    "What is the weather forecast for next week?",
    "Do you know anything about my dentist?",
    "Can you explain how gravity works?",
    "What was I doing last Thursday?",
)
WRITE_OTHER_ATTRIBUTE = (
    "I bought the {key} last month.",
    "The {key} is a few years old.",
    "I keep the {key} on the shelf.",
    "I painted the {key} blue yesterday.",
    "I washed the {key} after we got back.",
)
READ_OTHER_ATTRIBUTE = (
    "Where did I buy the {key}?",
    "What year did I get the {key}?",
    "How much does the {key} weigh?",
    "Did I record the size of the {key}?",
    "How should I clean the {key}?",
    "Which shade did I paint my {key}?",
    "How old is the {key}?",
)


def tokenize_spans(tokenizer, text, key=None, value=None):
    """Label key and value tokens using the exact source offset of each entity."""
    spans = []
    for label, part in ((1, key), (2, value)):
        if part is not None:
            matches = list(re.finditer(rf"(?<!\w){re.escape(part)}(?!\w)", text, re.IGNORECASE))
            if len(matches) != 1:
                raise ValueError("Synthetic spans must occur once as complete words.")
            start = matches[0].start()
            spans.append((start, start + len(part), label))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    labels = [
        next((label for first, last, label in spans if start < last and stop > first), 0)
        for start, stop in encoded["offset_mapping"]
    ]
    if any(label not in labels for _, _, label in spans):
        raise ValueError("Synthetic spans must contain a token.")
    return encoded["input_ids"], labels


def _episodes(tokenizer, seed, count, *, dev=False):
    rng = random.Random(seed)
    entities, vocabulary = (DEV_KEYS, DEV_VALUES) if dev else (TRAIN_KEYS, TRAIN_VALUES)
    write_templates, read_templates = (DEV_WRITE, DEV_READ) if dev else (WRITE, READ)
    episodes = []
    for _ in range(count):
        entity = rng.randrange(len(entities))
        key = entities[entity]
        field = rng.choice(FIELDS)
        value_length = rng.choice((1, 2, 2, 2, 3))
        value = " ".join(rng.sample(vocabulary, value_length))
        if rng.random() < 0.09:
            value += f" {rng.randrange(100, 1000)}"
        if not dev and rng.random() < 0.20:
            value = value.replace(" ", rng.choice(("-", "_")))
        experience = rng.choice(write_templates).format(key=key, value=value, field=field)
        question = rng.choice(read_templates).format(key=key, field=field)
        if not dev:
            if rng.random() < 0.58:
                experience = rng.choice(WRITE_CONTEXT) + experience[0].lower() + experience[1:]
            if rng.random() < 0.37:
                question = rng.choice(READ_CONTEXT) + question[0].lower() + question[1:]
        negative_write = rng.random() < 0.12
        negative_read = rng.random() < 0.17
        if negative_write:
            experience = rng.choice(WRITE_NEGATIVES + WRITE_OTHER_ATTRIBUTE).format(key=key)
        if negative_read:
            question = rng.choice(READ_NEGATIVES + READ_OTHER_ATTRIBUTE).format(key=key)
        episodes.append((
            tokenize_spans(tokenizer, experience, None if negative_write else key, None if negative_write else value),
            tokenize_spans(tokenizer, question, None if negative_read else key),
            entity,
        ))
    return episodes


def _pad(torch, examples, pad_id, device):
    width = max(len(ids) for ids, _ in examples)
    return tuple(
        torch.tensor(rows, device=device, dtype=torch.long) for rows in (
            [ids + [pad_id] * (width - len(ids)) for ids, _ in examples],
            [labels + [-100] * (width - len(labels)) for _, labels in examples],
        )
    )


def _key_loss(model, write_ids, write_labels, read_ids, read_labels, entities):
    import torch
    from torch.nn.functional import cross_entropy

    write_mask, read_mask = write_labels == 1, read_labels == 1
    ok = write_mask.any(1) & read_mask.any(1)
    if not ok.any():
        return model.key(write_ids, write_mask).sum() * 0
    writes = model.key(write_ids[ok], write_mask[ok])
    reads = model.key(read_ids[ok], read_mask[ok])
    similarity = reads @ writes.T
    same_key = entities[ok, None] == entities[ok][None, :]
    # Repeated examples of the same entity are equivalent positives.
    scores = (similarity * 16).masked_fill(same_key & ~torch.eye(len(similarity), device=similarity.device, dtype=torch.bool), -1e4)
    identity = torch.arange(len(similarity), device=similarity.device)
    return (cross_entropy(scores, identity) + cross_entropy(scores.T, identity)) / 2


def _value_ids(torch, size, exclusions, device):
    allowed = torch.ones(size, device=device, dtype=torch.bool)
    reserved = [token for token in exclusions if 0 <= token < size]
    allowed[reserved] = False
    return allowed.nonzero().flatten()


def _codec_loss(model, vocab, write_ids, write_labels, negatives, positives, random):
    import torch
    from torch.nn.functional import cross_entropy

    sampled = vocab[torch.randint(len(vocab), (positives,), device=vocab.device, generator=random)]
    tokens = torch.cat((write_ids[write_labels == 2], sampled))
    distractors = vocab[torch.randint(len(vocab), (negatives,), device=vocab.device, generator=random)]
    candidates, labels = torch.unique(torch.cat((tokens, distractors)), return_inverse=True)
    logits = model.decode(model.encode(tokens[None])[0], candidates)
    return cross_entropy(logits.float(), labels[:len(tokens)])


def _span_exact(torch, predicted, truth):
    return ((predicted == truth) | (truth == -100)).all(1).sum().item()


def _tag_logits(model, ids, mode, features=None):
    if features is None:
        return model.tag(ids, mode)
    return model.tag_features(features.to(ids.device), ids, mode)


def _confident_tags(torch, model, ids, mode, features=None):
    confidence, tags = _tag_logits(model, ids, mode, features).softmax(-1).max(-1)
    return torch.where(confidence >= 0.65, tags, 0)


def _evaluate(model, data, batch_size, feature_cache=None, *, evaluate_tagger=True):
    import torch

    totals = dict(oracle_addressing_exact=0, oracle_addressing_total=0,
                  oracle_value_token_exact=0, oracle_value_tokens=0)
    if evaluate_tagger:
        totals.update(write_exact=0, read_exact=0)
    with torch.no_grad():
        for start in range(0, len(data[0]), batch_size):
            w, wl, q, ql, entity = (array[start:start + batch_size] for array in data)
            if evaluate_tagger:
                wf, qf = (None, None) if feature_cache is None else (
                    array[start:start + batch_size] for array in feature_cache
                )
                totals["write_exact"] += _span_exact(torch, _confident_tags(torch, model, w, "write", wf), wl)
                totals["read_exact"] += _span_exact(torch, _confident_tags(torch, model, q, "read", qf), ql)
            wm, qm = wl == 1, ql == 1
            keep_w, keep_q = wm.any(1), qm.any(1)
            present = torch.isin(entity[keep_q], entity[keep_w])
            if keep_w.any() and present.any():
                keys = model.key(w[keep_w], wm[keep_w])
                queries = model.key(q[keep_q][present], qm[keep_q][present])
                winners = (queries @ keys.T).argmax(-1)
                totals["oracle_addressing_exact"] += (entity[keep_w][winners] == entity[keep_q][present]).sum().item()
                totals["oracle_addressing_total"] += present.sum().item()
            tokens = w[wl == 2]
            for chunk in tokens.split(32):
                if chunk.numel():
                    predictions = model.decode(model.encode(chunk[None])[0].half()).argmax(-1)
                    totals["oracle_value_token_exact"] += (predictions == chunk).sum().item()
                    totals["oracle_value_tokens"] += chunk.numel()
    return {"episodes": len(data[0]), **totals}


def _dataset(torch, tokenizer, seed, count, device, *, dev=False):
    episodes = _episodes(tokenizer, seed, count, dev=dev)
    writes, reads, entities = zip(*episodes)
    return (
        *_pad(torch, writes, tokenizer.pad_token_id, device),
        *_pad(torch, reads, tokenizer.pad_token_id, device),
        torch.tensor(entities, device=device),
    )


def _precompute_features(torch, model, data, batch_size, label):
    caches = []
    with torch.no_grad():
        for mode, ids in (("write", data[0]), ("read", data[2])):
            started = perf_counter()
            output = None
            for start in range(0, len(ids), batch_size):
                values = model.features(ids[start:start + batch_size]).to(dtype=torch.bfloat16)
                if output is None:
                    output = torch.empty((len(ids), *values.shape[1:]), dtype=torch.bfloat16, device="cpu")
                output[start:start + len(values)].copy_(values.cpu())
                if start == 0 or (start // batch_size + 1) % 50 == 0:
                    print(json.dumps({"precompute": label, "mode": mode, "episodes": start + len(values),
                                      "total": len(ids), "elapsed_s": perf_counter() - started}), flush=True)
            caches.append(output)
    return tuple(caches)


def _mask_tokens(torch, ids, labels, pad_id, generator, rate=0.12):
    # Replace some high-frequency token identities during tagger training. This
    # makes boundaries learnable from surrounding grammar for unseen values.
    eligible = (labels > 0) & ids.ne(pad_id)
    hide = torch.rand(ids.shape, device=ids.device, generator=generator) < rate
    return ids.masked_fill(eligible & hide, 0)


def _write_report(path, report):
    if path is None:
        return
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--train-episodes", type=int, default=6000)
    spans = parser.add_mutually_exclusive_group()
    spans.add_argument("--contextual", action="store_true", help="train a tagger over frozen Qwen contextual features")
    spans.add_argument("--prompted-spans", action="store_true",
                       help="use frozen Qwen span extraction at runtime and train only the address and codec")
    parser.add_argument("--context-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--negatives", type=int, default=512)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.train_episodes, args.context_batch_size,
           args.eval_every, args.negatives) < 1:
        parser.error("training, batch and evaluation sizes must be positive")
    if args.checkpoint.exists() or (args.report is not None and args.report.exists()):
        parser.error("choose a new checkpoint and report path")

    import torch
    from .backend import _gpu_lease

    device = torch.device(args.device)
    lease = _gpu_lease() if device.type == "cuda" else None
    try:
        train(args, device)
    finally:
        if lease is not None:
            os.close(lease)


def train(args, device):
    import torch
    from torch.nn.functional import cross_entropy
    from .neural import NeuralModel, load_qwen, save_checkpoint

    start = perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    tokenizer, embedding, head = load_qwen(args.model_path, device)
    context = None
    if args.contextual:
        from .neural import load_qwen_context

        context = load_qwen_context(args.model_path, device)
    model = (
        NeuralModel(embedding, head, context=context) if args.contextual else NeuralModel(embedding, head)
    ).to(device)
    codec_parameters = [*model.value_in.parameters(), *model.value_out.parameters(), model.scale]
    address_parameters = list(model.address.parameters())
    specialized = {id(param) for param in codec_parameters + address_parameters}
    optimizer_groups = [
        {"params": address_parameters, "lr": 0.0005},
        {"params": codec_parameters, "lr": 0.0001},
    ]
    if not args.prompted_spans:
        optimizer_groups.append({
            "params": [param for param in model.parameters() if param.requires_grad and id(param) not in specialized],
            "lr": 0.002,
        })
    trainable = [param for group in optimizer_groups for param in group["params"]]
    optimizer = torch.optim.AdamW(optimizer_groups)
    train = _dataset(torch, tokenizer, args.seed, args.train_episodes, device)
    dev = _dataset(torch, tokenizer, 41041, 72, device, dev=True)
    train_features = dev_features = None
    if args.contextual:
        model.eval()
        train_features = _precompute_features(torch, model, train, args.context_batch_size, "train")
        dev_features = _precompute_features(torch, model, dev, args.context_batch_size, "development")
    excluded = set(tokenizer.all_special_ids)
    for word in ACCEPT_VALUES + DEV_VALUES:
        excluded.update(tokenizer.encode(word, add_special_tokens=False))
        excluded.update(tokenizer.encode(" " + word, add_special_tokens=False))
    vocab = _value_ids(torch, head.shape[0], excluded, device)
    report = {
        "seed": args.seed,
        "development_seed": 41041,
        "trainable_parameters": sum(param.numel() for param in trainable),
        "training_episodes": len(train[0]),
        "device": str(device),
        "contextual": args.contextual,
        "prompted_spans": args.prompted_spans,
        "preparation_s": perf_counter() - start,
        "steps": args.steps,
        "stages": [],
    }
    best = None
    weights = torch.tensor([1., 3., 2.], device=device)
    model.train()
    for step in range(1, args.steps + 1):
        sample = torch.randint(len(train[0]), (args.batch_size,), device=device, generator=random_generator)
        w, wl, q, ql, entities = (array[sample] for array in train)
        key = _key_loss(model, w, wl, q, ql, entities)
        codec = _codec_loss(model, vocab, w, wl, args.negatives, 96, random_generator)
        loss = 0.3 * key + codec
        tag_losses = {}
        if not args.prompted_spans:
            if train_features is None:
                writer_ids = _mask_tokens(torch, w, wl, tokenizer.pad_token_id, random_generator)
                reader_ids = _mask_tokens(torch, q, ql, tokenizer.pad_token_id, random_generator)
                wf = qf = None
            else:
                writer_ids, reader_ids = w, q
                cpu_sample = sample.cpu()
                wf, qf = (array[cpu_sample] for array in train_features)
            writer = cross_entropy(_tag_logits(model, writer_ids, "write", wf).transpose(1, 2).float(), wl, weight=weights)
            reader = cross_entropy(_tag_logits(model, reader_ids, "read", qf).transpose(1, 2).float(), ql, weight=weights)
            tag_losses = {"writer": writer, "reader": reader}
            loss = loss + writer + reader
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 2.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            evaluation = _evaluate(model, dev, args.batch_size, dev_features,
                                   evaluate_tagger=not args.prompted_spans)
            model.train()
            entry = {"step": step, "elapsed_s": perf_counter() - start, "development": evaluation,
                     "loss": {name: value.item() for name, value in {"key": key, "codec": codec, **tag_losses}.items()}}
            report["stages"].append(entry)
            score = (evaluation.get("write_exact", 0) + evaluation.get("read_exact", 0),
                     evaluation["oracle_value_token_exact"], evaluation["oracle_addressing_exact"])
            if best is None or score >= best:
                best = score
                report["checkpoint_step"] = step
                save_checkpoint(model, args.checkpoint, prompted_spans=args.prompted_spans)
            _write_report(args.report, report)
            print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
