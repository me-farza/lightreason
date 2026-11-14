"""Train and run a reward-trained memory policy beside frozen local Qwen."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

import torch
from torch.nn.functional import normalize
from safetensors import safe_open
from safetensors.torch import load_file

from .backend import MODEL_VERSION, _exit_with_parent, _gpu_lease
from .dream_state import AlignedValues, load_state
from .dream_train import DreamExample
from .neural import (
    MAX_TEXT, MAX_TOKENS, MAX_VALUE_TOKENS, PAD, VALUE_DIM, NeuralModel, _atomic_write, _require_private_directory,
    load_qwen, load_qwen_context,
)
from .neural_demo import _write_report
from .rl_episodes import VALUES, make_episodes
from .rl_policy import (
    CAPACITY, QUERY_KINDS, Bank, MemoryPolicy, make_rollout, read_action, replace_slots, write_action, write_slots,
)
from .rl_train import evaluate, train


FORMAT = "lightreason-rl-memory-0.1"
FEATURE_WIDTH = 768


class FrozenQwenFeatures:
    def __init__(self, model_path, codec_path, device):
        tokenizer, table, head = load_qwen(model_path, device)
        self.context = load_qwen_context(model_path, device)
        self.tokenizer = tokenizer
        self.device = device
        self.codec = NeuralModel(table, head).to(device)
        self.codec.load_state_dict(load_file(codec_path, device="cpu"), strict=True)
        self.codec.eval().requires_grad_(False)
        seed = torch.Generator().manual_seed(62812)
        half = FEATURE_WIDTH // 2
        self.projection = (torch.randn(table.shape[-1], half, generator=seed) / table.shape[-1]**0.5).to(device)

    @torch.no_grad()
    def rows(self, ids, *, encode_values=True):
        mask = ids.ne(PAD)
        lex = normalize(self.codec.embedding[ids].float(), dim=-1)
        output = self.context.model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
        middle = normalize(output.hidden_states[16].float(), dim=-1)
        final = normalize(output.last_hidden_state.float(), dim=-1)
        feature = torch.cat((normalize(lex @ self.projection, dim=-1),
                             normalize((middle + final) @ self.projection, dim=-1)), dim=-1)
        keys = normalize(self.codec.address(lex), dim=-1)
        values = self.codec.encode(ids).to(torch.float16) if encode_values else None
        return feature.to(torch.float16), keys.to(torch.float16), values

    def ids(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT or "\0" in text:
            raise ValueError(f"Supply 1–{MAX_TEXT} characters.")
        row = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = row["input_ids"]
        if not 1 <= len(ids) <= MAX_TOKENS or set(ids).intersection(self.tokenizer.all_special_ids):
            raise ValueError("Text exceeds the token limit or contains control tokens.")
        return row

    def request(self, text, *, encode_values=True):
        ids = torch.tensor([self.ids(text)["input_ids"]], device=self.device)
        features, keys, values = self.rows(ids, encode_values=encode_values)
        return features, ids.ne(PAD), keys.float(), values.float() if values is not None else None

    def decode(self, value):
        ids = self.codec.decode(value).argmax(-1).tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def _position(encoded, offset):
    if offset is None:
        return 0
    matches = [i for i, (left, right) in enumerate(encoded["offset_mapping"]) if left <= offset < right]
    if len(matches) != 1:
        raise ValueError("The fixture does not have an unambiguous token span.")
    return matches[0]


def encode_episodes(encoder, collections):
    """Cache features of each utterance. Full raw episodes never enter the policy."""
    texts = {
        line["text"]: None for episodes in collections for episode in episodes
        for group in ("observations", "queries") for line in episode[group]
    }
    tokenized = [encoder.ids(text) for text in texts]
    ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(row["input_ids"]) for row in tokenized], batch_first=True, padding_value=PAD,
    )
    batches = []
    for part in ids.split(64):
        batches.append(tuple(x.cpu() for x in encoder.rows(part.to(encoder.device))))
    features, keys, values = (torch.cat(items) for items in zip(*batches, strict=True))
    lookup = {text: index for index, text in enumerate(texts)}

    def encoded(episodes):
        observations = [episode["observations"] for episode in episodes]
        queries = [episode["queries"] for episode in episodes]
        w = torch.tensor([[lookup[item["text"]] for item in episode] for episode in observations])
        q = torch.tensor([[lookup[item["text"]] for item in episode] for episode in queries])
        integers = {
            "write_admit": [[int(not item["is_distractor"]) for item in episode] for episode in observations],
            "write_transient": [[bool(item.get("is_transient")) for item in episode] for episode in observations],
            "write_spans": [[[
                _position(tokenized[lookup[item["text"]]], item["key_start"]),
                _position(tokenized[lookup[item["text"]]], item["value_start"]),
            ] for item in episode] for episode in observations],
            "query_spans": [[_position(tokenized[lookup[item["text"]]], item["key_start"])
                            for item in episode] for episode in queries],
            "after_write": [[item["after_write"] for item in episode] for episode in queries],
            "targets": [[item["target"] for item in episode] for episode in queries],
            "query_kind": [[QUERY_KINDS.index(item["kind"]) for item in episode] for episode in queries],
        }
        # Verify the one-token lexical reward assumption on actual tokenized sentences.
        for row, labels in zip(observations, integers["write_spans"], strict=True):
            for item, (_, value_position) in zip(row, labels, strict=True):
                if item["value"] is not None:
                    token = tokenized[lookup[item["text"]]]["input_ids"][value_position]
                    if encoder.tokenizer.decode([token]).strip() != item["value"]:
                        raise ValueError(f"The fixture value {item['value']!r} must occupy one Qwen token.")
        return {
            "writes": features[w], "write_keys": keys[w], "write_values": values[w],
            "write_mask": ids[w].ne(PAD), "queries": features[q], "query_keys": keys[q],
            "query_mask": ids[q].ne(PAD), **{key: torch.tensor(value) for key, value in integers.items()},
        }
    return [encoded(episodes) for episodes in collections]


def _device_rollout(rollout, device):
    def run(policy, batch, **options):
        return rollout(policy, {name: value.to(device) for name, value in batch.items()}, **options)
    return run


def _catalog(encoder):
    ids = [encoder.tokenizer.encode(" " + word, add_special_tokens=False) for word in VALUES]
    if any(len(row) != 1 for row in ids):
        raise ValueError("The synthetic reward catalog requires one token per value.")
    return encoder.codec.encode(torch.tensor(ids, device=encoder.device).flatten()).half().float()


def _digest(path):
    with Path(path).open("rb") as stream:
        from hashlib import file_digest

        return file_digest(stream, "sha256").hexdigest()


def fit(args, device):
    torch.manual_seed(args.seed)
    start = perf_counter()
    encoder = FrozenQwenFeatures(args.model, args.codec, device)
    collections = [make_episodes(args.seed, number, split, transient=True) for number, split in (
        (args.train_episodes, "train"), (args.dev_episodes, "validation"), (args.test_episodes, "test"),
    )]
    train_data, dev, test = encode_episodes(encoder, collections)
    catalog = _catalog(encoder)
    rollout = _device_rollout(make_rollout(catalog.float()), device)
    policy = MemoryPolicy(FEATURE_WIDTH).to(device)
    prepared = perf_counter() - start
    snapshots, report = train(
        policy, train_data, dev, rollout, warmstart_steps=args.warmstart_steps, rl_steps=args.rl_steps,
        batch_size=args.batch_size, seed=args.seed, matched_teacher=True, timeout_s=args.timeout,
        aligned_teacher=args.aligned_teacher,
        report_every=50, progress=lambda row: print(json.dumps(row, separators=(",", ":")), flush=True),
    )
    report.update(feature_preparation_s=prepared, held_out_test={}, test_episodes=args.test_episodes)
    for name, state in snapshots.items():
        policy.load_state_dict(state)
        report["held_out_test"][name] = evaluate(policy, test, rollout, batch_size=args.batch_size)
        _atomic_write(state, args.out / f"{name}.safetensors", {
            "version": FORMAT, "backbone": MODEL_VERSION, "codec_sha256": _digest(args.codec), "training_stage": name,
        })
    policy.load_state_dict(snapshots["rl"])
    for control in ("empty", "shuffled_queries"):
        report["held_out_test"][control] = evaluate(
            policy, test, _device_rollout(make_rollout(catalog, control=control), device), batch_size=args.batch_size,
        )
    report["wall_s"] = perf_counter() - start
    _write_report(args.out / "training.json", report)


def _verified_positions(encoder, text, key, value):
    encoded = encoder.ids(text)
    positions = []
    for label in (key, value):
        if not isinstance(label, str) or not label or label.strip() != label or len(label) > MAX_TEXT:
            raise ValueError("Each verified label must be a single token in the observation.")
        spans = [match.span() for match in re.finditer(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text)]
        if len(spans) != 1:
            raise ValueError("Each verified label must occur exactly once in the observation.")
        left, right = spans[0]
        position = _position(encoded, left)
        start, end = encoded["offset_mapping"][position]
        token = encoded["input_ids"][position]
        if not (start <= left and right == end and encoder.tokenizer.decode([token]).strip() == label):
            raise ValueError("Each verified label must be a single token in the observation.")
        positions.append(position)
    if positions[0] == positions[1]:
        raise ValueError("The verified key and value must be distinct tokens.")
    return positions


class PolicyMemory:
    """Perform learned policy actions against one private tensor state."""

    def __init__(self, model_path, codec_path, policy_path, state_path, device, *,
                 dream_every=0, dream_steps=48, destination_model_path=None):
        if type(dream_every) is not int or dream_every < 0:
            raise ValueError("Dream cadence must be a nonnegative message count.")
        if type(dream_steps) is not int or not 1 <= dream_steps <= 128:
            raise ValueError("Use between 1 and 128 dream steps.")
        self.device = torch.device(device)
        self.state = Path(state_path)
        self.dream_every = dream_every
        self.dream_steps = dream_steps
        self.model_path = model_path
        self.codec_path = codec_path
        self._encoder = None
        self.destination_model_path = destination_model_path
        self._destination = None
        self.target_binding = None
        if destination_model_path is not None:
            from edge35b.download import REVISION

            self.target_binding = torch.tensor(list(sha256(REVISION.encode("ascii")).digest()), dtype=torch.uint8)
        with safe_open(policy_path, framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
        if (metadata.get("version"), metadata.get("backbone"), metadata.get("codec_sha256")) != (
            FORMAT, MODEL_VERSION, _digest(codec_path),
        ):
            raise ValueError("Policy and codec checkpoint identity mismatch.")
        self.policy = MemoryPolicy(FEATURE_WIDTH).to(self.device)
        self._base_state = load_file(policy_path, device="cpu")
        self.policy.load_state_dict(self._base_state, strict=True)
        if not all(torch.isfinite(parameter).all() for parameter in self._base_state.values()):
            raise ValueError("Policy checkpoint contains invalid parameters.")
        self.policy.eval()
        self.binding = torch.tensor(list(bytes.fromhex(_digest(policy_path))), dtype=torch.uint8)

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = FrozenQwenFeatures(self.model_path, self.codec_path, self.device)
        return self._encoder

    def _lock(self, write):
        import fcntl
        from .neural import _private_open

        self.state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_directory(self.state.parent)
        fd = _private_open(self.state.with_suffix(self.state.suffix + ".lock"), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _load(self):
        bank, dream = load_state(
            self.state, self.binding, self.device, feature_width=FEATURE_WIDTH, policy_template=self._base_state,
        )
        if self.target_binding is not None:
            if dream.aligned is None:
                dream.aligned = AlignedValues(self.target_binding)
            elif not torch.equal(dream.aligned.binding, self.target_binding):
                raise ValueError("Memory state belongs to a different destination model.")
        self.policy.load_state_dict(dream.weights or self._base_state)
        return bank, dream

    def _save(self, bank, dream):
        tensors = {"binding": self.binding, "keys": bank.keys.half(),
                   "values": bank.values.half(), "occupied": bank.occupied}
        if dream.messages:
            tensors.update(dream.tensors())
        if dream.aligned is not None:
            tensors.update(dream.aligned.tensors())
        _atomic_write(tensors, self.state, {"version": FORMAT})

    def _target_rows(self, text, position):
        source = self.encoder.ids(text)
        start, end = source["offset_mapping"][position]
        span = text[start:end].strip()
        if not span:
            return None
        if self._destination is None:
            from .latent_bridge import load_destination

            self._destination = load_destination(self.destination_model_path)
        tokenizer, table = self._destination
        ids = tokenizer.encode(" " + span, add_special_tokens=False)
        if not 1 <= len(ids) <= MAX_VALUE_TOKENS or set(ids).intersection(tokenizer.all_special_ids):
            return None
        return table[ids]

    def dream_status(self):
        fd = self._lock(write=False)
        try:
            _, dream = self._load()
            return dream.report()
        finally:
            os.close(fd)

    @torch.inference_mode()
    def absorb(self, text, *, expected_key=None, expected_value=None, ignore=False):
        if (type(ignore) is not bool or (expected_key is None) != (expected_value is None)
                or (ignore and expected_key is not None)):
            raise ValueError("Supply both verified labels or mark the observation as ignored.")
        verified = ignore or expected_key is not None
        if verified and not self.dream_every:
            raise ValueError("Verified feedback requires a positive dream cadence.")
        positions = None if expected_key is None else _verified_positions(self.encoder, text, expected_key, expected_value)
        features, mask, keys, values = self.encoder.request(text)
        fd = self._lock(write=True)
        try:
            bank, dream = self._load()
            if ignore:
                updated, admitted = bank, False
                dream.add_feedback(DreamExample(features.half().cpu(), mask.cpu(), keys.half().cpu(), write=False))
            else:
                if positions is None:
                    admit, pointers = write_action(self.policy, features, mask)
                    key, value = pointers[0].tolist()
                else:
                    key, value = positions
                    admit = mask[:, key]
                    dream.add_feedback(DreamExample(features.half().cpu(), mask.cpu(), keys.half().cpu(),
                                                    values.half().cpu(), True, key, value))
                selected = write_slots(bank, admit, keys[:, key])
                updated = replace_slots(bank, selected, keys[:, key], values[:, value])
                admitted = bool(admit.item())
            aligned_changed = False
            if admitted and dream.aligned is not None:
                written = selected[0].nonzero().flatten()
                if len(written):
                    slot = written.item()
                    target = self._target_rows(text, value) if self.destination_model_path is not None else None
                    length = int(dream.aligned.lengths[slot])
                    aligned_changed = bool(length) if target is None else not torch.equal(
                        dream.aligned.rows[slot, :length], target,
                    )
                    if aligned_changed:
                        dream.aligned.replace(slot, target)
            changed = any(not torch.equal(getattr(updated, name), getattr(bank, name))
                          for name in ("keys", "values", "occupied"))
            result = {"status": "supported" if admitted else "unknown", "updated": changed}
            if self.dream_every:
                dream.tick(self.policy, self.dream_every, self.dream_steps)
                result["dream"] = dream.report()
            if changed or aligned_changed or self.dream_every:
                self._save(updated, dream)
            return result
        finally:
            os.close(fd)

    def _read(self, text, *, target_aligned=False):
        if target_aligned and self.destination_model_path is None:
            raise ValueError("Target-aligned recall requires a destination model path.")
        # The question produces addresses and policy features. Values are read
        # from the selected slot, without encoding question tokens as values.
        features, mask, keys, _ = self.encoder.request(text, encode_values=False)
        fd = self._lock(write=bool(self.dream_every))
        try:
            bank, dream = self._load()
            action = read_action(self.policy, bank, features, mask, keys).item()
            if self.dream_every:
                snapshot = Bank(bank.keys.half().cpu(), bank.values.half().cpu(), bank.occupied.cpu().clone())
                dream.add_guard(DreamExample(features.half().cpu(), mask.cpu(), keys.half().cpu(),
                                             bank=snapshot, slot=action))
                dream.tick(self.policy, self.dream_every, self.dream_steps)
                self._save(bank, dream)
            value = None
            if action < CAPACITY:
                if target_aligned:
                    length = int(dream.aligned.lengths[action])
                    if length:
                        value = dream.aligned.rows[action, :length].clone()
                else:
                    value = bank.values[:, action].half().cpu()
            return value, (dream.report() if self.dream_every else None)
        finally:
            os.close(fd)

    @torch.inference_mode()
    def vectors(self, text, *, target_aligned=False):
        value, _ = self._read(text, target_aligned=target_aligned)
        return value

    @torch.inference_mode()
    def recall(self, text):
        value, dream = self._read(text)
        answer = "" if value is None else self.encoder.decode(value.to(self.device))
        result = {"status": "supported" if answer else "unknown", "answer": answer}
        if dream is not None:
            result["dream"] = dream
        return result


def invoke(args, device):
    memory = PolicyMemory(args.model, args.codec, args.policy, args.state, device,
                          dream_every=args.dream_every, dream_steps=args.dream_steps,
                          destination_model_path=args.model_35b)
    if args.command == "dream-status":
        return memory.dream_status()
    if args.command == "absorb" and args.feedback_stdin:
        limit = MAX_TEXT * 40
        request = sys.stdin.read(limit + 1)
        if len(request) > limit:
            raise ValueError("The feedback request is too long.")
        payload = json.loads(request)
        if (not isinstance(payload, dict) or "text" not in payload
                or set(payload) - {"text", "expected_key", "expected_value", "ignore"}):
            raise ValueError("Supply a feedback JSON object with text and optional verified labels.")
        return memory.absorb(**payload)
    request = sys.stdin.read(MAX_TEXT + 1)
    if len(request) > MAX_TEXT:
        raise ValueError(f"Supply at most {MAX_TEXT} characters.")
    request = request.rstrip("\n")
    if args.command == "absorb":
        return memory.absorb(request, ignore=args.ignore)
    if args.command == "recall":
        return memory.recall(request)
    if os.path.lexists(args.out):
        raise ValueError("Choose a fresh vector output path.")
    _require_private_directory(args.out.parent)
    with torch.inference_mode():
        value, dream = memory._read(request, target_aligned=args.target_aligned)
    if value is None:
        result = {"status": "unknown"}
    else:
        _atomic_write({"values": value}, args.out, {"version": FORMAT})
        width = value.shape[-1] if args.target_aligned else VALUE_DIM
        result = {"status": "supported", "vector_rows": len(value), "vector_width": width}
    if dream is not None:
        result["dream"] = dream
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--model-35b", type=Path, help="pinned checkpoint for native 35B memory embeddings")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dream-every", type=int, default=0, help="attempt vector replay every N memory messages")
    parser.add_argument("--dream-steps", type=int, default=48, help="maximum optimizer steps per replay")
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    verbs = parser.add_subparsers(dest="command", required=True)
    training = verbs.add_parser("fit")
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--train-episodes", type=int, default=320)
    training.add_argument("--dev-episodes", type=int, default=96)
    training.add_argument("--test-episodes", type=int, default=128)
    training.add_argument("--warmstart-steps", type=int, default=60)
    training.add_argument("--rl-steps", type=int, default=120)
    training.add_argument("--batch-size", type=int, default=24)
    training.add_argument("--timeout", type=float, default=240)
    training.add_argument("--seed", type=int, default=41)
    training.add_argument("--aligned-teacher", action="store_true",
                          help="also compare imitation with explicit correct durable-write labels")
    for action in ("absorb", "recall", "vectors", "dream-status"):
        command = verbs.add_parser(action)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--state", type=Path, required=True)
        if action == "vectors":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--target-aligned", action="store_true", help="return native 35B embedding rows")
        if action == "absorb":
            feedback = command.add_mutually_exclusive_group()
            feedback.add_argument("--feedback-stdin", action="store_true", help="read text and verified labels as JSON")
            feedback.add_argument("--ignore", action="store_true", help="mark the observation as verified non-durable")
    args = parser.parse_args()
    if args.parent_pid is not None:
        _exit_with_parent(args.parent_pid)
    torch.set_num_threads(1)
    device = torch.device("cpu" if args.command == "dream-status" else args.device)
    lease = _gpu_lease() if device.type == "cuda" else None
    try:
        if args.command == "fit":
            args.out.mkdir(mode=0o700)
            fit(args, device)
        else:
            with redirect_stdout(sys.stderr):
                result = invoke(args, device)
            print(json.dumps(result))
    finally:
        if lease is not None:
            os.close(lease)


if __name__ == "__main__":
    main()
