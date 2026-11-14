"""Consolidate approved observations into local evidence and verified policy examples."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
from hashlib import file_digest, sha256
import json
import logging
import os
from pathlib import Path
import re
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from threading import Lock, Thread
from uuid import UUID, uuid4

from lightreason_memory.store import Store, _record, _required_text, _timestamp

from .dream_selection import is_conversation_only
from .skill_rehearsal import DAYS, lesson_key, training_batch

_FACTS = (
    ("profile", re.compile(r"my (?P<key>[\w -]+?) is (?P<value>[^?!\n]+)", re.I),
     "What is my {key}?"),
    ("preference", re.compile(r"I prefer (?P<value>[^?!\n]+?) for (?P<key>[^?!\n]+)", re.I),
     "What do I prefer for {key}?"),
    ("preference", re.compile(r"for (?P<key>[^?!\n]+?), I prefer (?P<value>[^?!\n]+)", re.I),
     "What do I prefer for {key}?"),
    ("preference", re.compile(r"I prefer (?P<value>[^?!\n]+)", re.I),
     "What is my general preference?"),
    ("experiment", re.compile(r"(?:our |the )?experiment (?P<key>[\w -]+?) (?:status is|is) (?P<value>[^?!\n]+)", re.I),
     "What is the status of experiment {key}?"),
    ("process", re.compile(r"(?:our |the )?(?P<key>[\w -]+? (?:schedule|protocol|procedure)) (?:now )?(?:specifies|is) (?P<value>[^?!\n]+)", re.I),
     "What does the {key} specify?"),
)
_COLOR = re.compile(r"(?P<key>[\w-]+) is (?P<value>purple|gray|blue|orange|yellow|green|red|brown|black|white|pink|cyan)", re.I)


_SCOPE = re.compile(r"(experiment|procedure):\s*([^?!:]+)", re.I)
_HEADING = re.compile(r"#{1,6}(?:\s+(.*)|$)")
_FENCE = re.compile(r"(`{3,}|~{3,})(.*)")
_LESSON = re.compile(r"(?:for|when|if) (?P<key>[^?!,:]+),\s*(?P<value>.+)", re.I)
_FIRST_ACTION = re.compile(r"(?P<value>.+?\bfirst) for (?P<key>[^?!,:]+)", re.I)
_BASELINE_DRIFT = re.compile(r"(?:the )?baseline drifts?", re.I)
_ACTION = re.compile(
    r"(?:we\s+)?(?:always\s+)?(?:apply|blind|calibrate|check|choose|clean|collect|compare|confirm|cool|discard|heat|hold|inspect|"
    r"keep|label|load|log|measure|mix|prepare|record|repeat|replace|report|rinse|run|select|set|shut down|"
    r"stop|test|use|verify|wait|weigh)\s+\S",
    re.I,
)
_STATUS = re.compile(r"status:\s*(.+)", re.I)
_TRAILING_EXPERIMENT = re.compile(r"(?:the |our )?(?P<key>[\w -]+?) experiment is (?P<value>[^?!]+)", re.I)
_EXPERIMENT_STATUS = re.compile(
    r"(?:complete(?:d)?|done|finished|ongoing|active|running|in progress|pending|scheduled|paused|"
    r"on hold|on track|delayed|cancelled|canceled|failed|blocked|ready|underway|under review|"
    r"awaiting\s+[^,;!?]+|ready for\s+[^,;!?]+)",
    re.I,
)
_RATIONALE = re.compile(
    r"(?:this|it) (?:avoids|helps|improves|isolates|minimizes|prevents|protects|reduces|removes)\s+\S.*",
    re.I,
)
_RATIONALE_CONSTRAINT = re.compile(r"\b(?:before|after|except|must|never|only|should|unless|without)\b", re.I)
_ILLUSTRATIVE = re.compile(r"for (?:example|instance)\s*,|if we wanted to\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:maybe|perhaps|possibly|probably|tentatively)\b", re.I)
_CORRECTION = re.compile(r"(?im)^\s*(?:(?:[-*+]|\d+[.)])\s+)?correction\s*:")
_EXTRA_STEP = re.compile(r"[,;!?]|\.\s|\b(?:and then|then|alternatively)\b", re.I)
_PROCESS_QUESTION = "What does the {key} specify?"
_EXPERIMENT_QUESTION = "What is the status of experiment {key}?"


def _note_sections(text):
    heading = None
    lines = []
    fence = ""
    for line in text.splitlines()[:80]:
        line = line.strip()
        marker = _FENCE.fullmatch(line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = ""
            continue
        if marker:
            fence = marker[1]
            continue
        title = _HEADING.fullmatch(line)
        scope = _SCOPE.fullmatch((title[1] or "").rstrip(" #") if title else line)
        if title or scope:
            yield heading, lines
            heading = (scope[1].casefold(), scope[2].strip()) if scope else None
            lines = []
            continue
        line = re.sub(r"^(?:[-*+] |\d+[.)]\s+)", "", line)
        line = re.sub(r"^Correction:\s*", "", line, flags=re.I).strip().rstrip(".")
        if line:
            lines.append(line)
    yield heading, lines


def _stated_fact(line):
    for category, pattern, prompt in _FACTS:
        fact = pattern.fullmatch(line)
        if fact:
            key = "general" if "key" not in fact.re.groupindex else fact["key"]
            return category, prompt, key, fact["value"]

    experiment = _TRAILING_EXPERIMENT.fullmatch(line)
    if experiment and _EXPERIMENT_STATUS.fullmatch(experiment["value"]):
        return "experiment", _EXPERIMENT_QUESTION, experiment["key"], experiment["value"]

    fact = _COLOR.fullmatch(line)
    if fact:
        return "profile", "What color is my {key}?", fact["key"], fact["value"]

    lesson = _LESSON.fullmatch(line) or _FIRST_ACTION.fullmatch(line)
    if lesson and not _ILLUSTRATIVE.match(line) and _compact_action(lesson["value"]):
        key = re.sub(r"\s+occurs$", "", lesson["key"].strip(), flags=re.I)
        if _BASELINE_DRIFT.fullmatch(key):
            key = "baseline drift"
        return "process", _PROCESS_QUESTION, key, lesson["value"]
    return None


def _compact_action(value):
    return _ACTION.match(value) and not (_UNCERTAIN.search(value) or _EXTRA_STEP.search(value))


def _single_directive(lines):
    actions = [line for line in lines if _compact_action(line)]
    if len(actions) != 1:
        return None
    action, = actions
    if all(
        _RATIONALE.fullmatch(line) and not (_EXTRA_STEP.search(line) or _RATIONALE_CONSTRAINT.search(line))
        for line in lines if line != action
    ):
        return action
    return None


def _episodes(entry):
    """Extract supplied facts and scoped actions while keeping the whole capture as evidence."""
    for heading, lines in _note_sections(entry["text"]):
        statuses = [] if not heading or heading[0] != "experiment" else [
            match[1].strip() for line in lines if (match := _STATUS.fullmatch(line))
        ]
        directive = _single_directive(lines) if heading and heading[0] == "procedure" else None
        for line in lines:
            fact = _stated_fact(line)
            if not fact and heading:
                category, key = heading
                if category == "experiment" and len(statuses) == 1 and _STATUS.fullmatch(line):
                    fact = ("experiment", _EXPERIMENT_QUESTION, key, statuses[0])
                elif category == "procedure" and line == directive:
                    fact = ("process", _PROCESS_QUESTION, key, line)
            if not fact:
                continue
            category, prompt, key, value = fact
            key = " ".join(key.split())
            value = value.strip()
            if not key or not value or len(key) > 100 or len(value) > 800:
                continue
            is_color = category == "profile" and _COLOR.fullmatch(f"{key} is {value}") is not None
            # The destination tokenizer makes the final 12-row decision.
            compact = (len(value) <= 100 and len(value.split()) <= 12
                       and not (_EXTRA_STEP.search(value) or _UNCERTAIN.search(value)))
            if category == "preference" and line.casefold().startswith("i prefer "):
                compact = compact and " for " not in key.casefold()
            yield {
                "capture_id": entry["id"], "source": entry["source"], "category": category,
                "question": ("What color is my {key}?" if is_color else prompt).format(key=key),
                "answer": value, "key": key, "observation": line + ".",
                "neural_eligible": is_color or compact and category in {"process", "experiment", "preference"},
                "training_kind": "color" if is_color else category,
                "reward": "Exact stated value earns +1; a different value earns -1.",
            }


class ProcedureMemory:
    """Train private addressing weights over frozen Qwen features for compact notes."""

    _format = "vicena-procedure-memory-1"
    _capacity = 24
    _width = 768
    _cutoff = 0.8

    def __init__(self, path, destination_model):
        self.path = path
        self.destination_model = destination_model
        self._destination = None
        self._destination_keys = None

    def _load(self):
        import torch
        from safetensors.torch import load

        from lightreason_memory.latent_freetoken_hook import HIDDEN_SIZE, MAX_ROWS
        from lightreason_memory.neural import _private_open, _require_private_directory

        if not self.path.parent.exists():
            return {}
        _require_private_directory(self.path.parent)
        try:
            fd = _private_open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return {}
        with os.fdopen(fd, "rb") as file:
            raw = file.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Procedure memory is too large.")
        state = load(raw)
        fields = {"keys", "weights", "biases", "vectors", "lengths"}
        if set(state) not in (fields, fields | {"probes"}) or state["keys"].ndim != 2:
            raise ValueError("Procedure memory has an unexpected layout.")
        count = len(state["keys"])
        if 1 <= count <= self._capacity and "probes" not in state:
            state["probes"] = torch.zeros((count, self._width))
        specs = {
            "keys": ((count, 32), torch.uint8),
            "weights": ((count, self._width), torch.float32),
            "biases": ((count,), torch.float32),
            "vectors": ((count, MAX_ROWS, HIDDEN_SIZE), torch.bfloat16),
            "lengths": ((count,), torch.long),
            "probes": ((count, self._width), torch.float32),
        }
        if (not 1 <= count <= self._capacity
                or any((state[k].shape, state[k].dtype) != spec or not torch.isfinite(state[k]).all()
                       for k, spec in specs.items())
                or not state["lengths"].ge(1).all() or not state["lengths"].le(MAX_ROWS).all()
                or len(state["keys"].unique(dim=0)) != count):
            raise ValueError("Procedure memory has invalid tensors.")
        return state

    @staticmethod
    def _encode(encoder, text):
        import torch

        with torch.no_grad():
            features, mask, _, _ = encoder.request(text, encode_values=False)
            # Normalize the pooled lexical and contextual halves independently.
            pooled = features[0, mask[0]].float().mean(0).cpu()
            result = torch.cat([torch.nn.functional.normalize(part, dim=0) for part in pooled.chunk(2)])
        if result.shape != (ProcedureMemory._width,) or not torch.isfinite(result).all() or not result.any():
            raise ValueError("Invalid frozen Qwen features for procedure memory.")
        return result

    def _destination_inputs(self):
        from lightreason_memory.latent_bridge import load_destination

        if self._destination is None:
            self._destination = load_destination(self.destination_model)
        return self._destination

    def _answer_rows(self, answer):
        import torch

        from lightreason_memory.latent_freetoken_hook import HIDDEN_SIZE, MAX_ROWS

        if self.destination_model is None:
            return None
        tokenizer, table = self._destination_inputs()
        ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not 1 <= len(ids) <= MAX_ROWS or set(ids).intersection(tokenizer.all_special_ids):
            return None
        payload = torch.zeros((MAX_ROWS, HIDDEN_SIZE), dtype=torch.bfloat16)
        payload[:len(ids)] = table[ids]
        return payload, len(ids)

    def decode_rows(self, rows):
        import torch

        from lightreason_memory.latent_freetoken_hook import HIDDEN_SIZE, MAX_ROWS

        if (rows.dtype != torch.bfloat16 or rows.device.type != "cpu"
                or rows.ndim != 2 or rows.shape[1] != HIDDEN_SIZE or not 1 <= len(rows) <= MAX_ROWS
                or not torch.isfinite(rows).all()):
            raise ValueError("Invalid native memory rows.")
        if self.destination_model is None:
            raise ValueError("Native memory readout requires the pinned destination tokenizer and embeddings.")
        tokenizer, table = self._destination_inputs()
        token_bits = table.view(torch.int16)[:len(tokenizer)]
        # Probe a few columns first. Bound the full-row comparison even if many keys collide.
        key_width = 8
        if self._destination_keys is None:
            self._destination_keys = token_bits[:, :key_width].contiguous()
        special = set(tokenizer.all_special_ids)
        decoded = []
        for row in rows.contiguous().view(torch.int16):
            candidates = self._destination_keys.eq(row[:key_width]).all(dim=1).nonzero().flatten()
            matches = []
            for ids in candidates.split(1024):
                exact = token_bits[ids].eq(row).all(dim=1)
                matches.extend(token_id for token_id in ids[exact].tolist() if token_id not in special)
                if len(matches) > 1:
                    break
            if len(matches) != 1:
                raise ValueError("Native memory row does not identify a unique non-special token.")
            decoded.append(matches[0])
        answer = tokenizer.decode(decoded, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise ValueError("Native memory rows decoded to empty text.")
        return answer

    def recall(self, encoder, question):
        state = self._load()
        if not state:
            return None
        slot, _ = self._select(state, self._encode(encoder, question))
        if slot is not None:
            return state["vectors"][slot, :state["lengths"][slot]].clone()
        return None

    def _select(self, state, features):
        if not state:
            return None, 0.0
        scores = (state["weights"] @ features + state["biases"]).sigmoid()
        score, slot = scores.max(dim=0)
        return (int(slot) if score >= self._cutoff else None), score.item()

    def _check(self, previous, candidate, question, features, expected, slot, rows, length, split):
        def assess(state):
            selected, score = self._select(state, features)
            if selected is None:
                actual = "abstain"
            elif selected != slot:
                actual = "another procedure"
            # A correction can keep the same address while replacing its value.
            elif (state["lengths"][selected] != length
                  or not state["vectors"][selected, :length].equal(rows[:length])):
                actual = "previous value"
            else:
                actual = expected
            return actual, score, 1 if actual == expected else -1

        before, score_before, reward_before = assess(previous)
        actual, score_after, reward_after = assess(candidate)
        return {"question": question, "expected": expected, "before": before, "actual": actual,
                "reward_before": reward_before, "reward_after": reward_after, "reward": reward_after,
                "score_before": score_before, "score_after": score_after, "split": split}

    def _optimize(self, gate, inputs, targets, on_step=None):
        import torch

        from lightreason_memory.rl_policy import _action

        samples = 24
        points = []
        generator = torch.Generator().manual_seed(42)
        optimizer = torch.optim.Adam(gate.parameters(), lr=.04)
        for step in range(1, 145):
            logits = gate(inputs).flatten()
            actions = torch.stack((torch.zeros_like(logits), logits), dim=-1)
            choice, log_prob, entropy = _action(
                actions.unsqueeze(0).expand(samples, -1, -1), "sample", generator, temperature=2.0,
            )
            reward = choice.eq(targets).float().mul(2).sub(1)
            advantage = (samples * reward - reward.sum(dim=0)) / (samples - 1)
            loss = -(advantage * log_prob + .002 * entropy).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                scores = gate(inputs).flatten().sigmoid()
                expected = torch.where(targets == 1, scores, 1 - scores)
                complete = bool(expected.min() > .975)
            if step == 1 or step % 8 == 0 or complete:
                # These measurements use the same pre-update policy as the sampled reward.
                sample_count = reward.numel()
                sample = {"step": step, "sample_count": sample_count, "mean_reward": reward.mean().item(),
                          "mean_entropy_nats": entropy.detach().mean().item(),
                          "success_rate": reward.eq(1).sum().item() / sample_count}
                points.append(sample)
                sampling = {"first": points[0], "last": sample, "points": points.copy()}
                if on_step is not None:
                    on_step(status="training", optimizer_steps=step, sampling=sampling)
            if complete:
                break
        return step, sampling

    @staticmethod
    def question_plan(example):
        key = example["key"]
        question = example["question"]
        kind = example.get("training_kind")
        if kind == "experiment":
            train_positive = f"How is the {key} experiment progressing?"
            check_positive = f"Where does the {key} experiment stand?"
            alternate = f"What is the latest update on the {key} experiment?"
        elif kind == "preference" and key.casefold() == "general":
            train_positive = "What do I usually prefer?"
            check_positive = "What is my usual choice?"
            alternate = "Which option do I tend to prefer?"
        elif kind == "preference":
            train_positive = f"What is my preference for {key}?"
            check_positive = f"Which option do I usually choose for {key}?"
            alternate = f"What do I tend to favor for {key}?"
        else:
            train_positive = f"What are the steps or settings for the {key}?"
            check_positive = f"For the {key}, what should I set or do?"
            alternate = f"How should I handle the {key} in the next run?"
        if question.casefold() == check_positive.casefold():
            check_positive = alternate
        positive = {"expected": "recall", "answer": example["answer"]}
        negative = {"expected": "abstain"}
        plan = [
            {"question": question, "split": "training", **positive},
            {"question": train_positive, "split": "training", **positive},
            {"question": "What are the steps or settings for the separate reference procedure?",
             "split": "training", **negative},
            {"question": "What does the unrelated experiment specify?", "split": "training", **negative},
            {"question": check_positive, "split": "held out", **positive},
            {"question": "What settings does the unrelated reference procedure use?", "split": "held out", **negative},
        ]
        for case in plan:
            case["origin"] = "local"
        variants = example.get("question_variants")
        if isinstance(variants, (list, tuple)) and len(variants) == 2:
            def normalized(text):
                return re.sub(r"\W+", " ", text.casefold()).strip()

            taken = {normalized(case["question"]) for case in plan}
            novel = [normalized(text) for text in variants if isinstance(text, str)]
            if len(novel) == 2 and all(novel) and len(set(novel)) == 2 and not taken.intersection(novel):
                plan.extend({"question": text.strip(), "split": split, "origin": "gemini", **positive}
                            for text, split in zip(variants, ("training", "held out"), strict=True))
        return plan

    def train(self, encoder, example, progress, *, trace=None, plan=None, on_progress=None):
        import torch

        trace = trace if trace is not None else {}
        trace.update(status="skipped", sampling={})
        answer = self._answer_rows(example["answer"])
        if answer is None:
            progress("A note needs more than 12 native embedding rows. Use saved evidence for its full text.")
            return None, 0, []
        key = example["key"]
        state = self._load()
        address = torch.tensor(list(sha256(key.casefold().encode()).digest()), dtype=torch.uint8)
        count = len(state["keys"]) if state else 0
        matching = (state["keys"] == address).all(dim=1).nonzero().flatten() if state else []
        slot = int(matching[0]) if len(matching) else count
        if slot >= self._capacity:
            progress("Procedure memory is full. Existing procedures were preserved.")
            return None, 0, []
        if plan is None:
            plan = self.question_plan(example)
        local_plan = [case for case in plan if case.get("origin") != "gemini"]
        features = {}

        def encode(question):
            if question not in features:
                features[question] = self._encode(encoder, question)
            return features[question]

        def fallback(reason, discarded_steps=0):
            trace.update(plan=local_plan, planner_fallback=reason, discarded_optimizer_steps=discarded_steps,
                         sampling={})
            progress(reason)
            if on_progress is not None:
                on_progress(status="training", optimizer_steps=discarded_steps, sampling={}, plan=local_plan,
                            planner_fallback=reason, discarded_optimizer_steps=discarded_steps)

                def publish(*, status, optimizer_steps, sampling):
                    on_progress(status=status, optimizer_steps=discarded_steps + optimizer_steps, sampling=sampling,
                                plan=local_plan, planner_fallback=reason, discarded_optimizer_steps=discarded_steps)
            else:
                publish = None
            checkpoint, steps, checks = self._train_plan(
                encode, example, progress, local_plan, state, slot, address, answer, trace, publish,
            )
            return checkpoint, discarded_steps + steps, checks

        if len(local_plan) != len(plan):
            try:
                for case in plan:
                    if case.get("origin") == "gemini":
                        encode(case["question"])
            except ValueError:
                return fallback("Gemini questions could not be encoded. Using local questions.")
        result = self._train_plan(encode, example, progress, plan, state, slot, address, answer, trace, on_progress)
        if trace["status"] == "held" and len(local_plan) != len(plan):
            return fallback("The Gemini question plan failed validation. Retrying with local questions.", result[1])
        return result

    def _train_plan(self, encode, example, progress, plan, state, slot, address, answer, trace, on_progress):
        import torch

        from lightreason_memory.neural import _atomic_write, _require_private_directory

        train_cases = [case for case in plan if case["split"] == "training"]
        check_cases = [case for case in plan if case["split"] == "held out"]
        check = [encode(case["question"]) for case in check_cases]
        rows, length = answer
        key = example["key"]
        count = len(state["keys"]) if state else 0

        def evaluate(candidate, cases, values):
            outcomes = []
            for case, vector in zip(cases, values, strict=True):
                recall = case["expected"] == "recall"
                outcome = self._check(state, candidate, case["question"], vector, case["expected"],
                                      slot if recall else None, rows, length, case["split"])
                if recall:
                    outcome["answer"] = example["answer"]
                outcome["origin"] = case.get("origin", "local")
                outcomes.append(outcome)
            return outcomes

        if slot < count:
            baseline = evaluate(state, check_cases, check)
            if all(outcome["reward"] == 1 for outcome in baseline):
                trace["status"] = "unchanged"
                return None, 0, baseline
        train = torch.stack([encode(case["question"]) for case in train_cases])
        targets = torch.tensor([int(case["expected"] == "recall") for case in train_cases])
        with torch.enable_grad():
            gate = torch.nn.Linear(self._width, 1)
            with torch.no_grad():
                gate.weight.zero_()
                gate.bias.fill_(-2.)
            step, trace["sampling"] = self._optimize(gate, train, targets, on_progress)
        if on_progress is not None:
            on_progress(status="validating", optimizer_steps=step, sampling=trace["sampling"])
        entry = {"keys": address, "weights": gate.weight[0].detach(), "biases": gate.bias[0].detach(),
                 "vectors": rows, "lengths": torch.tensor(length), "probes": check[0]}
        if slot < count:
            candidate = {name: tensor.clone() for name, tensor in state.items()}
            for name, value in entry.items():
                candidate[name][slot] = value
        elif state:
            candidate = {name: torch.cat((state[name], value.unsqueeze(0))) for name, value in entry.items()}
        else:
            candidate = {name: value.unsqueeze(0) for name, value in entry.items()}
        checks = evaluate(candidate, check_cases, check) + evaluate(candidate, train_cases, train)
        for old_slot, probe in enumerate(state.get("probes", ())):
            if old_slot == slot or not probe.any():
                continue
            ident = bytes(state["keys"][old_slot].tolist()).hex()[:8]
            checks.append(self._check(state, candidate, f"Prior procedure {ident}", probe, "retain", old_slot,
                                      state["vectors"][old_slot], state["lengths"][old_slot], "retention"))
        if any(outcome["reward"] < 0 for outcome in checks if outcome["split"] != "training"):
            trace["status"] = "held"
            if not any(case.get("origin") == "gemini" for case in plan):
                progress(f"Procedure policy for {key} did not pass validation. Earlier memory was kept.")
            return None, step, checks
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_directory(self.path.parent)
        _atomic_write(candidate, self.path, {"format": self._format})
        with self.path.open("rb") as file:
            checkpoint = file_digest(file, "sha256").hexdigest()
        progress(f"Learned procedure recall for {key} in {step} optimizer steps.")
        trace["status"] = "promoted"
        return checkpoint, step, checks


class PolicyTrainer:
    """Run the existing bounded reward/replay trainer with verified single-token labels."""

    def __init__(self, model_path, codec_path, policy_path, state_path, destination_model_path=None, *, device="cpu"):
        self.state_path = Path(state_path)
        self._args = (model_path, codec_path, policy_path, self.state_path, device)
        self._destination_model = destination_model_path
        self.procedure = ProcedureMemory(self.state_path.with_suffix(".procedure.safetensors"), destination_model_path)
        from .skill_memory import SkillMemory

        self.skills = SkillMemory(self.state_path.with_suffix(".skills.safetensors"), self.procedure._answer_rows)
        self._reader = None
        self._lock = Lock()

    def _memory(self, cadence):
        import torch
        from lightreason_memory.rl_memory import PolicyMemory

        if self._args[-1] == "cpu" and torch.get_num_threads() > 8:
            torch.set_num_threads(8)
        return PolicyMemory(*self._args, destination_model_path=self._destination_model, dream_every=cadence)

    def vectors(self, question):
        with self._lock:
            return self._vectors(question)

    def readout(self, question) -> str | None:
        """Decode only the rows selected from the active neural checkpoints."""
        with self._lock:
            rows = self._vectors(question)
            return None if rows is None else self.procedure.decode_rows(rows)

    def _vectors(self, question):
        has_color = self.state_path.is_file()
        has_procedure = self.procedure.path.is_file()
        has_skills = self.skills.path.is_file()
        if not (has_color or has_procedure or has_skills):
            return None
        if self._reader is None:
            self._reader = self._memory(0)
        try:
            self._reader.encoder.ids(question)
        except ValueError:
            return None
        if has_procedure:
            rows = self.procedure.recall(self._reader.encoder, question)
            if rows is not None:
                return rows
        if has_skills:
            rows = self.skills.recall(self._reader.encoder, question)
            if rows is not None:
                return rows
        if has_color:
            return self._reader.vectors(question, target_aligned=True)
        return None

    def skill_checkpoint_digest(self):
        from lightreason_memory.neural import _private_open

        with self._lock:
            try:
                fd = _private_open(self.skills.path, os.O_RDONLY)
            except FileNotFoundError:
                return None
            with os.fdopen(fd, "rb") as checkpoint:
                return file_digest(checkpoint, "sha256").hexdigest()

    def train_skills(self, batch, progress, prepare):
        from lightreason_memory.neural import _private_open, _require_private_directory
        from .skill_memory import SkillMemory

        with self._lock:
            if self._reader is None:
                self._reader = self._memory(0)
            active = self.skills.path
            active.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _require_private_directory(active.parent)
            with TemporaryDirectory(prefix=".dream-candidate-", dir=active.parent) as folder:
                candidate = Path(folder) / active.name
                if active.exists():
                    with (os.fdopen(_private_open(active, os.O_RDONLY), "rb") as original,
                          os.fdopen(_private_open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "wb") as copy):
                        copyfileobj(original, copy)
                result = SkillMemory(candidate, self.skills.row_encoder).train(
                    self._reader.encoder, **batch, progress=progress,
                )
                if result["promoted"]:
                    # Journal the validated result before publishing the tensor.
                    prepare(result)
                    candidate.replace(active)
                return result

    def evaluate_skills(self, batch):
        with self._lock:
            if self._reader is None:
                self._reader = self._memory(0)
            return self.skills.evaluate(self._reader.encoder, batch["checks"], labels=batch["labels"])


    def __call__(self, examples, progress, *, on_policy_run=None):
        eligible = [item for item in examples if item["neural_eligible"]]
        report = {"configured": True, "eligible": len(eligible), "attempts": 0, "promotions": 0,
                  "optimizer_steps": 0, "skipped": len(examples) - len(eligible), "outcomes": [],
                  "recall_checks": [], "procedure_updates": 0, "procedure_checks": [], "policy_runs": []}
        if not eligible:
            return report
        with self._lock:
            memory = self._memory(1)
            if self._reader is not None:
                memory._encoder = self._reader.encoder
            colors = [item for item in eligible if item["training_kind"] == "color"]
            for candidate_id, item in enumerate(eligible, start=1):
                if item["training_kind"] not in {"process", "experiment", "preference"}:
                    continue
                plan = self.procedure.question_plan(item)
                candidate = {"candidate_id": candidate_id, "key": item["key"], "answer": item["answer"],
                             "source": item.get("source"), "algorithm": "REINFORCE", "plan": plan}

                def publish(*, status, optimizer_steps=0, sampling=None, **fields):
                    on_policy_run({**candidate, "status": status, "optimizer_steps": optimizer_steps,
                                   "sampling": sampling or {}, "checks": [],
                                   "before_passed": 0, "after_passed": 0, **fields})

                if on_policy_run is not None:
                    publish(status="training")
                trace = {}
                checkpoint, steps, checks = self.procedure.train(
                    memory.encoder, item, progress, trace=trace, plan=plan,
                    on_progress=publish if on_policy_run is not None else None,
                )
                validation = [check for check in checks if check["split"] != "training"]
                run = {**candidate, "optimizer_steps": steps, "checks": checks,
                       "before_passed": sum(check["reward_before"] == 1 for check in validation),
                       "after_passed": sum(check["reward_after"] == 1 for check in validation), **trace}
                report["policy_runs"].append(run)
                if on_policy_run is not None:
                    on_policy_run(run)
                discarded = trace.get("discarded_optimizer_steps", 0)
                report["attempts"] += int(discarded > 0) + int(steps > discarded)
                report["optimizer_steps"] += steps
                report["procedure_checks"].extend(validation)
                report["procedure_updates"] += int(checkpoint is not None)
                report["skipped"] += int(checkpoint is None)
                if checkpoint is not None:
                    report["procedure_checkpoint"] = checkpoint
            before = memory.dream_status()
            for item in colors:
                try:
                    result = memory.absorb(item["observation"], expected_key=item["key"],
                                           expected_value=item["answer"])
                except ValueError:
                    report["skipped"] += 1
                    progress("A candidate did not have a unique single-token key and value.")
                    continue
                state = result["dream"]
                report["optimizer_steps"] += state["steps"]
                report["outcomes"].append(state["status"])
                progress(f"Policy replay for {item['key']}: {state['status']} ({state['steps']} steps).")
            after = memory.dream_status()
            report["attempts"] += after["attempts"] - before["attempts"]
            report["promotions"] = after["generation"] - before["generation"] + report["procedure_updates"]
            report["generation"] = after["generation"]
            reader = self._reader or self._memory(0)
            reader._encoder = memory.encoder
            self._reader = reader
            for item in colors:
                result = reader.recall(item["question"])
                actual = result["answer"]
                reward = 1 if actual.casefold() == item["answer"].casefold() else -1
                report["recall_checks"].append({"question": item["question"], "expected": item["answer"],
                                                "actual": actual, "reward": reward})
        return report


class Dream:
    """Persist approved captures before processing them in a single Dream job."""

    def __init__(self, db_path, trainer=None, *, planner=None):
        self.store = Store(db_path, "vicena-chat", "personal")
        self.trainer = trainer
        self.planner = planner
        self._run_lock_path = self.store._path.with_name(self.store._path.name + ".dream.lock")
        with self.store._connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS dream_capture (
                       seq INTEGER PRIMARY KEY, id TEXT UNIQUE, text TEXT, source TEXT, kind TEXT, at TEXT,
                       promoted INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS dream_run (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS dream_rehearsal (
                       stage TEXT PRIMARY KEY, payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS dream_skill_intent (
                       id INTEGER PRIMARY KEY, stage TEXT NOT NULL, payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS dream_fact (
                       fact_key TEXT PRIMARY KEY, memory_id TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS dream_wiki (
                       memory_id TEXT PRIMARY KEY REFERENCES memory_records(memory_id),
                       title TEXT NOT NULL, summary TEXT NOT NULL, tags TEXT NOT NULL)""")
            db.execute("INSERT OR IGNORE INTO dream_run VALUES (1, ?)", (json.dumps({
                "status": "idle", "stage": "capture", "events": [], "entries": [], "questions": [],
                "count_run_promoted": 0, "training": None, "error": None,
            }),))
        lease = self._lease()
        if lease is not None:
            try:
                if self.status()["status"] == "running":
                    self._progress("failed", "The previous Dream was interrupted. Pending captures are available.",
                                   status="failed", error="Previous Dream was interrupted.")
            finally:
                os.close(lease)

    def _lease(self):
        fd = os.open(self._run_lock_path, os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise

    def capture(self, text, source, kind="message", *, capture_request_id=None):
        text = _required_text(text, "capture", 16_384).strip()
        source = _required_text(source, "source", 1_024).strip()
        if kind not in {"message", "document"}:
            raise ValueError("Capture kind must be message or document.")
        if capture_request_id is not None:
            if not isinstance(capture_request_id, str):
                raise ValueError("Capture request ID must be a canonical version 4 UUID.")
            try:
                parsed = UUID(capture_request_id)
            except ValueError:
                raise ValueError("Capture request ID must be a canonical version 4 UUID.") from None
            if parsed.version != 4 or str(parsed) != capture_request_id:
                raise ValueError("Capture request ID must be a canonical version 4 UUID.")
        key = capture_request_id or uuid4().hex
        at = _timestamp(datetime.now(timezone.utc))
        with self.store._connect(write=True) as db:
            if capture_request_id is not None:
                previous = db.execute("SELECT text,source,kind FROM dream_capture WHERE id = ?", (key,)).fetchone()
                if previous is not None:
                    if tuple(previous) != (text, source, kind):
                        raise ValueError("Capture request ID already identifies another message.")
                    return key
            db.execute("INSERT INTO dream_capture (id,text,source,kind,at) VALUES (?, ?, ?, ?, ?)",
                       (key, text, source, kind, at))
        return key

    @staticmethod
    def _follow_up(text, source, category, *, recognized=False):
        lines = {lesson_key(line) for line in text.splitlines() if line.strip()}
        if not lines or lines <= {lesson_key(lesson) for day in DAYS for lesson in day}:
            return None
        questions = {
            "process": "For this procedure, what is the first action you want Vicena to learn?",
            "experiment": "What is the current outcome or status of this experiment in one sentence?",
            "preference": "What is one specific preference you want Vicena to remember?",
            "profile": "Is there a specific preference or recurring procedure you want Vicena to learn from this note?",
        }
        return {"source": source, "question": questions.get(
            category, "What first action, experiment outcome, or preference should Vicena learn from this note?"
        ), "reason": "not_trainable" if recognized else "unrecognized"}

    def status(self):
        with self.store._connect() as db:
            db.execute("BEGIN")
            report = json.loads(db.execute("SELECT payload FROM dream_run WHERE id = 1").fetchone()[0])
            # Captures can be pending (0), archived as evidence (1), or kept out of the wiki (2).
            captures, promoted, pending = db.execute(
                "SELECT count(*), count(CASE WHEN promoted = 1 THEN 1 END), "
                "count(CASE WHEN promoted = 0 THEN 1 END) FROM dream_capture"
            ).fetchone()
        report.setdefault("night", 0)
        report.setdefault("history", [])
        report.setdefault("intake", None)
        report.setdefault("clarifications", [])
        if report["status"] == "complete" and report["intake"] is None and not report["questions"]:
            follow_ups = [
                follow_up for entry in report["entries"]
                if (follow_up := self._follow_up(entry["text"], entry["source"], entry["category"])) is not None
            ]
            if follow_ups:
                report["clarifications"] = follow_ups
                report["intake"] = {"status": "needs_detail", "captured_count": len(report["entries"]),
                                    "recognized_lessons": 0, "trainable_lessons": 0}
        training = report["training"] or {}
        report["counts"] = {"captures": captures, "promoted": promoted, "pending": pending,
                            "questions": len(report["questions"]), "neural_examples": training.get("eligible", 0)}
        return report

    def _progress(self, stage, message, **fields):
        with self.store._connect(write=True) as db:
            report = json.loads(db.execute("SELECT payload FROM dream_run WHERE id = 1").fetchone()[0])
            report.update(stage=stage, **fields)
            report["events"] = [*report["events"], {"stage": stage, "message": message}][-50:]
            db.execute("UPDATE dream_run SET payload = ? WHERE id = 1", (json.dumps(report),))

    def start(self, *, retry=False):
        lease = self._lease()
        if lease is None:
            return self.status()
        try:
            claimed = self._claim_night(retry)
            if claimed:
                Thread(target=self._run, args=(lease, *claimed), daemon=True, name="vicena-dream").start()
            else:
                os.close(lease)
        except BaseException:
            os.close(lease)
            raise
        return self.status()

    def _claim_night(self, retry_requested):
        with self.store._connect(write=True) as db:
            report = json.loads(db.execute("SELECT payload FROM dream_run WHERE id = 1").fetchone()[0])
            pending = [row["seq"] for row in db.execute(
                "SELECT seq FROM dream_capture WHERE promoted = 0 ORDER BY seq LIMIT 16")]
            cut = pending[-1] if pending else db.execute("SELECT coalesce(max(seq), 0) FROM dream_capture").fetchone()[0]
            night = report.get("night", 0)
            skill = report.get("skill") or {}
            training = report.get("training") or {}
            held = skill.get("retryable") or any(
                run["status"] == "held" for run in training.get("policy_runs", ())
            )
            # A legacy scripted correction may already be archived from the first
            # rehearsal pass. Its second pass still needs a turn.
            rehearsal_due = False
            if not pending and skill.get("status") == "promoted" and skill.get("stage") == "day1":
                batch = training_batch(self._approved_rehearsal(db, cut))
                rehearsal_due = bool(batch and batch["supersedes"])
            retry = report["status"] in {"failed", "running"} or bool(
                held and (retry_requested or (not pending and skill.get("retryable")))
            )
            if not pending and not retry and not rehearsal_due:
                return None
            history = report.get("history", [])
            if retry and "batch" in report:
                first, cut = report["batch"]
            else:
                first = pending[0] if pending else cut + 1
            if not retry or night == 0:
                if night:
                    history = [*history, {
                        "night": night, "status": report["status"], "questions": len(report["questions"]),
                        "captures": (report.get("selection") or {}).get("reviewed", len(report["entries"])),
                        "neural_updates": training.get("promotions", 0),
                    }][-16:]
                night += 1
            message = "Reading approved chat messages and documents."
            next_report = {"night": night, "history": history, "batch": [first, cut],
                           "status": "running", "stage": "capture", "attempt_id": uuid4().hex,
                           "entries": [], "questions": [],
                           "events": [{"stage": "capture", "message": message}],
                           "count_run_promoted": 0, "training": None, "skill": None, "error": None}
            db.execute("UPDATE dream_run SET payload = ? WHERE id = 1", (json.dumps(next_report),))
            return first, cut

    @staticmethod
    def _approved_rehearsal(db, cut):
        return [dict(row) for row in db.execute(
            "SELECT text FROM dream_capture WHERE seq <= ? AND promoted != 2 AND lower(text) LIKE '%synthetic%' "
            "AND lower(text) LIKE '%rehearsal%' ORDER BY seq", (cut,))]

    def rehearsal_report(self):
        with self.store._connect() as db:
            rows = db.execute("SELECT stage, payload FROM dream_rehearsal").fetchall()
        return {row["stage"]: json.loads(row["payload"]) for row in rows}

    def _record_skill_result(self, stage, result):
        with self.store._connect(write=True) as db:
            db.execute("INSERT OR REPLACE INTO dream_rehearsal VALUES (?, ?)", (stage, json.dumps(result)))
            db.execute("DELETE FROM dream_skill_intent WHERE id = 1")

    def _rehearse(self, approved):
        if not hasattr(self.trainer, "train_skills"):
            return None
        actual_digest = self.trainer.skill_checkpoint_digest()
        with self.store._connect() as db:
            intent = db.execute("SELECT stage, payload FROM dream_skill_intent WHERE id = 1").fetchone()
        if intent:
            result = json.loads(intent["payload"])
            if actual_digest == result["checkpoint_sha256"]:
                self._record_skill_result(intent["stage"], result)
                self._progress("train", "Recovered a promoted skill checkpoint after an interrupted report write.")
            else:
                with self.store._connect(write=True) as db:
                    db.execute("DELETE FROM dream_skill_intent WHERE id = 1")

        report = self.rehearsal_report()
        day1 = report.get("day1", {})
        day2 = report.get("day2", {})
        last_promoted = day2 if day2.get("promoted") else day1 if day1.get("promoted") else {}
        if actual_digest != last_promoted.get("checkpoint_sha256"):
            return {"status": "inconsistent", "retryable": False,
                    "message": "The skill checkpoint and training report disagree. Recover the matched checkpoint before training."}
        if day2.get("promoted"):
            return {"status": "up_to_date", "stage": "day2", "retryable": False,
                    "message": "The two-day skill rehearsal is up to date."}

        stage = "day2" if day1.get("promoted") else "day1"
        # Limit the first run to Day 1 even if the correction is already in
        # approved notes. The next run checks the correction's capture order.
        lessons = {lesson_key(text) for text in DAYS[0]}
        if stage == "day2":
            lessons.update(lesson_key(text) for text in DAYS[1])
        captures = [{"text": line} for row in approved for line in row["text"].splitlines()
                    if lesson_key(line) in lessons]
        if not captures and not day1:
            return None
        batch = training_batch(captures)
        if batch is None or (stage == "day2" and not batch["supersedes"]):
            return {"status": "awaiting_lessons", "stage": stage, "retryable": False,
                    "message": "Waiting for both Day 1 lessons." if stage == "day1" else
                               "Waiting for a correction captured after both Day 1 lessons."}

        self._progress("train", f"Training the local 3B skill policy for {stage.replace('day', 'day ')}.")

        def prepare(result):
            with self.store._connect(write=True) as db:
                db.execute("INSERT OR REPLACE INTO dream_skill_intent VALUES (1, ?, ?)",
                           (stage, json.dumps(result)))

        result = self.trainer.train_skills(batch, lambda message: self._progress("train", message), prepare)
        self._record_skill_result(stage, result)
        if result["promoted"]:
            return {"status": "promoted", "stage": stage, "retryable": False,
                    "message": "Dream promoted the skill checkpoint."}
        return {"status": "held", "stage": stage, "retryable": True,
                "message": "The skill candidate failed validation. The active memory is unchanged. Run Dream to retry."}

    def _run(self, lease, first, cut):
        try:
            with self.store._connect() as db:
                batch = [dict(row) for row in db.execute(
                    "SELECT * FROM dream_capture WHERE seq BETWEEN ? AND ? ORDER BY seq", (first, cut))]
                approved = self._approved_rehearsal(db, cut)
            episodes = []
            entries = []
            clarifications = []
            chat_only = set()
            for row in batch:
                facts = list(_episodes(row))[:4]
                auto_chat = row["kind"] == "message" and row["source"] == "Saved chat message"
                if auto_chat and not facts and is_conversation_only(row["text"]):
                    chat_only.add(row["id"])
                    continue
                if auto_chat:
                    correction = _CORRECTION.search(row["text"]) is not None
                    for fact in facts:
                        fact.update(auto_chat=True, correction=correction)
                category = facts[0]["category"] if facts else row["kind"]
                episodes.extend(facts)
                entries.append({"id": row["id"], "text": row["text"][:1000], "source": row["source"],
                                "category": category})
                if any(fact["neural_eligible"] for fact in facts):
                    continue
                follow_up = self._follow_up(row["text"], row["source"], category, recognized=bool(facts))
                if follow_up:
                    clarifications.append(follow_up)
            latest_lessons = set()
            for fact in reversed(episodes):
                identity = (fact["category"], fact["key"].casefold())
                if identity in latest_lessons and fact["neural_eligible"]:
                    fact.update(neural_eligible=False, selection_reason="A newer note revises this lesson.")
                latest_lessons.add(identity)
            pending = [row for row in batch if not row["promoted"] and row["id"] not in chat_only]
            self._progress("questions", f"Reviewed {len(batch)} captures. Checking {len(entries)} notes for lessons.",
                           entries=entries)
            wiki = self._plan_with_gemini(entries, episodes, {row["id"] for row in pending})
            eligible = [fact for fact in episodes if fact["neural_eligible"]]
            intake = None
            if episodes or clarifications:
                status = "ready" if eligible else "evidence_only" if episodes else "needs_detail"
                intake = {"status": status, "captured_count": len(entries), "recognized_lessons": len(episodes),
                          "trainable_lessons": len(eligible)}
            public_fields = ("question", "answer", "category", "source", "reward", "neural_eligible", "selection_reason")
            public = [{key: item[key] for key in public_fields if key in item} for item in episodes]
            selection = {"reviewed": len(batch), "conversation_only": len(chat_only),
                         "excluded_lessons": sum("selection_reason" in fact for fact in episodes),
                         "training_lessons": len(eligible)}
            message = f"Selected {len(eligible)} training candidates from {len(entries)} approved notes."
            if clarifications:
                message += f" Added {len(clarifications)} follow-up questions for notes that need detail."
            self._progress("review", message, questions=public, intake=intake,
                           clarifications=clarifications, selection=selection)
            if self.trainer and eligible:
                self._progress("train", f"Running local reward and replay checks on {len(eligible)} verified examples.")
                def progress(message):
                    self._progress("train", message)

                if isinstance(self.trainer, PolicyTrainer):
                    candidates = {}

                    def publish(run):
                        candidates[run["candidate_id"]] = run
                        runs = list(candidates.values())
                        key, status, steps = run["key"], run["status"], run["optimizer_steps"]
                        if status == "training":
                            message = f"{key}: optimizer step {steps}." if steps else f"{key}: training questions ready. Preparing model features."
                        elif status == "validating":
                            message = f"{key}: checking the candidate before saving."
                        else:
                            message = f"{key}: policy {status}."
                        self._progress("train", message, training={
                            "configured": True, "eligible": len(eligible), "policy_runs": runs,
                            "attempts": sum(item["optimizer_steps"] > 0 for item in runs),
                            "optimizer_steps": sum(item["optimizer_steps"] for item in runs),
                            "promotions": sum(item["status"] == "promoted" for item in runs),
                        })

                    training = self.trainer(episodes, progress, on_policy_run=publish)
                else:
                    training = self.trainer(episodes, progress)
            else:
                reason = "No explicit compact lessons in this batch." if self.trainer else "Policy runtime unavailable."
                training = {"configured": self.trainer is not None, "eligible": len(eligible), "attempts": 0,
                            "promotions": 0, "optimizer_steps": 0, "reason": reason}
                self._progress("train", reason)
            if chat_only:
                # Keep the capture receipt for idempotent chat retries, with no wiki record.
                with self.store._connect(write=True) as db:
                    db.executemany("UPDATE dream_capture SET promoted = 2 WHERE id = ? AND promoted = 0",
                                   [(capture_id,) for capture_id in chat_only])
            skill = self._rehearse(approved)
            self._progress("promote", f"Saving {len(pending)} approved captures in local evidence.",
                           training=training, skill=skill)
            for row in pending:
                self._promote(row, [fact for fact in episodes if fact["capture_id"] == row["id"]], wiki.get(row["id"]))
            updates = training.get("promotions", 0)
            held = sum(run["status"] == "held" for run in training.get("policy_runs", ()))
            if held:
                message = f"Dream saved {updates} neural updates and held {held} candidates that failed validation."
            elif updates:
                message = f"Dream saved {updates} neural updates. You can use them in a new chat."
            elif skill and skill["status"] != "up_to_date":
                message = skill["message"]
            elif chat_only and not entries:
                message = f"Reviewed {len(chat_only)} chat-only turns. No training examples were selected."
            elif intake and not eligible and clarifications:
                message = "Dream saved the approved notes as evidence. Answer a follow-up to provide a trainable lesson."
            else:
                message = "Dream finished processing the approved material. No new neural updates."
            self._progress("complete", message, status="complete", count_run_promoted=len(pending))
        except Exception as error:
            logging.getLogger(__name__).error("Dream failed with %s", type(error).__name__)
            self._progress("failed", "Dream stopped. Unprocessed captures remain available for another run.",
                           status="failed", error="Could not finish Dream. Retry to resume pending captures.")
        finally:
            os.close(lease)

    def _plan_with_gemini(self, entries, episodes, pending_ids):
        if not self.planner or not pending_ids:
            return {}
        planning_entries = [entry for entry in entries if entry["id"] in pending_ids]
        planning_episodes = [ep for ep in episodes if ep["capture_id"] in pending_ids
                             and ep["neural_eligible"] and ep["training_kind"] in {"process", "experiment", "preference"}]
        self._progress("questions", "Gemini is drafting practice questions and organizing wiki notes.",
                       planning={"status": "running"})
        try:
            plan = self.planner.plan(planning_entries, planning_episodes)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            logging.getLogger(__name__).warning("Dream planner failed (%s)", type(error).__name__)
            plan = {"questions": {}, "wiki": []}
        variants = plan["questions"]
        skipped = set(plan.get("skip", ()))
        excluded = 0
        for index, episode in enumerate(planning_episodes):
            if (index in skipped and episode.get("auto_chat") and not episode.get("correction")
                    and episode["category"] in {"experiment", "profile"}):
                episode.update(neural_eligible=False, selection_reason="The planner kept this detail as evidence only.")
                excluded += 1
            elif index in variants:
                episode["question_variants"] = variants[index]
        wiki = {item["capture_id"]: item for item in plan["wiki"] if item["capture_id"] in pending_ids}
        if variants or wiki or excluded:
            self._progress("questions",
                           f"Gemini drafted {2 * len(variants)} practice and validation questions and organized {len(wiki)} wiki notes.",
                           planning={"status": "ready", "proposed_questions": 2 * len(variants),
                                     "wiki_notes": len(wiki), "excluded_lessons": excluded})
        else:
            self._progress("questions", "Gemini returned no usable proposals. Dream is using its local questions and wiki index.",
                           planning={"status": "fallback", "proposed_questions": 0, "wiki_notes": 0})
        return wiki

    def _promote(self, row, facts, wiki=None):
        # One explicit fact can supersede an earlier capture. A document with
        # several facts is kept intact as evidence for each of its topics.
        fact_key = None
        if len(facts) == 1 and row["kind"] == "message":
            fact_key = f"{facts[0]['category']}:{facts[0]['key'].casefold()}"
        with self.store._connect(write=True) as db:
            previous = db.execute("SELECT memory_id FROM dream_fact WHERE fact_key = ?", (fact_key,)).fetchone()
            saved = self.store._insert(db, event_id=f"dream:{row['id']}", content=row["text"],
                                       source=row["source"], timestamp=row["at"], kind="user_report",
                                       supersedes_id=previous["memory_id"] if previous else None)
            if previous:
                db.execute("UPDATE memory_records SET status = 'superseded', superseded_by_id = ? WHERE memory_id = ?",
                           (saved["memory_id"], previous["memory_id"]))
            if fact_key:
                db.execute("INSERT OR REPLACE INTO dream_fact VALUES (?, ?)",
                           (fact_key, saved["memory_id"]))
            if wiki:
                db.execute("INSERT INTO dream_wiki VALUES (?, ?, ?, ?)",
                           (saved["memory_id"], wiki["title"], wiki["summary"], json.dumps(wiki["tags"])))
            db.execute("UPDATE dream_capture SET promoted = 1 WHERE id = ?", (row["id"],))

    _WIKI_SCOPE = "m.user_id = ? AND m.workspace_id = ? AND m.event_id LIKE 'dream:%'"
    _WIKI_SELECT = f"""
        SELECT m.memory_id, m.content, m.source, m.timestamp, m.status,
               m.supersedes_id, m.superseded_by_id, w.title, w.summary, w.tags
        FROM memory_records m LEFT JOIN dream_wiki w ON w.memory_id = m.memory_id
        WHERE {_WIKI_SCOPE}
    """

    @staticmethod
    def _wiki_note(row):
        facts = list(_episodes({"id": row["memory_id"], "text": row["content"], "source": row["source"]}))[:4]
        first_line = next((line.strip().lstrip("# ").strip() for line in row["content"].splitlines() if line.strip()), "")
        title = facts[0]["key"] if facts else first_line
        tags = json.loads(row["tags"]) if row["tags"] else []
        for fact in facts:
            tags.extend([fact["category"], fact["key"].casefold()])
        return {
            "id": row["memory_id"], "title": row["title"] or title[:80] or "Untitled note",
            "summary": row["summary"] or " ".join(row["content"].split())[:280],
            "tags": list(dict.fromkeys(tags)), "category": facts[0]["category"] if facts else "note",
            "organized_by": "Gemini" if row["title"] else "local",
            **{name: row[name] for name in ("content", "source", "timestamp", "status",
                                            "supersedes_id", "superseded_by_id")},
        }

    def wiki(self):
        return self.wiki_snapshot()[0]

    def wiki_snapshot(self):
        with self.store._connect() as db:
            db.execute("BEGIN")
            rows = db.execute(f"{self._WIKI_SELECT} ORDER BY m.seq DESC LIMIT 100", self.store._scope).fetchall()
            pending = db.execute("SELECT count(*) FROM dream_capture WHERE promoted = 0").fetchone()[0]
        return [self._wiki_note(row) for row in rows], pending

    def find_wiki_notes(self, query):
        """Find up to three active wiki notes by a named topic, with exact source receipts."""
        words = re.findall(r"\w+", query.casefold())
        if not words:
            return []
        phrase = " ".join(words)
        with self.store._connect() as db:
            rows = db.execute(
                f"{self._WIKI_SELECT} ORDER BY m.seq DESC LIMIT 100", self.store._scope,
            ).fetchall()
        matches = []
        for row in rows:
            if row["status"] != "active":
                continue
            note = self._wiki_note(row)
            fields = (note["title"], note["source"], note["content"], *note["tags"])
            haystack = " ".join(re.findall(r"\w+", " ".join(fields).casefold()))
            if f" {phrase} " in f" {haystack} ":
                matches.append(note)
                if len(matches) == 3:
                    break
        return matches

    def memory_inventory(self):
        with self.store._connect() as db:
            db.execute("BEGIN")
            counts = {row["status"]: row["total"] for row in db.execute(
                f"SELECT m.status, count(*) AS total FROM memory_records m WHERE {self._WIKI_SCOPE} GROUP BY m.status",
                self.store._scope,
            )}
            pending = db.execute("SELECT count(*) FROM dream_capture WHERE promoted = 0").fetchone()[0]
            rows = db.execute(
                f"{self._WIKI_SELECT} AND m.status = 'active' ORDER BY m.seq DESC LIMIT 8", self.store._scope,
            ).fetchall()
        return {"active_count": counts.get("active", 0), "superseded_count": counts.get("superseded", 0),
                "pending_count": pending, "notes": [self._wiki_note(row) for row in rows]}

    def recall(self, question, limit=3):
        query = question.strip().lower().rstrip("?!. ")
        if not 1 <= limit <= 100:
            raise ValueError("Recall limit must be between 1 and 100.")
        profile = re.fullmatch(r"what (?:do you know|have you learned|do you remember) about me", query)
        experiments = re.fullmatch(
            r"(?:what|which) (?:experiments|expriemnts) (?:did we do|have we done|are we doing|did we run|have we run)",
            query,
        )
        if not (profile or experiments):
            return self.store.search(question[:2048], limit=limit)
        categories = {"profile", "preference"} if profile else {"experiment", "process"}
        with self.store._connect() as db:
            records = db.execute(
                "SELECT m.* FROM memory_records m WHERE m.user_id = ? AND m.workspace_id = ? "
                "AND m.status = 'active' AND m.event_id LIKE 'dream:%' ORDER BY m.seq DESC LIMIT 100",
                self.store._scope,
            ).fetchall()
        matches = []
        for row in records:
            topics = {item["category"] for item in _episodes(
                {"id": row["memory_id"], "text": row["content"], "source": row["source"]},
            )}
            if topics & categories:
                matches.append(_record(row))
                if len(matches) == limit:
                    break
        return matches
