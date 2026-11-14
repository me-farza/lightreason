"""Run private memory-vector prefill on the audited 35B FreeToken stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from threading import RLock
from uuid import uuid4

from demo_chat.messages import PromptText

from .latent_freetoken_hook import HIDDEN_SIZE, INGRESS_ENV, MARKER, MARKER_ID, MAX_ROWS, embedding_digest
from .remote import Reasoner35B

PLACEHOLDER = "[[MEMORY]]"
_MODEL = Path("freetoken/models/qwen3_5_moe/model.py")
_ANCHOR = "        x = embed_input_ids(self.embed_tokens, input_ids, batch)\n"
_HOOK_IMPORT = "from lightreason_latent_ingress import inject_memory_embeddings\n"
_HOOK = "        x = inject_memory_embeddings(x, batch)\n"


class LatentReasoner35B(Reasoner35B):
    """Qwen35B consumes real embedding rows in place of private sentinel positions."""

    def __init__(self, model_path, stage, *, worker_logs=True):
        super().__init__(model_path, stage, worker_logs=worker_logs)
        self._latent_lock = RLock()
        self._latent_failed = False

    @property
    def _ingress(self):
        return Path(self.control.name) / "latent"

    def _start_monitored(self, command, **options):
        env = options["env"].copy()
        original = self.expert_cache_dir / "python-tune"
        copy = self._ingress / "python-tune"
        shutil.copytree(original, copy, ignore=shutil.ignore_patterns("__pycache__"))
        self._patch_embedding(copy)
        paths = env["PYTHONPATH"].split(os.pathsep)
        if paths.count(str(original)) != 1:
            raise RuntimeError("Could not identify the staged FreeToken runtime.")
        env["PYTHONPATH"] = os.pathsep.join(
            str(copy) if entry == str(original) else entry for entry in paths
        )
        env[INGRESS_ENV] = str(self._ingress)
        self._latent_failed = False
        return super()._start_monitored(command, **{**options, "env": env})

    @staticmethod
    def _patch_embedding(root):
        path = root / _MODEL
        source = path.read_text()
        if source.count(_ANCHOR) != 1:
            raise RuntimeError("The Qwen35B embedding hook does not match the staged source.")
        imports = "import torch\n"
        if source.count(imports) != 1:
            raise RuntimeError("The Qwen35B import hook does not match the staged source.")
        path.write_text(source.replace(imports, imports + _HOOK_IMPORT).replace(_ANCHOR, _ANCHOR + _HOOK))
        shutil.copyfile(Path(__file__).with_name("latent_freetoken_hook.py"), root / "lightreason_latent_ingress.py")

    def answer(self, prompt, token_budget):
        with self._latent_lock:
            if self._latent_failed:
                raise RuntimeError("Restart the reasoner after the failed memory input.")
            system = getattr(prompt, "system_prompt", None) or ""
            if any(marker in text for marker in (MARKER, PLACEHOLDER) for text in (str(prompt), system)):
                raise ValueError("Reserved memory marker requires a vector payload.")
            return super().answer(prompt, token_budget)

    def answer_with_vectors(self, prompt, token_budget, embeddings):
        """Pass tensors through the one-shot prefill hook beside the original question."""
        import torch
        from safetensors.torch import save_file

        with self._latent_lock:
            if self._latent_failed:
                raise RuntimeError("Restart the reasoner after the failed memory input.")
            if self.control is None:
                raise RuntimeError("Start the reasoner before submitting memory inputs.")
            rows = torch.as_tensor(embeddings, device="cpu").to(torch.bfloat16).contiguous()
            if (rows.ndim != 2 or rows.shape[1] != HIDDEN_SIZE or not 1 <= len(rows) <= MAX_ROWS
                    or not torch.isfinite(rows).all()):
                raise ValueError(f"Provide 1–{MAX_ROWS} finite rows of width {HIDDEN_SIZE}.")
            system = getattr(prompt, "system_prompt", None)
            if (str(prompt).count(PLACEHOLDER) != 1 or MARKER in str(prompt)
                    or any(marker in (system or "") for marker in (MARKER, PLACEHOLDER))):
                raise ValueError("The prompt requires exactly one [[MEMORY]] placeholder.")
            if self.tokenizer.encode(MARKER, add_special_tokens=False) != [MARKER_ID]:
                raise RuntimeError("Unexpected Qwen35B marker token.")
            rendered = PromptText(str(prompt).replace(PLACEHOLDER, MARKER * len(rows)), system)
            ids = self._input_ids(rendered)
            positions = [i for i, token in enumerate(ids) if token == MARKER_ID]
            if len(positions) != len(rows) or positions != list(range(positions[0], positions[0] + len(rows))):
                raise RuntimeError("Qwen35B did not render one contiguous placeholder per memory row.")
            nonce = uuid4().hex
            digest = embedding_digest(rows)
            pending = self._ingress / "pending.safetensors"
            receipt = self._ingress / "injected.json"
            staged = self._ingress / "pending.part"
            if any(path.exists() for path in (pending, staged, receipt)):
                raise RuntimeError("A previous memory request was not cleared.")
            try:
                save_file({"embeddings": rows}, staged, metadata={
                    "request_id": nonce, "prompt_ids": json.dumps(ids, separators=(",", ":")),
                })
                staged.replace(pending)
                answer = super().answer(rendered, token_budget)
                if not receipt.exists():
                    raise RuntimeError("No prefill receipt: failed to inject Lightreason memory.")
                audit = json.loads(receipt.read_text())
                if (pending.exists() or audit["request_id"] != nonce or audit["positions"] != positions
                        or audit["sha256"] != digest):
                    raise RuntimeError("FreeToken did not verify all requested memory embedding rows.")
            except BaseException:
                self._latent_failed = True
                raise
            finally:
                for path in (pending, receipt, staged, receipt.with_suffix(".part")):
                    path.unlink(missing_ok=True)
            return {
                **answer, "injected_rows": len(positions), "injection_positions": positions,
                "memory_embedding_sha256": digest,
            }
