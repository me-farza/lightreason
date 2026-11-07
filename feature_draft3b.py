"""Auxiliary-only, target-specific EAGLE3-style feature proposer.

Adapted architecture/three-step recurrence from SafeAILab/EAGLE commit
cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b, Apache-2.0:
https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/traineagle3/cnets.py

This is a new auxiliary model, never an authoritative replacement. Original
target Parameters are neither shared nor optimized. Direct aux_to_target IDs
are deliberately NOT EAGLE's native d2t offset representation.

Offline records are unshifted: input_ids[S], loss_mask[S], target residual taps
aux_hidden_state[S,6144] (HF[2,18,33]), normalized hidden_state[S,2048], and
teacher_next_ids[S] (full-target argmax at t predicts t+1; last is -100).
features[t] + embedding(ids[t+1]) predicts the state at t+1 and next-ID at t+2.
Training propagates its own state for three steps, with a separate hypothetical
branch for each sequence row. Later branch KV is visible only to that row.

Train with FP32 auxiliary masters inside BF16 autocast; infer after converting
the AUXILIARY ONLY to BF16, or under autocast. The frozen target stays unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = nn = F = None

_Module = object if nn is None else nn.Module
OFFICIAL_SOURCE_COMMIT = "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"
TARGET_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
TARGET_TAPS = (2, 18, 33)
FEATURE_SCHEMA = "lightreason_feature_draft3b_unshifted_v1"


@dataclass(frozen=True)
class FeatureDraftConfig:
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    draft_vocab_size: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    initializer_range: float = 0.02
    target_revision: str = TARGET_REVISION
    target_taps: tuple = TARGET_TAPS

    def validate(self):
        if (min(self.hidden_size, self.intermediate_size, self.num_attention_heads,
                self.num_key_value_heads, self.vocab_size, self.draft_vocab_size) < 1 or
                self.hidden_size % self.num_attention_heads or
                self.num_attention_heads % self.num_key_value_heads or
                self.head_dim % 2 or self.draft_vocab_size > self.vocab_size or
                self.rms_norm_eps <= 0 or self.rope_theta <= 1):
            raise ValueError("Invalid auxiliary feature-draft geometry")
        if tuple(self.target_taps) != TARGET_TAPS:
            raise ValueError("This feature contract requires HF taps2,18,33")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def projection_shapes(self):
        h, kv = self.hidden_size, self.num_key_value_heads * self.head_dim
        return {"fc": (h, 3*h), "q": (h, 2*h), "k": (kv, 2*h),
                "v": (kv, 2*h), "o": (h, h),
                "gate": (self.intermediate_size, h),
                "up": (self.intermediate_size, h),
                "down": (h, self.intermediate_size),
                "head": (self.draft_vocab_size, h)}


def alignment_indices(sequence_length, step):
    """Valid (initial feature row, conditioning position, predicted-ID position)."""
    if sequence_length < 2 or step < 0:
        raise ValueError("Use a sequence length>=2 and nonnegative step")
    return [(r, r+step+1, r+step+2)
            for r in range(max(0, sequence_length-step-2))]


def hypothesis_allowed_keys(rows, step, row):
    """Pure-Python attention oracle: base prefix plus same-row branch keys."""
    if rows < 1 or step < 0 or not 0 <= row < rows:
        raise ValueError("Invalid hypothesis mask coordinates")
    return list(range(row+1)) + [block*rows+row for block in range(1, step+1)]


def _storage_key(parameter):
    return str(parameter.device), parameter.untyped_storage().data_ptr()


def assert_optimizer_excludes_target(auxiliary, optimizer, target_parameters):
    """Reject original identities/storage, frozen tensors or omitted trainables."""
    originals = list(target_parameters)
    forbidden_ids = {id(p) for p in originals}
    forbidden_storage = {_storage_key(p) for p in originals}
    wanted = {id(p) for p in auxiliary.parameters() if p.requires_grad}
    actual = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if (id(parameter) in forbidden_ids or _storage_key(parameter) in forbidden_storage or
                    not parameter.requires_grad):
                raise RuntimeError("Optimizer includes an original target alias or frozen parameter")
            actual.append(id(parameter))
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise RuntimeError("Optimizer must contain every auxiliary trainable exactly once")
    return {"original_target_parameters_and_storage_excluded": True,
            "frozen_auxiliary_parameters_excluded": True,
            "auxiliary_trainable_tensor_count": len(actual)}


class FeatureRMSNorm(_Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return value.to(x.dtype) * self.weight.to(x.dtype)


def _repeat_kv(value, groups):
    return value if groups == 1 else value.repeat_interleave(groups, dim=1)


class FeatureDraft3B(_Module):
    auxiliary_only = True
    authoritative = False

    def __init__(self, config, embedding_weight, head_rows, aux_to_target):
        if torch is None:
            raise RuntimeError("PyTorch is required for tensor/model operations")
        super().__init__()
        config.validate()
        h = config.hidden_size
        ids = aux_to_target.detach().to(device="cpu", dtype=torch.long).contiguous()
        if (ids.shape != (config.draft_vocab_size,) or ids.unique().numel() != ids.numel() or
                ids.min().item() < 0 or ids.max().item() >= config.vocab_size or
                tuple(embedding_weight.shape) != (config.vocab_size, h) or
                tuple(head_rows.shape) != (config.draft_vocab_size, h)):
            raise ValueError("Embedding/head/direct-ID map differ from auxiliary config")
        self.config = config
        # Independent frozen copies; these are never aliases of target storage.
        self.embed_tokens = nn.Embedding.from_pretrained(
            embedding_weight.detach().to(device="cpu", dtype=torch.bfloat16).clone(), freeze=True)
        self.register_buffer("teacher_head_weight", head_rows.detach().to(
            device="cpu", dtype=torch.bfloat16).clone(), persistent=False)
        self.register_buffer("aux_to_target", ids.clone())
        reverse = torch.full((config.vocab_size,), -1, dtype=torch.long)
        reverse[ids] = torch.arange(config.draft_vocab_size)
        self.register_buffer("target_to_aux", reverse)
        self.fc = nn.Linear(3*h, h, bias=False, dtype=torch.float32)
        self.hidden_norm = FeatureRMSNorm(h, config.rms_norm_eps)
        self.input_norm = FeatureRMSNorm(h, config.rms_norm_eps)
        kv = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(2*h, h, bias=False, dtype=torch.float32)
        self.k_proj = nn.Linear(2*h, kv, bias=False, dtype=torch.float32)
        self.v_proj = nn.Linear(2*h, kv, bias=False, dtype=torch.float32)
        self.o_proj = nn.Linear(h, h, bias=False, dtype=torch.float32)
        self.post_attention_norm = FeatureRMSNorm(h, config.rms_norm_eps)
        self.gate_proj = nn.Linear(h, config.intermediate_size, bias=False, dtype=torch.float32)
        self.up_proj = nn.Linear(h, config.intermediate_size, bias=False, dtype=torch.float32)
        self.down_proj = nn.Linear(config.intermediate_size, h, bias=False, dtype=torch.float32)
        self.norm = FeatureRMSNorm(h, config.rms_norm_eps)
        self.lm_head = nn.Linear(h, config.draft_vocab_size, bias=False, dtype=torch.float32)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
        with torch.no_grad():
            self.lm_head.weight.copy_(head_rows.detach().cpu().float())
        inv = 1.0 / (config.rope_theta ** (
            torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
        self.register_buffer("rope_inv_freq", inv, persistent=False)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Casting the auxiliary to BF16 for serving must not permanently round
        # the rotary frequency table. Rebuild its original FP32 CPU constants.
        inv = 1.0 / (self.config.rope_theta ** (
            torch.arange(0, self.config.head_dim, 2, dtype=torch.float32) / self.config.head_dim))
        self.rope_inv_freq = inv.to(device=self.embed_tokens.weight.device)
        return result

    @classmethod
    def from_target(cls, target, aux_to_target):
        c = target.config
        if (target.training or any(p.requires_grad or p.dtype != torch.bfloat16 for p in target.parameters()) or
                c.hidden_size != 2048 or c.intermediate_size != 11008 or c.num_hidden_layers != 36 or
                c.num_attention_heads != 16 or c.num_key_value_heads != 2 or c.vocab_size != 151936 or
                c.rms_norm_eps != 1e-6 or c.rope_theta != 1e6 or c.model_type != "qwen2" or
                target.lm_head.weight is not target.get_input_embeddings().weight):
            raise ValueError("Use the original frozen/eval all-BF16 Qwen2.5-3B target")
        ids = aux_to_target.detach().to(device=target.lm_head.weight.device, dtype=torch.long)
        config = FeatureDraftConfig(hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
            num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
            vocab_size=c.vocab_size, draft_vocab_size=ids.numel(), rms_norm_eps=c.rms_norm_eps,
            rope_theta=c.rope_theta)
        result = cls(config, target.get_input_embeddings().weight,
                     target.lm_head.weight.detach().index_select(0, ids), ids)
        target_ids = {id(p) for p in target.parameters()}
        target_storage = {_storage_key(p) for p in target.parameters()}
        if any(id(p) in target_ids or _storage_key(p) in target_storage for p in result.parameters()):
            raise RuntimeError("Auxiliary construction aliased original target parameters")
        return result

    def _qkv(self, state, input_ids, position_ids):
        b, length, _ = state.shape
        h, nk, d = self.config.num_attention_heads, self.config.num_key_value_heads, self.config.head_dim
        emb = self.embed_tokens(input_ids).to(state.dtype)
        value = torch.cat((self.input_norm(emb), self.hidden_norm(state)), dim=-1)
        q = self.q_proj(value).view(b, length, h, d).transpose(1, 2)
        k = self.k_proj(value).view(b, length, nk, d).transpose(1, 2)
        v = self.v_proj(value).view(b, length, nk, d).transpose(1, 2)
        freq = position_ids.float().unsqueeze(-1) * self.rope_inv_freq.float()
        angles = torch.cat((freq, freq), dim=-1).unsqueeze(1)
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        rotate = lambda x: torch.cat((-x[..., d//2:], x[..., :d//2]), dim=-1)
        return q*cos + rotate(q)*sin, k*cos + rotate(k)*sin, v

    def _finish(self, state, attention):
        b, length, _ = state.shape
        attention = attention.transpose(1, 2).contiguous().view(b, length, self.config.hidden_size)
        state = state + self.o_proj(attention)
        normalized = self.post_attention_norm(state)
        return state + self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))

    def project_target_features(self, features):
        if features.shape[-1] != 3*self.config.hidden_size:
            raise ValueError("Expected concatenated three-tap target features")
        return self.fc(features)

    def compute_logits(self, state):
        return self.lm_head(self.norm(state))

    def forward(self, feature_state, input_ids, position_ids, past_key_values=None, attention_mask=None):
        """Runtime teacher refill or speculative step: (logits,state,(K,V)).

        feature_state width3H projects verified target features, while H reuses
        predicted auxiliary state. Caller owns verified-feature cache refill and
        must discard rejected hypotheses. No target inference or final emission.
        """
        state = (self.project_target_features(feature_state)
                 if feature_state.shape[-1] == 3*self.config.hidden_size else feature_state)
        if state.shape[-1] != self.config.hidden_size:
            raise ValueError("Use target3H features or predicted H state")
        q, k, v = self._qkv(state, input_ids, position_ids)
        past = 0 if past_key_values is None else past_key_values[0].shape[2]
        if past_key_values is not None:
            k = torch.cat((past_key_values[0], k), dim=2)
            v = torch.cat((past_key_values[1], v), dim=2)
        if attention_mask is None:
            rows = torch.arange(state.shape[1], device=state.device) + past
            keys = torch.arange(k.shape[2], device=state.device)
            attention_mask = (keys[None, :] <= rows[:, None])[None, None]
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        out = F.scaled_dot_product_attention(q, _repeat_kv(k, groups), _repeat_kv(v, groups),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        state = self._finish(state, out)
        return self.compute_logits(state), state, (k, v)

    def hypothesis_step(self, state, input_ids, position_ids, blocks=()):
        """Differentiable base-causal + independent diagonal-branch SDPA."""
        q, k, v = self._qkv(state, input_ids, position_ids)
        blocks = tuple(blocks) + ((k, v),)
        length = state.shape[1]
        base = torch.ones((length, length), dtype=torch.bool, device=state.device).tril()
        diagonal = torch.eye(length, dtype=torch.bool, device=state.device)
        mask = torch.cat([base] + [diagonal] * (len(blocks)-1), dim=-1)[None, None]
        all_k = torch.cat([pair[0] for pair in blocks], dim=2)
        all_v = torch.cat([pair[1] for pair in blocks], dim=2)
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        attended = F.scaled_dot_product_attention(q, _repeat_kv(all_k, groups),
            _repeat_kv(all_v, groups), attn_mask=mask, dropout_p=0.0, is_causal=False)
        return self._finish(state, attended), blocks

    def unroll_loss(self, record, steps=3):
        """Three-step restricted teacher CE; reports full-target-ID agreement.

        Never teacher-force recurrent features after step0. Tokens are shifted
        original corpus tokens as in official EAGLE3 training. Loss excludes rows
        whose full-target greedy ID lies outside the auxiliary vocabulary.
        """
        # Collector records may also contain string provenance metadata.
        record = {name: record[name] for name in ("input_ids", "loss_mask",
            "aux_hidden_state", "hidden_state", "teacher_next_ids")}
        ids, feature = record["input_ids"], record["aux_hidden_state"]
        if ids.ndim == 1:
            record = {name: value.unsqueeze(0) for name, value in record.items()}
            ids, feature = record["input_ids"], record["aux_hidden_state"]
        if ids.shape[0] != 1 or ids.shape[1] < steps+2 or steps < 1:
            raise ValueError("Use batch1 and sequence length>=steps+2")
        length = ids.shape[1]-1
        state = self.project_target_features(feature[:, :-1].detach())
        with torch.no_grad():
            teacher_logits = F.linear(record["hidden_state"][:, 1:].detach(), self.teacher_head_weight)
            probabilities = teacher_logits.float().softmax(-1)
        blocks, total, metrics = (), None, []
        for step in range(steps):
            # Same leading row r describes a branch rooted at original positionr.
            end = length-step
            tokens = F.pad(ids[:, step+1:], (0, step))
            position = torch.arange(length, device=ids.device)[None] + step + 1
            state, blocks = self.hypothesis_step(state, tokens, position, blocks)
            logits = self.compute_logits(state).float()
            target_p = F.pad(probabilities[:, step:], (0, 0, 0, step))
            teacher = F.pad(record["teacher_next_ids"][:, step+1:], (0, step), value=-100)
            mask = F.pad(record["loss_mask"][:, step+1:].bool(), (0, step), value=False)
            valid = mask & (teacher >= 0)
            mapped = self.target_to_aux[teacher.clamp_min(0)]
            eligible = valid & (mapped >= 0)
            losses = -(target_p * logits.log_softmax(-1)).sum(-1)
            denominator = eligible.sum()
            loss = (losses * eligible).sum() / denominator.clamp_min(1)
            total = (0.8**step)*loss if total is None else total + (0.8**step)*loss
            predicted_ids = self.aux_to_target[logits.argmax(-1)]
            metrics.append({"step": step, "loss": loss.detach(),
                "valid_rows": valid.sum().detach(), "eligible_rows": denominator.detach(),
                "correct_full_target_ids": ((predicted_ids == teacher) & valid).sum().detach(),
                "end_valid_conditioning_exclusive": end})
        return total, metrics

    def trainable_state_dict(self):
        return {name: parameter.detach().cpu().clone()
                for name, parameter in self.named_parameters() if parameter.requires_grad}

    def load_trainable_state_dict(self, values):
        parameters = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if values.keys() != parameters.keys():
            raise ValueError("Auxiliary checkpoint trainable names changed")
        with torch.no_grad():
            for name, parameter in parameters.items():
                if values[name].shape != parameter.shape:
                    raise ValueError("Auxiliary checkpoint shape changed: " + name)
                parameter.copy_(values[name])

    def metadata(self):
        return {"role": "auxiliary_feature_proposer_never_authoritative",
            "schema": FEATURE_SCHEMA, "config": asdict(self.config),
            "official_architecture_source_commit": OFFICIAL_SOURCE_COMMIT,
            "mapping": "aux_to_target[j] is DIRECT original target ID; no offset",
            "target_feature_taps_HF_tuple_indices": list(TARGET_TAPS),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen_embedding_parameters": self.embed_tokens.weight.numel(),
            "teacher_head_is_separate_frozen_original_rows": True,
            "training_attention": "causal base prefix plus same-row diagonal hypothesis keys",
            "quality_acceptance_and_speed_verified": False}
