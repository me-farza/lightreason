"""Experimental bounded search for an unchanged, bias-free BF16 output head.

The DP4A index only eliminates impossible raw-greedy winners. Every retained
row is reread from the original BF16 weight and evaluated with a configurable
FP32 FMA/lane reduction. The output has -inf for eliminated rows, so it is not
a replacement for probabilities, sampling, logprobs, or logit processors.

The numerical contract is CONDITIONAL until the actual 3B cuBLAS dispatch has
been profiled and its full-row storage bits matched. A successful lane8 audit
on the 0.5B model does not establish the 3B reduction order. Lane counts8/16/32
are probes, not assertions. Input/weight guards select all rows for nonfinite
or extreme operands; the verifier's nonfinite parity remains conditional.

For finite guarded operands, an exact INT32 index dot supplies a center and
Cauchy-Schwarz bounds the index error. gamma_K covers the FP32 dot reduction;
an explicit additive allowance covers FTZ/underflow. FP64 norm estimates use
a K-dependent conservative relative envelope and upward inflation. Bound
arithmetic uses directed FP64 rounding and outward BF16 endpoints. Equality
is retained to preserve first-index argmax ties. Weight bits are never changed.

Sources: NVIDIA PTX floating-point/dp4a semantics, and the standard gamma_n
forward-error bound. This is an audited prototype, not a universal proof of
cuBLAS arithmetic or a claim of end-to-end performance.
https://docs.nvidia.com/cuda/parallel-thread-execution/
https://www.tuhh.de/ti3/paper/rump/Ru11a.pdf
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def numerical_contract(k):
    """CPU-only constants; every positive error allowance is rounded upward."""
    if not isinstance(k, int) or k < 4 or k % 4:
        raise ValueError("DP4A8 requires a positive K divisible by4")
    if k * 127 ** 2 > 2 ** 31 - 1:
        raise ValueError("The auxiliary dot could overflow signed INT32")
    fp32_envelope = k * 2.0 ** -24
    fp64_envelope = (10 * k + 32) * 2.0 ** -53
    if fp32_envelope >= 1.0 or fp64_envelope >= 1.0:
        raise ValueError("Invalid floating-point error envelope")
    # This also dominates the FP64 setup/query norm envelope for K=2048.
    inflation = max(1.0 + 2.0 ** -32, 1.0 / (1.0 - fp64_envelope))
    return {
        "K": k,
        "gamma_k_up": math.nextafter(fp32_envelope / (1.0 - fp32_envelope), math.inf),
        "norm_inflate_up": math.nextafter(inflation, math.inf),
        "fp64_relative_envelope": fp64_envelope,
        "underflow_up": math.nextafter(4.0 * k * 2.0 ** -126 * (1.0 + 2.0 ** 41), math.inf),
        "int32_absolute_dot_bound": k * 127 ** 2,
        "operand_abs_guard": 2.0 ** 40,
        "conditional_only": True,
    }


@triton.jit
def _up_add(a, b):
    return tl.inline_asm_elementwise("add.rp.f64 $0, $1, $2;", "=d,d,d", [a, b],
                                    dtype=tl.float64, is_pure=True, pack=1)


@triton.jit
def _down_add(a, b):
    return tl.inline_asm_elementwise("add.rm.f64 $0, $1, $2;", "=d,d,d", [a, b],
                                    dtype=tl.float64, is_pure=True, pack=1)


@triton.jit
def _up_mul(a, b):
    return tl.inline_asm_elementwise("mul.rp.f64 $0, $1, $2;", "=d,d,d", [a, b],
                                    dtype=tl.float64, is_pure=True, pack=1)


@triton.jit
def _down_mul(a, b):
    return tl.inline_asm_elementwise("mul.rm.f64 $0, $1, $2;", "=d,d,d", [a, b],
                                    dtype=tl.float64, is_pure=True, pack=1)


@triton.jit
def _bf16_outward(value, UP: tl.constexpr):
    if UP:
        f32 = tl.inline_asm_elementwise("cvt.rp.f32.f64 $0, $1;", "=f,d", [value],
                                       dtype=tl.float32, is_pure=True, pack=1)
    else:
        f32 = tl.inline_asm_elementwise("cvt.rm.f32.f64 $0, $1;", "=f,d", [value],
                                       dtype=tl.float32, is_pure=True, pack=1)
    bits = f32.to(tl.uint32, bitcast=True)
    negative = (bits & 0x80000000) != 0
    increment = (~negative if UP else negative) & ((bits & 0xffff) != 0)
    rounded = ((bits >> 16) + increment.to(tl.uint32)).to(tl.uint16)
    return rounded.to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _query_index(X, Quant, Words, Even, Odd, Stats, Bad, WeightBad,
                 K: tl.constexpr, BLOCK: tl.constexpr, INFLATE: tl.constexpr,
                 BITS: tl.constexpr, WORD_BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    x = tl.load(X + index, index < K, other=0).to(tl.float32)
    finite = tl.abs(x) < float("inf")
    maximum = tl.max(tl.where(finite, tl.abs(x), 0.0), 0)
    invalid = tl.sum((~finite & (index < K)).to(tl.int32), 0) > 0
    invalid = invalid | (maximum > 2.0 ** 40) | ((maximum != 0) & (maximum < 2.0 ** -40))
    invalid = invalid | (tl.load(WeightBad) != 0)
    scale = tl.where((maximum != 0) & ~invalid, maximum / 127.0, 1.0)
    safe_x = tl.where(invalid, 0.0, x)
    quant = tl.minimum(127.0, tl.maximum(-127.0, tl.floor(safe_x / scale + 0.5))).to(tl.int8)
    tl.store(Quant + index, quant, index < K)
    unsigned = quant.to(tl.int32) & 255
    word = tl.arange(0, WORD_BLOCK)
    if BITS == 8:
        packed = tl.full((WORD_BLOCK,), 0, tl.uint32)
        for byte in tl.static_range(4):
            packed = packed | (tl.gather(unsigned, word * 4 + byte, 0).to(tl.uint32) << (8 * byte))
        tl.store(Words + word, packed.to(tl.int32, bitcast=True), word < K // 4)
    else:
        even = tl.full((WORD_BLOCK,), 0, tl.uint32)
        odd = tl.full((WORD_BLOCK,), 0, tl.uint32)
        for byte in tl.static_range(4):
            even = even | (tl.gather(unsigned, word * 8 + 2 * byte, 0).to(tl.uint32) << (8 * byte))
            odd = odd | (tl.gather(unsigned, word * 8 + 2 * byte + 1, 0).to(tl.uint32) << (8 * byte))
        tl.store(Even + word, even.to(tl.int32, bitcast=True), word < K // 8)
        tl.store(Odd + word, odd.to(tl.int32, bitcast=True), word < K // 8)
    xd = safe_x.to(tl.float64)
    dequant = quant.to(tl.float64) * scale.to(tl.float64)
    error = xd - dequant
    inflate64 = tl.full((), INFLATE, tl.float64)
    tl.store(Stats, scale.to(tl.float64))
    tl.store(Stats + 1, libdevice.sqrt(tl.sum(xd * xd, 0)) * inflate64)
    tl.store(Stats + 2, libdevice.sqrt(tl.sum(error * error, 0)) * inflate64)
    tl.store(Bad, invalid.to(tl.int32))


@triton.jit
def _dp4a(a, b, accumulator):
    return tl.inline_asm_elementwise("dp4a.s32.s32 $0, $1, $2, $3;", "=r,r,r,r",
                                    [a, b, accumulator], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _extend_nibbles(codes):
    signed4 = codes ^ 0x08080808
    return (signed4 | ((signed4 & 0x08080808) * 30)).to(tl.int32, bitcast=True)


@triton.jit
def _screen(Words, QueryWords, QueryEven, QueryOdd, Scores,
            N: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, 32)[None, :]
    accumulator = tl.full((ROWS, 32), 0, tl.int32)
    WORDS: tl.constexpr = K // (4 if BITS == 8 else 8)
    for iteration in tl.static_range((WORDS + 31) // 32):
        word = tl.broadcast_to(lane + iteration * 32, (ROWS, 32))
        valid = (rows[:, None] < N) & (word < WORDS)
        packed = tl.load(Words + rows[:, None] * WORDS + word, valid, other=0)
        if BITS == 8:
            query = tl.load(QueryWords + word, word < WORDS, other=0)
            accumulator = _dp4a(packed, query, accumulator)
        else:
            value = packed.to(tl.uint32, bitcast=True)
            even = _extend_nibbles(value & 0x0f0f0f0f)
            odd = _extend_nibbles((value >> 4) & 0x0f0f0f0f)
            qe = tl.load(QueryEven + word, word < WORDS, other=0)
            qo = tl.load(QueryOdd + word, word < WORDS, other=0)
            accumulator = _dp4a(even, qe, accumulator)
            accumulator = _dp4a(odd, qo, accumulator)
    tl.store(Scores + rows, tl.sum(accumulator, 1), rows < N)


@triton.jit
def _bounds(Scores, WeightScale, WeightNorm, QuantNorm, ErrorNorm, QueryStats,
            Bad, Lower, Upper, N: tl.constexpr, BLOCK: tl.constexpr,
            GAMMA: tl.constexpr, UNDERFLOW: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < N
    if tl.load(Bad) != 0:
        lower = tl.full((BLOCK,), -float("inf"), tl.float32)
        upper = tl.full((BLOCK,), float("inf"), tl.float32)
    else:
        integer = tl.load(Scores + rows, valid, other=0).to(tl.float64)
        sw = tl.load(WeightScale + rows, valid, other=0)
        nw = tl.load(WeightNorm + rows, valid, other=0)
        nq = tl.load(QuantNorm + rows, valid, other=0)
        ew = tl.load(ErrorNorm + rows, valid, other=0)
        sx, nx, ex = tl.load(QueryStats), tl.load(QueryStats + 1), tl.load(QueryStats + 2)
        scale_lo, scale_hi = _down_mul(sw, sx), _up_mul(sw, sx)
        center_lo = _down_mul(tl.where(integer >= 0, scale_lo, scale_hi), integer)
        center_hi = _up_mul(tl.where(integer >= 0, scale_hi, scale_lo), integer)
        error = _up_add(_up_mul(ew, nx), _up_mul(nq, ex))
        fp32_error = _up_mul(tl.full((BLOCK,), GAMMA, tl.float64), _up_mul(nw, nx))
        error = _up_add(_up_add(error, fp32_error), tl.full((BLOCK,), UNDERFLOW, tl.float64))
        lower = _bf16_outward(_down_add(center_lo, -error), False)
        upper = _bf16_outward(_up_add(center_hi, error), True)
    tl.store(Lower + rows, lower, valid)
    tl.store(Upper + rows, upper, valid)


@triton.jit
def _fold(X, W, accumulator, rows, valid_rows, base,
          K: tl.constexpr, LANES: tl.constexpr, ROWS: tl.constexpr,
          ACTIVE: tl.constexpr):
    lane = tl.arange(0, LANES)[None, :]
    xs = ()
    ws = ()
    for item in tl.static_range(ACTIVE):
        offset = tl.broadcast_to(lane + (base + item) * LANES, (ROWS, LANES))
        valid = valid_rows[:, None] & (offset < K)
        xs += (tl.load(X + offset, valid, other=0).to(tl.float32),)
        ws += (tl.load(W + rows[:, None] * K + offset, valid, other=0).to(tl.float32),)
    for item in tl.static_range(ACTIVE):
        accumulator = tl.fma(xs[item], ws[item], accumulator)
    return accumulator


@triton.jit
def _evaluate(X, W, rows, valid_rows, K: tl.constexpr, LANES: tl.constexpr,
              LOG_LANES: tl.constexpr, ROWS: tl.constexpr, TILE: tl.constexpr):
    accumulator = tl.full((ROWS, LANES), 0.0, tl.float32)
    ITEMS: tl.constexpr = K // LANES
    for block in range(ITEMS // TILE):
        accumulator = _fold(X, W, accumulator, rows, valid_rows,
                            block * TILE, K, LANES, ROWS, TILE)
    if ITEMS % TILE:
        accumulator = _fold(X, W, accumulator, rows, valid_rows,
                            (ITEMS // TILE) * TILE, K, LANES, ROWS, ITEMS % TILE)
    lane = tl.arange(0, LANES)[None, :]
    for stage in tl.static_range(LOG_LANES):
        distance = LANES >> (stage + 1)
        partner = tl.broadcast_to(lane ^ distance, (ROWS, LANES))
        accumulator = accumulator + tl.gather(accumulator, partner, 1)
    return tl.gather(accumulator, tl.full((ROWS, 1), 0, tl.int32), 1).reshape((ROWS,))


@triton.jit
def _verify_full(X, W, Output, N: tl.constexpr, K: tl.constexpr,
                 LANES: tl.constexpr, LOG_LANES: tl.constexpr,
                 ROWS: tl.constexpr, TILE: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    result = _evaluate(X, W, rows, rows < N, K, LANES, LOG_LANES, ROWS, TILE)
    tl.store(Output + rows, result.to(tl.bfloat16), rows < N)


@triton.jit
def _verify_selected(X, W, Upper, MaxLower, Selected, Output,
                     N: tl.constexpr, K: tl.constexpr, LANES: tl.constexpr,
                     LOG_LANES: tl.constexpr, ROWS: tl.constexpr, TILE: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    selected = (rows < N) & (tl.load(Upper + rows, rows < N, other=-float("inf")) >= tl.load(MaxLower))
    tl.store(Selected + rows, selected, rows < N)
    if tl.sum(selected.to(tl.int32), 0) > 0:
        result = _evaluate(X, W, rows, selected, K, LANES, LOG_LANES, ROWS, TILE)
        result = tl.where(selected, result, -float("inf"))
    else:
        result = tl.full((ROWS,), -float("inf"), tl.float32)
    tl.store(Output + rows, result.to(tl.bfloat16), rows < N)


@triton.jit
def _ranks(Upper, MaxLower, Selected, Rank, Count, Output,
           N: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < N
    selected = valid & (tl.load(Upper + rows, valid, other=-float("inf")) >= tl.load(MaxLower))
    tl.store(Selected + rows, selected, valid)
    tl.store(Rank + rows, tl.cumsum(selected.to(tl.int32), 0) - 1, valid)
    tl.store(Count + tl.program_id(0), tl.sum(selected.to(tl.int32), 0))
    tl.store(Output + rows, -float("inf"), valid)


@triton.jit
def _scatter(Selected, Rank, Prefix, Indices, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    selected = (rows < N) & tl.load(Selected + rows, rows < N, other=False)
    base = tl.load(Prefix + block - 1, block > 0, other=0)
    tl.store(Indices + base + tl.load(Rank + rows, selected, other=0), rows, selected)


@triton.jit
def _verify_compact(X, W, Indices, Prefix, Output, K: tl.constexpr,
                    COUNT_BLOCKS: tl.constexpr, LANES: tl.constexpr,
                    LOG_LANES: tl.constexpr, ROWS: tl.constexpr,
                    GRID: tl.constexpr, TILE: tl.constexpr):
    total = tl.load(Prefix + COUNT_BLOCKS - 1)
    for iteration in range(tl.cdiv(total, GRID * ROWS)):
        position = (iteration * GRID + tl.program_id(0)) * ROWS + tl.arange(0, ROWS)
        valid = position < total
        if tl.sum(valid.to(tl.int32), 0) > 0:
            rows = tl.load(Indices + position, valid, other=0)
            result = _evaluate(X, W, rows, valid, K, LANES, LOG_LANES, ROWS, TILE)
            tl.store(Output + rows, result.to(tl.bfloat16), valid)


def _validate_verifier(k, lanes, rows, tile, warps):
    if lanes not in (8, 16, 32) or k % lanes:
        raise ValueError("Verifier requires lanes8/16/32 and K divisible by lanes")
    if rows not in (4, 8, 16, 32) or tile not in (8, 16, 32, 64):
        raise ValueError("Use verifier rows4/8/16/32 and preload tile8/16/32/64")
    if warps not in (2, 4, 8):
        raise ValueError("Use verifier warps2/4/8")


@torch.no_grad()
def verify_full_rows(x, weight, verifier_lanes=8, block_rows=8, tile=16, warps=4):
    """Independent full-shape audit path; does not build or use an INT8 index."""
    n, k = weight.shape
    _validate_verifier(k, verifier_lanes, block_rows, tile, warps)
    if x.numel() != k or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Use one BF16 query and the original BF16 weight")
    if not x.is_contiguous() or not weight.is_contiguous() or x.device != weight.device:
        raise ValueError("Query and original weight must be contiguous on the same device")
    output = torch.empty((*x.shape[:-1], n), device=weight.device, dtype=torch.bfloat16)
    _verify_full[(triton.cdiv(n, block_rows),)](x, weight, output, n, k,
        verifier_lanes, verifier_lanes.bit_length() - 1, block_rows, tile,
        num_warps=warps, enable_fp_fusion=False)
    return output


class BoundedHead3B(nn.Module):
    """DP4A8 screening plus original-BF16 row verification; conditional only."""
    def __init__(self, original, chunk_rows=2048, block_rows=8, compact=True,
                 index_bits=8, screen_rows=8, verifier_lanes=8, verifier_tile=16,
                 verifier_warps=4, experimental_index4=False):
        super().__init__()
        if original.bias is not None or original.weight.dtype != torch.bfloat16:
            raise ValueError("Use the original bias-free BF16 output head")
        self.weight = original.weight
        n, k = self.weight.shape
        if not self.weight.is_contiguous() or self.weight.device.type != "cuda":
            raise ValueError("Original weight must be contiguous and on CUDA")
        self.contract = numerical_contract(k)
        _validate_verifier(k, verifier_lanes, block_rows, verifier_tile, verifier_warps)
        if index_bits not in (4, 8) or (index_bits == 4 and not experimental_index4):
            raise ValueError("DP4A8 is default; index4 requires experimental_index4=True")
        if index_bits == 4 and k % 8:
            raise ValueError("Experimental index4 requires K divisible by8")
        if screen_rows not in (4, 8, 16, 32) or chunk_rows < 1:
            raise ValueError("Use screening rows4/8/16/32 and a positive setup chunk size")
        self.n, self.k = n, k
        self.block_rows, self.compact = block_rows, compact
        self.index_bits, self.screen_rows = index_bits, screen_rows
        self.verifier_lanes, self.verifier_tile, self.verifier_warps = verifier_lanes, verifier_tile, verifier_warps
        self.count_blocks = triton.cdiv(n, 256)
        self.weight_version = self.weight._version

        def buffer(name, shape, dtype):
            value = torch.empty(shape, device=self.weight.device, dtype=dtype)
            self.register_buffer(name, value, persistent=False)
            return value

        buffer("index_weight", (n, k), torch.int8)
        if index_bits == 4:
            buffer("index_packed", (n, k // 2), torch.uint8)
        for name in ("weight_scale", "weight_norm", "quant_norm", "error_norm"):
            buffer(name, (n,), torch.float64)
        buffer("weight_bad", (1,), torch.int32).zero_()
        # Original BF16 storage is retained, never overwritten or requantized.
        with torch.no_grad():
            for start in range(0, n, chunk_rows):
                end = min(n, start + chunk_rows)
                w = self.weight[start:end].to(torch.float64)
                finite = torch.isfinite(w)
                bad = (~finite).any() | (w.abs().amax() > 2.0 ** 40)
                torch.maximum(self.weight_bad, bad.to(torch.int32).view(1), out=self.weight_bad)
                w = torch.where(finite, w, 0.0)
                maximum = w.abs().amax(-1)
                quant_max = 127 if index_bits == 8 else 7
                scale32 = torch.where(maximum != 0, maximum / quant_max, 1.0).to(torch.float32)
                tiny = ((scale32.abs() < 2.0 ** -126) & (maximum != 0)).any()
                torch.maximum(self.weight_bad, tiny.to(torch.int32).view(1), out=self.weight_bad)
                scale32 = torch.where(scale32 != 0, scale32, 1.0)
                scale = scale32.to(torch.float64)
                quant = torch.round(w / scale[:, None]).clamp(-quant_max, quant_max).to(torch.int8)
                dequant = quant.to(torch.float64) * scale[:, None]
                self.index_weight[start:end].copy_(quant)
                if index_bits == 4:
                    codes = (quant.to(torch.int16) + 8).to(torch.uint8)
                    self.index_packed[start:end].copy_(codes[:, 0::2] | (codes[:, 1::2] << 4))
                self.weight_scale[start:end].copy_(scale)
                for name, value in (("weight_norm", w), ("quant_norm", dequant), ("error_norm", w - dequant)):
                    upper_norm = value.square().sum(-1).sqrt() * self.contract["norm_inflate_up"]
                    getattr(self, name)[start:end].copy_(upper_norm)
        buffer("query_index", (k,), torch.int8)
        buffer("query_words", (k // 4,), torch.int32)
        buffer("query_even", (k // 8,), torch.int32)
        buffer("query_odd", (k // 8,), torch.int32)
        buffer("query_stats", (3,), torch.float64)
        buffer("query_bad", (1,), torch.int32)
        buffer("integer_scores", (n,), torch.int32)
        buffer("lower", (n,), torch.float32)
        buffer("upper", (n,), torch.float32)
        buffer("maximum_lower", (), torch.float32)
        buffer("selected", (n,), torch.bool)
        buffer("logits", (n,), torch.bfloat16)
        if compact:
            buffer("local_rank", (n,), torch.int32)
            buffer("block_count", (self.count_blocks,), torch.int32)
            buffer("block_prefix", (self.count_blocks,), torch.int32)
            buffer("indices", (n,), torch.int32)

    def component_functions(self, x):
        functions = {
            "query": lambda: _query_index[(1,)](x, self.query_index, self.query_words,
                self.query_even, self.query_odd, self.query_stats, self.query_bad, self.weight_bad,
                self.k, triton.next_power_of_2(self.k), self.contract["norm_inflate_up"], self.index_bits,
                triton.next_power_of_2(self.k // (4 if self.index_bits == 8 else 8)),
                num_warps=4, enable_fp_fusion=False),
            "screen": lambda: _screen[(triton.cdiv(self.n, self.screen_rows),)](
                self.index_weight.view(torch.int32) if self.index_bits == 8 else self.index_packed.view(torch.int32),
                self.query_words, self.query_even, self.query_odd, self.integer_scores,
                self.n, self.k, self.index_bits, self.screen_rows, num_warps=4),
            "bounds": lambda: _bounds[(triton.cdiv(self.n, 256),)](self.integer_scores,
                self.weight_scale, self.weight_norm, self.quant_norm, self.error_norm,
                self.query_stats, self.query_bad, self.lower, self.upper,
                self.n, 256, self.contract["gamma_k_up"], self.contract["underflow_up"],
                num_warps=4, enable_fp_fusion=False),
            "amax": lambda: torch.amax(self.lower, dim=0, out=self.maximum_lower),
        }
        if self.compact:
            functions.update({
                "select": lambda: _ranks[(self.count_blocks,)](self.upper, self.maximum_lower,
                    self.selected, self.local_rank, self.block_count, self.logits,
                    self.n, 256, num_warps=4),
                "scan": lambda: torch.cumsum(self.block_count, dim=0, dtype=torch.int32, out=self.block_prefix),
                "scatter": lambda: _scatter[(self.count_blocks,)](self.selected, self.local_rank,
                    self.block_prefix, self.indices, self.n, 256, num_warps=4),
                "verify": lambda: _verify_compact[(128,)](x, self.weight, self.indices,
                    self.block_prefix, self.logits, self.k, self.count_blocks,
                    self.verifier_lanes, self.verifier_lanes.bit_length() - 1,
                    self.block_rows, 128, self.verifier_tile,
                    num_warps=self.verifier_warps, enable_fp_fusion=False),
            })
        else:
            functions["verify"] = lambda: _verify_selected[(triton.cdiv(self.n, self.block_rows),)](
                x, self.weight, self.upper, self.maximum_lower, self.selected, self.logits,
                self.n, self.k, self.verifier_lanes, self.verifier_lanes.bit_length() - 1,
                self.block_rows, self.verifier_tile,
                num_warps=self.verifier_warps, enable_fp_fusion=False)
        return functions

    @torch.no_grad()
    def forward(self, x):
        if x.numel() != self.k:
            return F.linear(x, self.weight)
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != self.weight.device:
            raise ValueError("Decode query must be contiguous BF16 on the original weight device")
        if self.weight._version != self.weight_version:
            raise RuntimeError("The original head changed after index construction")
        for function in self.component_functions(x).values():
            function()
        return self.logits.view(*x.shape[:-1], self.n)

    @torch.no_grad()
    def audit(self, x):
        self(x)
        return self.audit_buffers(x)

    @torch.no_grad()
    def audit_buffers(self, x):
        actual = self.logits.reshape(-1)
        reference = F.linear(x, self.weight).reshape(-1)
        finite = torch.isfinite(reference)
        same = actual.view(torch.int16) == reference.view(torch.int16)
        enclosed = (reference.float() >= self.lower) & (reference.float() <= self.upper)
        winner = int(reference.argmax().item())
        return {"selected_rows": int(self.selected.sum().item()),
                "selected_fraction": float(self.selected.float().mean().item()),
                "selected_bits_exact": bool((same | ~self.selected).all().item()),
                "selected_bit_mismatches": int((~same & self.selected).sum().item()),
                "all_finite_reference_logits_enclosed": bool((enclosed | ~finite).all().item()),
                "reference_winner_selected": bool(self.selected[winner].item()),
                "reference_argmax": winner, "candidate_argmax": int(actual.argmax().item()),
                "argmax_exact": bool(actual.argmax() == reference.argmax()),
                "all_rows_fallback": bool(self.query_bad.item()),
                "conditional_only": True}


# Explicitly named compatibility alias; this file never patches the 0.5B head.
BoundedHeadPrototype3B = BoundedHead3B
