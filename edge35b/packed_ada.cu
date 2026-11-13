// One warp per hidden row. Compute the gate and up projections together.
using bf16 = unsigned short;

__device__ float unpack(bf16 value) {
    return __uint_as_float(static_cast<unsigned>(value) << 16);
}

__device__ bf16 pack(float value) {
    unsigned bits = __float_as_uint(value);
    return static_cast<bf16>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

extern "C" __global__ void packed_gate_silu(
    const unsigned long long* table, const bf16* hidden, bf16* active
) {
    const int expert = blockIdx.y;
    const auto* weight = reinterpret_cast<const bf16*>(table[4 * expert]);
    if (!weight) return;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    float gate = 0.0f;
    float up = 0.0f;
    const auto* gate_row = weight + row * 2048;
    const auto* up_row = gate_row + 512 * 2048;
    for (int column = lane; column < 2048; column += 32) {
        const float x = unpack(hidden[column]);
        gate += unpack(gate_row[column]) * x;
        up += unpack(up_row[column]) * x;
    }
    for (int shift = 16; shift; shift /= 2) {
        gate += __shfl_down_sync(0xffffffff, gate, shift);
        up += __shfl_down_sync(0xffffffff, up, shift);
    }
    if (lane == 0) {
        gate = unpack(pack(gate));
        up = unpack(pack(up));
        const float silu = unpack(pack(gate / (1.0f + expf(-gate))));
        active[expert * 512 + row] = pack(silu * up);
    }
}
