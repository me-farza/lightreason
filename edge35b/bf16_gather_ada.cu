// Decode the lossless BF16 transport format directly into the cache slots.
// Each block owns a disjoint, contiguous section of one row and bank.
struct __align__(8) WordPair {
    unsigned int x, y;
};

struct __align__(16) WordQuad {
    unsigned int x, y, z, w;
};

__device__ __forceinline__ unsigned int unpack_pair(unsigned int signs_and_fractions,
                                                     unsigned int deltas,
                                                     unsigned int base_pair) {
    const unsigned int sign = ((signs_and_fractions & 0x80u) << 8) |
                              ((signs_and_fractions & 0x8000u) << 16);
    const unsigned int fraction = (signs_and_fractions & 0x7fu) |
                                 ((signs_and_fractions & 0x7f00u) << 8);
    const unsigned int offset = (deltas & 0x0fu) | ((deltas & 0xf0u) << 12);
    return sign | fraction | ((base_pair + offset) << 7);
}

__device__ __forceinline__ int lower_bound(const int* indices, int count, int value) {
    int first = 0;
    int last = count;
    while (first < last) {
        const int mid = first + (last - first) / 2;
        if (indices[mid] < value) {
            first = mid + 1;
        } else {
            last = mid;
        }
    }
    return first;
}

template <bool CacheInL2>
__device__ __forceinline__ void gather(const long long* descriptors,
                                       const long long* destinations,
                                       const long long* sizes,
                                       const int* source_ids,
                                       const int* destination_ids,
                                       const long long* count,
                                       int experts) {
    const int stripe = blockIdx.x;
    const int bank = blockIdx.y;
    const int row = blockIdx.z;
    if (row >= *count) {
        return;
    }
    const auto* desc = descriptors + (bank * experts + source_ids[row]) * 3;
    const auto* src = reinterpret_cast<const unsigned char*>(desc[0]);
    const int base = static_cast<int>(desc[1]);
    const int escapes = static_cast<int>(desc[2]);
    const int row_bytes = static_cast<int>(sizes[bank]);
    auto* dst = reinterpret_cast<WordQuad*>(destinations[bank] +
                                           static_cast<long long>(destination_ids[row]) * row_bytes);
    const int units = (row_bytes >> 4) / gridDim.x;
    const int begin = stripe * units;
    const int end = begin + units;
    if (base < 0) {
        const auto* raw = reinterpret_cast<const WordQuad*>(src);
        for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
            dst[i] = raw[i];
        }
        return;
    }

    const auto* signs = reinterpret_cast<const WordPair*>(src);
    const auto* deltas = reinterpret_cast<const unsigned int*>(src + row_bytes / 2);
    const unsigned int base_pair = static_cast<unsigned int>(base) * 0x10001u;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        WordPair sm;
        unsigned int exp;
        if constexpr (CacheInL2) {
            asm("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(sm.x), "=r"(sm.y) : "l"(signs + i));
            asm("ld.global.cg.u32 %0, [%1];" : "=r"(exp) : "l"(deltas + i));
        } else {
            sm = signs[i];
            exp = deltas[i];
        }
        dst[i] = {unpack_pair(sm.x, exp, base_pair),
                  unpack_pair(sm.x >> 16, exp >> 8, base_pair),
                  unpack_pair(sm.y, exp >> 16, base_pair),
                  unpack_pair(sm.y >> 16, exp >> 24, base_pair)};
    }

    if (escapes > 0) {
        // Escape indices are sorted, so only this block can repair its section.
        const auto* indices = reinterpret_cast<const int*>(src + row_bytes / 2 + row_bytes / 4);
        const auto* values = reinterpret_cast<const unsigned short*>(indices + escapes);
        auto* restored = reinterpret_cast<unsigned short*>(dst);
        __shared__ int first;
        __shared__ int last;
        if (threadIdx.x == 0) {
            first = lower_bound(indices, escapes, begin * 8);
            last = lower_bound(indices, escapes, end * 8);
        }
        __syncthreads();
        for (int i = first + threadIdx.x; i < last; i += blockDim.x) {
            restored[indices[i]] = values[i];
        }
    }
}

extern "C" __global__ void gather_bf16(const long long* descriptors, const long long* destinations,
                                       const long long* sizes, const int* source_ids,
                                       const int* destination_ids, const long long* count, int experts) {
    gather<false>(descriptors, destinations, sizes, source_ids, destination_ids, count, experts);
}

extern "C" __global__ void gather_bf16_l2(const long long* descriptors, const long long* destinations,
                                          const long long* sizes, const int* source_ids,
                                          const int* destination_ids, const long long* count, int experts) {
    gather<true>(descriptors, destinations, sizes, source_ids, destination_ids, count, experts);
}

__device__ __forceinline__ unsigned int unpack_cpu_tile_pair(unsigned int low,
                                                              unsigned int codes,
                                                              unsigned int base_pair) {
    const unsigned int nibbles = (codes & 0x0fu) | ((codes & 0xf0u) << 12);
    const unsigned int high = (base_pair + (nibbles & 0x00070007u)) | ((nibbles & 0x00080008u) << 4);
    return (low & 0xffu) | ((low & 0xff00u) << 8) | (high << 8);
}

// One warp copies one 256-value CPU tile. Bases reside in GPU memory.
// Each descriptor supplies the packed host row, base array, and raw row.
template <bool AllowCompact>
__device__ __forceinline__ void gather_cpu_tiles(const long long* descriptors,
                                                const long long* destinations,
                                                const long long* sizes,
                                                const int* source_ids,
                                                const int* destination_ids,
                                                const long long* count, int experts) {
    const int bank = blockIdx.y;
    const int row = blockIdx.z;
    if (row >= *count) {
        return;
    }
    const auto* desc = descriptors + (bank * experts + source_ids[row]) * 3;
    const auto* packed = reinterpret_cast<const unsigned char*>(desc[0]);
    const auto* bases = reinterpret_cast<const unsigned char*>(desc[1]);
    const auto* raw = reinterpret_cast<const WordQuad*>(desc[2]);
    const int row_bytes = static_cast<int>(sizes[bank]);
    auto* dst = reinterpret_cast<WordQuad*>(destinations[bank] +
                                           static_cast<long long>(destination_ids[row]) * row_bytes);
    const int tiles_per_stripe = (row_bytes / 512) / gridDim.x;
    const int start = blockIdx.x * tiles_per_stripe;
    const int end = start + tiles_per_stripe;
    if (bases == nullptr) {
        for (int i = start * 32 + threadIdx.x; i < end * 32; i += blockDim.x) {
            dst[i] = raw[i];
        }
        return;
    }
    const int lane = threadIdx.x & 31;
    // A compact GPU row stores raw fallbacks inline. The header records the
    // base byte and how many 128-byte raw extensions precede the tile.
    const bool compact = AllowCompact && desc[2] == -1;
    for (int tile = start + (threadIdx.x >> 5); tile < end; tile += blockDim.x >> 5) {
        unsigned int header = 0;
        if (lane == 0) {
            header = compact ? reinterpret_cast<const unsigned int*>(bases)[tile] : bases[tile];
        }
        header = __shfl_sync(0xffffffffu, header, 0);
        const unsigned int base = compact ? header & 255 : header;
        const auto* tile_data = packed + tile * 384 + (compact ? (header >> 8) * 128 : 0);
        const int i = tile * 32 + lane;
        if (base == 255) {
            dst[i] = compact ? reinterpret_cast<const WordQuad*>(tile_data)[lane] : raw[i];
            continue;
        }
        const WordPair low = reinterpret_cast<const WordPair*>(tile_data)[lane];
        const unsigned int codes = reinterpret_cast<const unsigned int*>(tile_data + 256)[lane];
        const unsigned int pair = base * 0x10001u;
        dst[i] = {unpack_cpu_tile_pair(low.x, codes, pair),
                  unpack_cpu_tile_pair(low.x >> 16, codes >> 8, pair),
                  unpack_cpu_tile_pair(low.y, codes >> 16, pair),
                  unpack_cpu_tile_pair(low.y >> 16, codes >> 24, pair)};
    }
}

extern "C" __global__ void gather_bf16_cpu_tiles(const long long* descriptors,
                                                 const long long* destinations,
                                                 const long long* sizes,
                                                 const int* source_ids,
                                                 const int* destination_ids,
                                                 const long long* count, int experts) {
    gather_cpu_tiles<false>(descriptors, destinations, sizes, source_ids, destination_ids, count, experts);
}

extern "C" __global__ void gather_bf16_resident(const long long* descriptors,
                                                const long long* destinations,
                                                const long long* sizes,
                                                const int* source_ids,
                                                const int* destination_ids,
                                                const long long* count, int experts) {
    gather_cpu_tiles<true>(descriptors, destinations, sizes, source_ids, destination_ids, count, experts);
}

// Move the selected experts into a compact BF16 work buffer for the unchanged
// expert GEMM. For a miss, the same source load also fills the primary cache.
extern "C" __global__ void stage_bf16_primary(
    const long long* sources, const long long* miss_sources, const long long* layouts, const long long* prefix_tables,
    const int* plan, const int* count, int experts, int packed_slots, int max_rows) {
    const int row = blockIdx.z;
    if (row >= *count) return;
    const int bank = blockIdx.y;
    const int expert = plan[row];
    const int slot = plan[max_rows + row];
    const bool miss = plan[2 * max_rows + row];
    const bool packed_slot = slot >= 0 && slot < packed_slots;
    if (!miss && slot >= packed_slots) return;
    const auto* layout = layouts + bank * 5;
    const int row_bytes = static_cast<int>(layout[4]);
    auto* cached = slot < 0 ? nullptr : reinterpret_cast<unsigned char*>(
        packed_slot ? layout[0] + static_cast<long long>(slot) * layout[3]
                    : layout[1] + static_cast<long long>(slot - packed_slots) * row_bytes);
    auto* dest = slot >= packed_slots ? reinterpret_cast<WordQuad*>(cached)
        : reinterpret_cast<WordQuad*>(layout[2] + static_cast<long long>(row + 1) * row_bytes);
    const auto* source = sources + (bank * experts + expert) * 3;
    const auto* refill = miss_sources + (bank * experts + expert) * 3;
    const auto* code = reinterpret_cast<const unsigned char*>(source[0]);
    const auto* bases = reinterpret_cast<const unsigned char*>(source[1]);
    const auto* raw = reinterpret_cast<const WordQuad*>(source[2]);
    const bool replica_packed = miss && refill[2] == -1;
    const bool replica_raw = miss && refill[1] == 0 && refill[2] != source[2];
    const auto* replica = reinterpret_cast<const unsigned char*>(refill[0]);
    const auto* replica_raw_row = reinterpret_cast<const WordQuad*>(refill[2]);
    const auto* replica_headers = reinterpret_cast<const unsigned int*>(refill[1]);
    const int tiles = row_bytes / 512;
    const int tiles_per_block = tiles / gridDim.x;
    const int first = blockIdx.x * tiles_per_block;
    const int last = first + tiles_per_block;

    if (!packed_slot && replica_raw) raw = replica_raw_row;
    if (!packed_slot && !replica_packed && (replica_raw || bases == nullptr)) {
        for (int i = first * 32 + threadIdx.x; i < last * 32; i += blockDim.x) {
            dest[i] = raw[i];
        }
        return;
    }

    const auto* groups = reinterpret_cast<const unsigned long long*>(prefix_tables[bank]) + expert * (tiles / 32);
    const int lane = threadIdx.x & 31;
    for (int tile = first + (threadIdx.x >> 5); tile < last; tile += blockDim.x >> 5) {
        unsigned int base = 0, prior = 0, replica_offset = 0;
        if (lane == 0) {
            if (replica_packed) {
                const auto header = replica_headers[tile];
                replica_offset = tile * 384 + (header >> 8) * 128;
                if (!packed_slot) base = header & 255;
            }
            if (packed_slot || !replica_packed) base = bases[tile];
            if (packed_slot) {
                const auto group = groups[tile >> 5];
                const unsigned int earlier = (1u << (tile & 31)) - 1;
                prior = (group >> 32) + __popc(static_cast<unsigned int>(group) & earlier);
            }
        }
        base = __shfl_sync(0xffffffffu, base, 0);
        prior = __shfl_sync(0xffffffffu, prior, 0);
        replica_offset = __shfl_sync(0xffffffffu, replica_offset, 0);
        const int offset = tile * 384 + prior * 128;
        const int i = tile * 32 + lane;
        WordQuad values;
        if (base == 255) {
            if (replica_packed) values = reinterpret_cast<const WordQuad*>(replica + replica_offset)[lane];
            else if (packed_slot && !miss) values = reinterpret_cast<const WordQuad*>(cached + offset)[lane];
            else if (replica_raw) values = replica_raw_row[i];
            else values = raw[i];
            if (packed_slot && miss) reinterpret_cast<WordQuad*>(cached + offset)[lane] = values;
        } else {
            WordPair low;
            unsigned int nibble;
            if (replica_raw) {
                values = replica_raw_row[i];
                const unsigned int halves[4] = {values.x, values.y, values.z, values.w};
                unsigned int lows[4];
                nibble = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const auto value = halves[j];
                    lows[j] = (value & 255u) | ((value >> 8) & 65280u);
                    const unsigned int a = (((value >> 8) & 127u) - base) | ((value >> 12) & 8u);
                    const unsigned int b = (((value >> 24) & 127u) - base) | ((value >> 28) & 8u);
                    nibble |= (a | (b << 4)) << (j * 8);
                }
                low = {lows[0] | (lows[1] << 16), lows[2] | (lows[3] << 16)};
            } else {
                const auto* data = replica_packed ? replica + replica_offset
                    : packed_slot && !miss ? cached + offset : code + tile * 384;
                low = reinterpret_cast<const WordPair*>(data)[lane];
                nibble = reinterpret_cast<const unsigned int*>(data + 256)[lane];
                const unsigned int pair = base * 0x10001u;
                values = {unpack_cpu_tile_pair(low.x, nibble, pair),
                          unpack_cpu_tile_pair(low.x >> 16, nibble >> 8, pair),
                          unpack_cpu_tile_pair(low.y, nibble >> 16, pair),
                          unpack_cpu_tile_pair(low.y >> 16, nibble >> 24, pair)};
            }
            if (packed_slot && miss) {
                auto* dst = cached + offset;
                reinterpret_cast<WordPair*>(dst)[lane] = low;
                reinterpret_cast<unsigned int*>(dst + 256)[lane] = nibble;
            }
        }
        dest[i] = values;
    }
}

// Fill the original BF16 slot. A cacheable miss also writes compact bytes
// into the reclaimable prefill region, without a second host read.
extern "C" __global__ void gather_bf16_compact_tail(
    const long long* sources, const long long* destinations, const long long* sizes,
    const long long* prefix_tables, const long long* tail, const int* source_ids,
    const int* dest_ids, const long long* count, const int* entries, const long long* resident,
    int experts, int max_rows) {
    const int row = blockIdx.z;
    if (row >= *count) return;
    const int bank = blockIdx.y;
    const int expert = source_ids[row];
    const int entry = entries[row];
    const bool is_resident = entries[max_rows + row] != 0;
    const int row_bytes = static_cast<int>(sizes[bank]);
    auto* dest = reinterpret_cast<WordQuad*>(destinations[bank] + static_cast<long long>(dest_ids[row]) * row_bytes);
    const auto* table = is_resident ? resident : sources;
    const auto* source = table + (bank * experts + expert) * 3;
    const auto* bases = reinterpret_cast<const unsigned char*>(source[1]);
    const auto* raw = reinterpret_cast<const WordQuad*>(source[2]);
    const auto* mapped = reinterpret_cast<const unsigned char*>(source[0]);
    auto* storage = entry < 0 ? nullptr : reinterpret_cast<unsigned char*>(
        tail[bank * 2] + static_cast<long long>(entry) * tail[bank * 2 + 1]);
    const int tiles = row_bytes / 512;
    const int start = blockIdx.x * (tiles / gridDim.x);
    const int end = start + tiles / gridDim.x;
    if (bases == nullptr) {
        for (int i = start * 32 + threadIdx.x; i < end * 32; i += blockDim.x) dest[i] = raw[i];
        return;
    }
    const bool replica = source[2] == -1;
    const int lane = threadIdx.x & 31;
    for (int tile = start + (threadIdx.x >> 5); tile < end; tile += blockDim.x >> 5) {
        unsigned int header = 0;
        if (lane == 0) {
            if (replica) {
                header = reinterpret_cast<const unsigned int*>(bases)[tile];
            } else {
                header = bases[tile];
                if (storage != nullptr) {
                    const auto* prefixes = reinterpret_cast<const unsigned long long*>(prefix_tables[bank]);
                    const auto prefix = prefixes[expert * (tiles / 32) + (tile >> 5)];
                    const unsigned int prior = (prefix >> 32) +
                        __popc(static_cast<unsigned int>(prefix) & ((1u << (tile & 31)) - 1));
                    header |= prior << 8;
                }
            }
        }
        header = __shfl_sync(0xffffffffu, header, 0);
        const unsigned int base = header & 255;
        const int offset = tile * 384 + (header >> 8) * 128;
        const int i = tile * 32 + lane;
        if (base == 255) {
            const WordQuad values = is_resident ? reinterpret_cast<const WordQuad*>(storage + offset)[lane]
                : replica ? reinterpret_cast<const WordQuad*>(mapped + offset)[lane] : raw[i];
            dest[i] = values;
            if (!is_resident && storage != nullptr) reinterpret_cast<WordQuad*>(storage + offset)[lane] = values;
            continue;
        }
        const auto* data = is_resident ? storage + offset : mapped + (replica ? offset : tile * 384);
        const WordPair low = reinterpret_cast<const WordPair*>(data)[lane];
        const unsigned int nibble = reinterpret_cast<const unsigned int*>(data + 256)[lane];
        const unsigned int pair = base * 0x10001u;
        dest[i] = {unpack_cpu_tile_pair(low.x, nibble, pair),
                   unpack_cpu_tile_pair(low.x >> 16, nibble >> 8, pair),
                   unpack_cpu_tile_pair(low.y, nibble >> 16, pair),
                   unpack_cpu_tile_pair(low.y >> 16, nibble >> 24, pair)};
        if (!is_resident && storage != nullptr) {
            reinterpret_cast<WordPair*>(storage + offset)[lane] = low;
            reinterpret_cast<unsigned int*>(storage + offset + 256)[lane] = nibble;
        }
    }
}
