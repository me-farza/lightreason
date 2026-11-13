#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace freetoken::cpu_moe {

constexpr int kPackedReadTile = 256;
constexpr uint8_t kRawReadTile = 255;
constexpr int kLastCpuReadPolicy = 8;

// The low BF16 byte is unchanged. The high byte is a signed nibble index
// into an eight-exponent window for this tile. Outliers keep the raw tile.
inline size_t pack_cpu_read_tiles(const uint16_t* raw, uint8_t* packed, uint8_t* bases, size_t tiles) {
  size_t encoded = 0;
  for (size_t tile = 0; tile < tiles; ++tile) {
    const uint16_t* weights = raw + tile * kPackedReadTile;
    int low = 127, high = 0;
    for (int i = 0; i < kPackedReadTile; ++i) {
      const int exponent = (weights[i] >> 8) & 127;
      low = std::min(low, exponent);
      high = std::max(high, exponent);
    }
    if (high - low > 7) {
      bases[tile] = kRawReadTile;
      continue;
    }
    bases[tile] = low;
    uint8_t* fraction = packed + tile * (kPackedReadTile * 3 / 2);
    uint8_t* indices = fraction + kPackedReadTile;
    for (int i = 0; i < kPackedReadTile; i += 2) {
      const int a = weights[i] >> 8;
      const int b = weights[i + 1] >> 8;
      fraction[i] = weights[i] & 255;
      fraction[i + 1] = weights[i + 1] & 255;
      const int ac = ((a & 127) - low) | ((a & 128) >> 4);
      const int bc = ((b & 127) - low) | ((b & 128) >> 4);
      indices[i / 2] = ac | (bc << 4);
    }
    for (int i = 0; i < kPackedReadTile; ++i) {
      const int code = (indices[i / 2] >> ((i & 1) * 4)) & 15;
      const int high_byte = (low + (code & 7)) | ((code & 8) << 4);
      if ((fraction[i] | (high_byte << 8)) != weights[i]) {
        throw std::runtime_error("Packed CPU expert failed its BF16 bit audit");
      }
    }
    ++encoded;
  }
  return encoded;
}

#if CPU_MOE_X86
CPU_MOE_TARGET("avx2,fma") __attribute__((always_inline))
inline __m256 cpu_read_f32(__m128i bits) {
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

CPU_MOE_TARGET("avx2,fma") __attribute__((always_inline))
inline void unpack_cpu_read16(const uint8_t* low, const uint8_t* codes, __m128i table, __m256* weight) {
  const __m128i nibbles = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes)));
  const __m128i offsets = _mm_and_si128(_mm_or_si128(nibbles, _mm_slli_epi16(nibbles, 4)),
                                        _mm_set1_epi16(0x0f0f));
  const __m128i high = _mm_shuffle_epi8(table, offsets);
  const __m128i fraction = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low));
  weight[0] = cpu_read_f32(_mm_unpacklo_epi8(fraction, high));
  weight[1] = cpu_read_f32(_mm_unpackhi_epi8(fraction, high));
}

CPU_MOE_TARGET("avx2,fma") __attribute__((always_inline))
inline void unpack_cpu_read32(const uint8_t* low, const uint8_t* codes, __m256i table, __m256* weight) {
  const __m256i nibbles = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(codes)));
  const __m256i offsets = _mm256_and_si256(_mm256_or_si256(nibbles, _mm256_slli_epi16(nibbles, 4)),
                                           _mm256_set1_epi16(0x0f0f));
  const __m256i high = _mm256_shuffle_epi8(table, offsets);
  const __m256i fraction = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(low));
  const __m256i lo = _mm256_unpacklo_epi8(fraction, high);
  const __m256i hi = _mm256_unpackhi_epi8(fraction, high);
  weight[0] = cpu_read_f32(_mm256_castsi256_si128(lo));
  weight[1] = cpu_read_f32(_mm256_castsi256_si128(hi));
  weight[2] = cpu_read_f32(_mm256_extracti128_si256(lo, 1));
  weight[3] = cpu_read_f32(_mm256_extracti128_si256(hi, 1));
}

template <int Routes>
CPU_MOE_TARGET("avx2,fma") __attribute__((always_inline))
inline void accumulate_cpu_read(const __m256* weight, const bf16_t* const (&inputs)[Routes],
                                int offset, __m256 (&acc)[Routes][4]) {
  for (int route = 0; route < Routes; ++route) {
    for (int lane = 0; lane < 4; ++lane) {
      const auto x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(inputs[route] + offset + lane * 8));
      acc[route][lane] = _mm256_fmadd_ps(weight[lane], cpu_read_f32(x), acc[route][lane]);
    }
  }
}

template <int Routes, int Ahead>
CPU_MOE_TARGET("avx2,fma") __attribute__((always_inline))
inline void raw_cpu_read_span(const bf16_t* raw, const bf16_t* const (&inputs)[Routes],
                              int begin, int end, __m256 (&acc)[Routes][4]) {
  for (int i = begin; i < end; i += 32) {
    if constexpr (Ahead > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(raw + i) + Ahead, _MM_HINT_T0);
    }
    __m256 weight[4];
    for (int lane = 0; lane < 4; ++lane) {
      weight[lane] = cpu_read_f32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + i + lane * 8)));
    }
    accumulate_cpu_read(weight, inputs, i, acc);
  }
}

template <int Routes, int Ahead, bool Packed, bool Wide = true>
CPU_MOE_TARGET("avx2,fma")
void dot_cpu_reads(const bf16_t* raw, const uint8_t* packed, const uint8_t* bases,
                   const bf16_t* const (&inputs)[Routes], int width, float (&output)[Routes]) {
  __m256 acc[Routes][4];
  for (auto& lanes : acc) {
    for (auto& lane : lanes) lane = _mm256_setzero_ps();
  }
  if constexpr (!Packed) {
    raw_cpu_read_span<Routes, Ahead>(raw, inputs, 0, width, acc);
  } else {
    for (int tile = 0; tile < width / kPackedReadTile; ++tile) {
      const int offset = tile * kPackedReadTile;
      const int base = bases[tile];
      if (base == kRawReadTile) {
        raw_cpu_read_span<Routes, PF_AHEAD>(raw, inputs, offset, offset + kPackedReadTile, acc);
        continue;
      }
      const auto* fraction = packed + tile * (kPackedReadTile * 3 / 2);
      const auto* codes = fraction + kPackedReadTile;
      const __m128i offsets = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7,
                                             -128, -127, -126, -125, -124, -123, -122, -121);
      const __m128i table = _mm_add_epi8(_mm_set1_epi8(base), offsets);
      for (int i = 0; i < kPackedReadTile; i += 32) {
        if constexpr (Ahead > 0) {
          _mm_prefetch(reinterpret_cast<const char*>(fraction + i) + Ahead, _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(codes + i / 2) + Ahead / 2, _MM_HINT_T0);
        }
        __m256 weight[4];
        if constexpr (Wide) {
          unpack_cpu_read32(fraction + i, codes + i / 2, _mm256_broadcastsi128_si256(table), weight);
        } else {
          unpack_cpu_read16(fraction + i, codes + i / 2, table, weight);
          unpack_cpu_read16(fraction + i + 16, codes + i / 2 + 8, table, weight + 2);
        }
        accumulate_cpu_read(weight, inputs, offset + i, acc);
      }
    }
  }
  for (int route = 0; route < Routes; ++route) {
    const auto& a = acc[route];
    output[route] = hsum256(_mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3])));
  }
}
#endif

}  // namespace freetoken::cpu_moe
