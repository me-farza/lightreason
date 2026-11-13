/* Run single-token BF16 expert projections directly from mapped host weights. */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define HIDDEN 2048
#define INNER 512
#define ROUTES 8

static float from_bfloat(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t to_bfloat(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

static __m256 load_bfloat(const uint16_t *weights) {
    __m128i words = _mm_loadu_si128((const __m128i *)weights);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(words), 16));
}

static float dot(const uint16_t *weights, const float *input, int width) {
    __m256 low = _mm256_setzero_ps();
    __m256 high = _mm256_setzero_ps();
    for (int i = 0; i < width; i += 16) {
        low = _mm256_fmadd_ps(load_bfloat(weights + i), _mm256_loadu_ps(input + i), low);
        high = _mm256_fmadd_ps(load_bfloat(weights + i + 8), _mm256_loadu_ps(input + i + 8), high);
    }
    __m256 lanes = _mm256_add_ps(low, high);
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(lanes), _mm256_extractf128_ps(lanes, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void expert_rows(const uint16_t *const *gate_up, const uint16_t *const *down,
                 const uint16_t *input, uint16_t *output, int experts, int threads) {
    float hidden[HIDDEN];
    float active[ROUTES][INNER];
    for (int i = 0; i < HIDDEN; ++i) {
        hidden[i] = from_bfloat(input[i]);
    }

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (int index = 0; index < experts * INNER; ++index) {
            int expert = index / INNER;
            int row = index % INNER;
            const uint16_t *gate_row = gate_up[expert] + row * HIDDEN;
            const uint16_t *up_row = gate_row + INNER * HIDDEN;
            float gate = from_bfloat(to_bfloat(dot(gate_row, hidden, HIDDEN)));
            float up = from_bfloat(to_bfloat(dot(up_row, hidden, HIDDEN)));
            float silu = from_bfloat(to_bfloat(gate / (1.0f + expf(-gate))));
            active[expert][row] = from_bfloat(to_bfloat(silu * up));
        }
        #pragma omp for schedule(static)
        for (int index = 0; index < experts * HIDDEN; ++index) {
            int expert = index / HIDDEN;
            int row = index % HIDDEN;
            output[index] = to_bfloat(dot(down[expert] + row * INNER, active[expert], INNER));
        }
    }
}

/* A verify block can route a different input token to each cold expert. */
void expert_block_rows(const uint16_t *const *gate_up, const uint16_t *const *down,
                       const uint16_t *input, const int32_t *token_ids, uint16_t *output,
                       int tokens, int experts, int threads) {
    float hidden[16][HIDDEN];
    float active[16 * ROUTES][INNER];
    for (int token = 0; token < tokens; ++token) {
        for (int i = 0; i < HIDDEN; ++i) {
            hidden[token][i] = from_bfloat(input[token * HIDDEN + i]);
        }
    }
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (int index = 0; index < experts * INNER; ++index) {
            int expert = index / INNER;
            int row = index % INNER;
            const uint16_t *gate_row = gate_up[expert] + row * HIDDEN;
            const uint16_t *up_row = gate_row + INNER * HIDDEN;
            const float *vector = hidden[token_ids[expert]];
            float gate = from_bfloat(to_bfloat(dot(gate_row, vector, HIDDEN)));
            float up = from_bfloat(to_bfloat(dot(up_row, vector, HIDDEN)));
            float silu = from_bfloat(to_bfloat(gate / (1.0f + expf(-gate))));
            active[expert][row] = from_bfloat(to_bfloat(silu * up));
        }
        #pragma omp for schedule(static)
        for (int index = 0; index < experts * HIDDEN; ++index) {
            int expert = index / HIDDEN;
            int row = index % HIDDEN;
            output[index] = to_bfloat(dot(down[expert] + row * INNER, active[expert], INNER));
        }
    }
}
