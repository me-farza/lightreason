// Compiled routing metadata and persistent workers for Packed's BF16 CPU projections.
#include "cpu_experts.c"

#include <algorithm>
#include <array>
#include <barrier>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {
class Workers {
public:
    explicit Workers(int count) : gate_ready(count) {
        for (int id = 0; id < count; ++id) {
            threads.emplace_back([this, id, count] { worker(id, count); });
        }
    }

    ~Workers() {
        {
            std::lock_guard lock(mutex);
            stop = true;
        }
        available.notify_all();
        for (auto& thread : threads) thread.join();
    }

    void submit(const uint16_t* const* gates, const uint16_t* const* downs,
                const uint16_t* input, uint16_t* result, int count) {
        {
            std::lock_guard lock(mutex);
            std::copy_n(gates, count, gate.begin());
            std::copy_n(downs, count, down.begin());
            for (int i = 0; i < HIDDEN; ++i) hidden[i] = from_bfloat(input[i]);
            experts = count;
            output = result;
            pending = threads.size();
            ++epoch;
        }
        available.notify_all();
    }

    void wait() {
        std::unique_lock lock(mutex);
        done.wait(lock, [this] { return pending == 0; });
    }

private:
    void worker(int id, int count) {
        unsigned last_epoch = 0;
        while (true) {
            {
                std::unique_lock lock(mutex);
                available.wait(lock, [&] { return stop || epoch != last_epoch; });
                if (stop) return;
                last_epoch = epoch;
            }
            const int gate_rows = experts * INNER;
            for (int i = gate_rows * id / count; i < gate_rows * (id + 1) / count; ++i) {
                const int expert = i / INNER;
                const int row = i % INNER;
                const uint16_t* gate_row = gate[expert] + row * HIDDEN;
                const uint16_t* up_row = gate_row + INNER * HIDDEN;
                const float gate_value = from_bfloat(to_bfloat(dot(gate_row, hidden.data(), HIDDEN)));
                const float up_value = from_bfloat(to_bfloat(dot(up_row, hidden.data(), HIDDEN)));
                const float silu = from_bfloat(to_bfloat(gate_value / (1.0f + expf(-gate_value))));
                active[expert][row] = from_bfloat(to_bfloat(silu * up_value));
            }
            gate_ready.arrive_and_wait();
            const int down_rows = experts * HIDDEN;
            for (int i = down_rows * id / count; i < down_rows * (id + 1) / count; ++i) {
                const int expert = i / HIDDEN;
                const int row = i % HIDDEN;
                output[i] = to_bfloat(dot(down[expert] + row * INNER, active[expert].data(), INNER));
            }
            {
                std::lock_guard lock(mutex);
                if (--pending == 0) done.notify_one();
            }
        }
    }

    std::mutex mutex;
    std::condition_variable available, done;
    std::barrier<> gate_ready;
    std::vector<std::thread> threads;
    std::array<const uint16_t*, ROUTES> gate, down;
    std::array<float, HIDDEN> hidden;
    std::array<std::array<float, INNER>, ROUTES> active;
    uint16_t* output = nullptr;
    unsigned epoch = 0;
    size_t pending = 0;
    int experts = 0;
    bool stop = false;
};
}

extern "C" {
void packed_plan(const int64_t* ids, int64_t* table, int32_t* last_seen, int step,
                 int32_t* experts, uint8_t* recent) {
    std::array<int, ROUTES> slots{0, 1, 2, 3, 4, 5, 6, 7};
    std::sort(slots.begin(), slots.end(), [&](int a, int b) { return ids[a] < ids[b]; });
    for (int i = 0; i < ROUTES; ++i) {
        const int expert = static_cast<int>(ids[slots[i]]);
        experts[i] = expert;
        recent[i] = step - last_seen[expert] <= 4;
        last_seen[expert] = step;
        table[4 * i + 2] = slots[i];
    }
}

void* packed_workers_create(int threads) { return new Workers(threads); }
void packed_workers_destroy(void* workers) { delete static_cast<Workers*>(workers); }
void packed_workers_submit(void* workers, const uint16_t* const* gate, const uint16_t* const* down,
                           const uint16_t* input, uint16_t* output, int experts) {
    static_cast<Workers*>(workers)->submit(gate, down, input, output, experts);
}
void packed_workers_wait(void* workers) { static_cast<Workers*>(workers)->wait(); }
}
