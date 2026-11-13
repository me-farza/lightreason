"""Check mixed raw and packed cache writes on CPU or a captured CUDA graph."""

import os
import unittest

import numpy as np
import torch

from edge35b.bf16_expert_codec import exponent_window, pack_verified_row, plan_bank, row_bytes
from edge35b.freetoken_packed_transfer import _copy_raw_tiles, _copy_rows, _repair_rows, _snapshot_fetched, _split_hot_misses
from edge35b.width_autotune import GATHER_GEOMETRIES


class PackedTransportTests(unittest.TestCase):
    def test_exponent_range_and_raw_fallback(self):
        # All 65,536 BF16 bit patterns, including signed zeros and NaN payloads.
        values = np.arange(65536, dtype=np.uint16)
        base, escapes = exponent_window(values)
        pack_verified_row(values, np.empty(row_bytes(values.size, escapes), dtype=np.uint8), base, escapes)
        common = np.full(values.shape, 0x3F80, dtype=np.uint16)
        plans, size = plan_bank(np.stack([common, values, common]))
        self.assertIsNone(plans[1])
        self.assertEqual(plans[0][2], 0)
        self.assertEqual(plans[2][0], row_bytes(common.size, 0))
        self.assertEqual(size, 2 * row_bytes(common.size, 0))

    @unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "enable the CPU Triton interpreter")
    def test_mixed_bank_cpu(self):
        self.check_mixed_bank("cpu")

    @unittest.skipUnless(os.environ.get("LIGHTREASON_TEST_PACKED_CUDA") == "1", "enable CUDA validation")
    def test_mixed_bank_cuda_graph(self):
        self.check_mixed_bank("cuda", sizes=(1024 * 2048, 2048 * 512), capacity=64, pairs=4096, blocks=128)

    @unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "enable the CPU Triton interpreter")
    def test_hot_miss_split_cpu(self):
        self.check_hot_miss_split("cpu")

    @unittest.skipUnless(os.environ.get("LIGHTREASON_TEST_PACKED_CUDA") == "1", "enable CUDA validation")
    def test_hot_miss_split_cuda_graph(self):
        self.check_hot_miss_split("cuda")

    @unittest.skipUnless(os.environ.get("LIGHTREASON_TEST_PACKED_CUDA") == "1", "enable CUDA validation")
    def test_raw_tiles_cuda_graph(self):
        from freetoken.kernel.pinned import device_ptr

        rng = np.random.default_rng(841)
        host, replicas, output = [], [], []
        for elements in (1024 * 2048, 2048 * 512):
            bits = rng.integers(0, 65536, (4, elements), dtype=np.uint16).view(np.int16)
            host.append(torch.from_numpy(bits).pin_memory())
            # Distinct bytes catch an incorrect choice of source in this fixture.
            replicas.append(torch.from_numpy(~bits[[3, 1]]).cuda())
            output.append(torch.empty((6, elements), dtype=torch.int16, device="cuda"))

        def device_array(values, dtype=torch.int64):
            return torch.tensor(values, dtype=dtype, device="cuda")

        source_rows = [3, 0, 1, 2]
        slots = [4, 1, 5, 0]
        count, limit = device_array([0]), device_array([0], torch.int32)
        lookup = device_array([-1, 1, -1, 0], torch.int32)
        args = (device_array([b.data_ptr() for b in output]), device_array([device_ptr(b) for b in host]),
                device_array([b.data_ptr() for b in replicas]), device_array([b.shape[1] * 2 for b in host]),
                device_array(source_rows + [-1] * 60, torch.int32), device_array(slots + [-1] * 60, torch.int32),
                count, lookup, limit)
        for stripes, tile in GATHER_GEOMETRIES.values():
            with self.subTest(stripes=stripes, tile=tile):
                def copy():
                    _copy_raw_tiles[(stripes, 2, 64)](*args, stripes, tile, True, num_warps=4)

                copy()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    copy()
                for n, budget in ((0, 2), (3, 1), (4, 2), (4, 0)):
                    count.fill_(n)
                    limit.fill_(budget)
                    for out in output:
                        out.fill_(0x5A5A)
                    graph.replay()
                    for cold, hot, out in zip(host, replicas, output, strict=True):
                        expected = torch.full_like(out, 0x5A5A, device="cpu")
                        for expert, slot in zip(source_rows[:n], slots[:n], strict=True):
                            resident = {1: 1, 3: 0}.get(expert, -1)
                            expected[slot].copy_(hot[resident].cpu() if 0 <= resident < budget else cold[expert])
                        self.assertTrue(torch.equal(out.cpu(), expected))

    def check_hot_miss_split(self, device):
        capacity = 64
        src = torch.arange(capacity, device=device, dtype=torch.int32)
        dst = src.flip(0).contiguous()
        count = torch.zeros(1, device=device, dtype=torch.int64)
        lookup = torch.full((256,), -1, device=device, dtype=torch.int32)
        lookup[0], lookup[3], lookup[63] = 11, 8, 1
        limit = torch.tensor([12], device=device, dtype=torch.int32)
        host_src, host_dst, gpu_src, gpu_dst = (torch.empty_like(src) for _ in range(4))
        host_count, gpu_count = (torch.empty_like(count) for _ in range(2))

        def split():
            _split_hot_misses[(1,)](src, dst, count, lookup, limit, host_src, host_dst, host_count,
                                   gpu_src, gpu_dst, gpu_count, capacity, num_warps=1)

        replay = split
        if device == "cuda":
            split()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                split()
            replay = graph.replay
        for n, budget in ((0, 12), (1, 12), (5, 9), (capacity, 4), (capacity, 12)):
            count.fill_(n)
            limit.fill_(budget)
            replay()
            hot = {i: e for i, e in ((0, 11), (3, 8), (63, 1)) if e < budget}
            cold_ids = [i for i in range(n) if i not in hot]
            hot_ids = [i for i in range(n) if i in hot]
            self.assertEqual(host_count.item(), len(cold_ids))
            self.assertEqual(gpu_count.item(), len(hot_ids))
            self.assertEqual(host_src[:len(cold_ids)].tolist(), cold_ids)
            self.assertEqual(gpu_src[:len(hot_ids)].tolist(), [hot[i] for i in hot_ids])
            self.assertEqual(host_dst[:len(cold_ids)].tolist(), [63-i for i in cold_ids])
            self.assertEqual(gpu_dst[:len(hot_ids)].tolist(), [63-i for i in hot_ids])
            self.assertEqual(src.tolist(), list(range(capacity)))
            self.assertEqual(dst.tolist(), list(reversed(range(capacity))))

    def check_mixed_bank(self, device, sizes=(64, 32), capacity=4, pairs=8, blocks=4):
        rng = np.random.default_rng(76)
        banks, destinations, descriptors, keep_alive = [], [], [], []

        def address(array, *, resident=False):
            if device == "cpu":
                return array.ctypes.data
            if resident:
                replica = torch.from_numpy(array.view(np.int16)).to(device)
                keep_alive.append(replica)
                return replica.data_ptr()
            from freetoken.kernel.pinned import device_ptr

            pinned = torch.from_numpy(array.view(np.uint8)).pin_memory()
            keep_alive.append(pinned)
            return device_ptr(pinned)

        for n in sizes:
            bits = rng.integers(0, 2**16, (4, n), dtype=np.uint16)
            bits[2:] = (bits[2:] & 0x807F) | ((120 + (bits[2:] & 7)) << 7)
            bits[2, [0, 7, n - 1]] = [0x8000, 0x7F81, 0xFFFF]
            output = torch.full((5, n), 0x5A5A, dtype=torch.int16, device=device)
            table = []
            for expert, row in enumerate(bits):
                if expert < 2:
                    table.append((address(row, resident=(expert == 0)), -1, 0))
                    continue
                base, escapes = exponent_window(row)
                packed = np.empty(row_bytes(n, escapes), dtype=np.uint8)
                pack_verified_row(row, packed, base, escapes)
                table.append((address(packed), base, escapes))
                keep_alive.append(packed)
            banks.append(bits)
            destinations.append(output)
            descriptors.append(table)

        src_ids, dst_slots = [2, 0, 3, -1], [4, 1, 2, -1]
        sizes = torch.tensor([bank.shape[1] * 2 for bank in banks], device=device)
        src = torch.tensor(src_ids + [-1] * (capacity - len(src_ids)), dtype=torch.int32, device=device)
        dst = torch.tensor(dst_slots + [-1] * (capacity - len(dst_slots)), dtype=torch.int32, device=device)
        fetched = torch.empty_like(src)
        count = torch.tensor([0], device=device)
        args = (torch.tensor(descriptors, device=device),
                torch.tensor([out.data_ptr() for out in destinations], device=device), sizes, src, dst, count)

        def copy():
            _copy_rows[(blocks, 2)](*args, 4, pairs, blocks, num_warps=8 if capacity == 64 else 4)
            _repair_rows[(capacity, 2)](*args, 4, 256 if capacity == 64 else 8, num_warps=4)
            _snapshot_fetched[(1,)](fetched, count, src, 0, capacity, num_warps=1)

        replay = copy
        if device == "cuda":
            copy()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                copy()
            replay = graph.replay
        for copies in (0, 3):
            count.fill_(copies)
            replay()
            self.assertEqual(fetched.tolist(), src_ids[:copies] + [-1] * (capacity - copies))
            for bits, out in zip(banks, destinations, strict=True):
                expected = np.full(out.shape, 0x5A5A, dtype=np.uint16)
                for expert, slot in zip(src_ids[:copies], dst_slots[:copies], strict=True):
                    expected[slot] = bits[expert]
                np.testing.assert_array_equal(out.cpu().numpy().view(np.uint16), expected)


if __name__ == "__main__":
    unittest.main()
