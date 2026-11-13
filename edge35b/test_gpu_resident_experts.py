"""Exercise inline fallback tiles and layout switching under Ada graph replay."""

import unittest

import torch

from edge35b.gpu_resident_experts import GEOMETRY, encode
from edge35b.native_bf16_gather import NativeBf16Gather


class ResidentCopyTests(unittest.TestCase):
    def test_mixed_tiles_and_descriptor_switch(self):
        gen = torch.Generator().manual_seed(7091)
        words = torch.randint(0, 128, (32, 256), generator=gen, dtype=torch.int32)
        words |= torch.randint(0, 2, words.shape, generator=gen, dtype=torch.int32) << 15
        exponents = 118 + torch.randint(0, 8, words.shape, generator=gen, dtype=torch.int32)
        exponents[1::4] = torch.arange(256) % 128
        words |= exponents << 7
        original = words.to(torch.int16).view(torch.bfloat16)
        data, headers, encoded, fallback = encode(original)
        self.assertEqual((encoded, fallback), (24, 8))
        packed, meta = (torch.from_numpy(value).to("cuda") for value in (data, headers))
        other = original.view(torch.int16) ^ 31
        raw = other.to("cuda")
        descriptors = torch.tensor([[[packed.data_ptr(), meta.data_ptr(), -1],
                                     [0, 0, raw.data_ptr()]]], device="cuda", dtype=torch.int64)
        output = torch.full((4, *original.shape), -123, device="cuda", dtype=torch.int16)
        destinations = torch.tensor([output.data_ptr()], device="cuda", dtype=torch.int64)
        sizes = torch.tensor([original.numel() * 2], device="cuda", dtype=torch.int64)
        ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        slots = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
        count = torch.tensor([2], device="cuda", dtype=torch.int64)
        native = NativeBf16Gather()

        def copy():
            native.copy(descriptors, destinations, sizes, ids, slots, count,
                        max_copies=2, experts=2, geometry=GEOMETRY)

        copy()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copy()
        references = [original.view(torch.int16), other]
        for n in (2, 0, 1):
            descriptors.copy_(descriptors.flip(1))
            references.reverse()
            output.fill_(-123)
            count.fill_(n)
            graph.replay()
            expected = torch.full_like(output.cpu(), -123)
            for slot, reference in zip((1, 3)[:n], references[:n], strict=True):
                expected[slot] = reference
            self.assertTrue(torch.equal(output.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
