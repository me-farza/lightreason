import itertools
import unittest

from edge35b.dflash_prefix_selection import STEPS, TOP_K, select_prefix

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Requires the benchmark's PyTorch environment")
class PrefixSelectionTest(unittest.TestCase):
    def test_matches_exhaustive_expected_prefix(self):
        gen = torch.Generator().manual_seed(4301)
        ids = torch.arange(STEPS * TOP_K).reshape(1, STEPS, TOP_K)
        unary = torch.full(ids.shape, -1000.0)
        unary[..., :2] = torch.randn((1, STEPS, 2), generator=gen)
        anchor = torch.tensor([STEPS * TOP_K])
        predecessors, successors = (torch.randn((anchor.item() + 1, 3), generator=gen) for _ in range(2))
        state = torch.randn((1, STEPS, 3), generator=gen)

        def score(path):
            prev = anchor.item()
            expected = survival = 0.0
            probabilities = []
            for slot, candidate in enumerate(path):
                choices = ids[0, slot]
                values = unary[0, slot] + (successors[choices] * predecessors[prev] * state[0, slot]).sum(-1)
                p = values.softmax(dim=-1)[candidate].item()
                survival = p if slot == 0 else survival * p
                expected += survival
                probabilities.append(p)
                prev = choices[candidate].item()
            return expected, probabilities

        best = max(itertools.product(range(2), repeat=STEPS), key=lambda path: score(path)[0])
        expected_tokens = [ids[0, i, rank].item() for i, rank in enumerate(best)]
        got, confidence = select_prefix(ids, unary, state, anchor, predecessors, successors)
        self.assertEqual(got[0].tolist(), expected_tokens)
        torch.testing.assert_close(confidence[0], torch.tensor(score(best)[1]))


if __name__ == "__main__":
    unittest.main()
