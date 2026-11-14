"""CPU tests for the synthetic RL memory task."""

import unittest

from .rl_episodes import MEMORY_KEYS, UNKNOWN_TARGET, VALUES, make_episodes


class RLEpisodesTests(unittest.TestCase):
    def test_generation_is_seeded_and_case_layouts_vary(self):
        episodes = make_episodes(10, 20)
        self.assertEqual(episodes, make_episodes(10, 20))
        self.assertNotEqual(episodes, make_episodes(11, 20))
        positions = {
            next(i for i, row in enumerate(episode["observations"]) if row["is_correction"])
            for episode in episodes
        }
        self.assertGreater(len(positions), 1)

    def test_holdout_pairs_do_not_overlap_and_lantern_is_covered(self):
        pairs = []
        for split in ("train", "validation", "test"):
            observed = {
                (row["key"], row["value"])
                for episode in make_episodes(123, 128, split)
                for row in episode["observations"]
                if not row["is_distractor"]
            }
            self.assertEqual({key for key, _ in observed}, set(MEMORY_KEYS))
            self.assertEqual({value for _, value in observed}, set(VALUES))
            lantern_colors = {value for key, value in observed if key == "lantern"}
            self.assertGreaterEqual(len(lantern_colors), 2)
            pairs.append(observed)
        self.assertFalse(pairs[0] & pairs[1])
        self.assertFalse(pairs[0] & pairs[2])
        self.assertFalse(pairs[1] & pairs[2])

    def test_offsets_chronology_and_unknown_labels(self):
        for episode in make_episodes(49, 40, "test"):
            observations, queries = episode["observations"], episode["queries"]
            self.assertEqual(len(observations), 8)
            self.assertEqual(len(queries), 14)
            self.assertEqual({query["kind"] for query in queries},
                             {"initial", "unknown", "retention", "correction", "distractor"})
            self.assertEqual(sorted(query["after_write"] for query in queries),
                             [query["after_write"] for query in queries])
            retained = {}
            for index, observation in enumerate(observations, 1):
                if observation["is_distractor"]:
                    self.assertIsNone(observation["target"])
                    self.assertIsNone(observation["slot"])
                else:
                    text = observation["text"]
                    key, value = observation["key"], observation["value"]
                    key_start, value_start = observation["key_start"], observation["value_start"]
                    self.assertEqual(key, text[key_start:key_start + len(key)])
                    self.assertEqual(value, text[value_start:value_start + len(value)])
                    self.assertEqual(MEMORY_KEYS[observation["slot"]], key)
                    self.assertEqual(VALUES[observation["target"]], value)
                    if observation["is_correction"]:
                        self.assertIn(key, retained)
                        self.assertNotEqual(retained[key], value)
                    retained[key] = value
                for query in (row for row in queries if row["after_write"] == index):
                    key = query["key"]
                    self.assertEqual(query["value"], retained.get(key))
                    self.assertEqual(MEMORY_KEYS[query["slot"]], key)
                    key_start = query["key_start"]
                    self.assertEqual(key, query["text"][key_start:key_start + len(key)])
                    target = VALUES.index(retained[key]) if key in retained else UNKNOWN_TARGET
                    self.assertEqual(query["target"], target)

    def test_transient_examples_have_write_labels_but_no_durable_answer(self):
        for episode in make_episodes(41, 32, "test", transient=True):
            transient, = (row for row in episode["observations"] if row.get("is_transient"))
            self.assertFalse(transient["is_distractor"])
            self.assertIsNotNone(transient["target"])
            queries = [row for row in episode["queries"] if row["kind"] == "transient"]
            self.assertEqual(len(queries), 2)
            self.assertTrue(all(row["key"] == transient["key"] and row["target"] == UNKNOWN_TARGET
                                for row in queries))

    def test_bad_arguments(self):
        for args in ((0, 0), (True, 1), (0, 1, "dev")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                make_episodes(*args)


if __name__ == "__main__":
    unittest.main()
