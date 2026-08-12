import unittest

from generate_and_train import pop_opt_value


class GenerateAndTrainHelperTests(unittest.TestCase):
    def test_pop_forwarded_trainer_supports_split_and_equals_forms(self):
        split = ["--seed", "3", "--trainer", "gpu", "--batch-size", "8"]
        self.assertEqual(pop_opt_value("--trainer", split), "gpu")
        self.assertEqual(split, ["--seed", "3", "--batch-size", "8"])

        equals = ["--trainer=cpu", "--seed", "5"]
        self.assertEqual(pop_opt_value("--trainer", equals), "cpu")
        self.assertEqual(equals, ["--seed", "5"])

    def test_missing_forwarded_trainer_value_fails_closed(self):
        with self.assertRaisesRegex(SystemExit, "requires a value"):
            pop_opt_value("--trainer", ["--trainer"])


if __name__ == "__main__":
    unittest.main()
