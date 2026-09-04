"""Tests for Terry's request-facing seed contract."""

import random
import unittest
from unittest.mock import patch

from localhost_melodyflow import AudioProcessingError, resolve_seed


class ResolveSeedTests(unittest.TestCase):
    def test_explicit_seed_is_used_as_is(self):
        self.assertEqual(resolve_seed(1234), 1234)
        self.assertEqual(resolve_seed("1234"), 1234)
        self.assertEqual(resolve_seed(0), 0)

    def test_missing_blank_or_negative_seed_is_random(self):
        for value in (None, "", -1, "-1", -99):
            with patch.object(random, "randint", return_value=4242) as randint:
                self.assertEqual(resolve_seed(value), 4242)
            randint.assert_called_once_with(0, 99999)

    def test_random_seed_stays_in_ui_range(self):
        for _ in range(200):
            self.assertIn(resolve_seed(None), range(100000))

    def test_junk_seed_is_a_request_error(self):
        with self.assertRaisesRegex(AudioProcessingError, "Invalid seed"):
            resolve_seed("banana")


if __name__ == "__main__":
    unittest.main()
