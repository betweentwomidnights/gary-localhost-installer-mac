"""Tests for Gary's request-facing seed contract."""

import random
import unittest
from unittest.mock import patch

from g4l_localhost import AudioRequest, resolve_seed


class ResolveSeedTests(unittest.TestCase):
    def test_explicit_seed_is_preserved(self):
        self.assertEqual(resolve_seed(1234), 1234)
        self.assertEqual(resolve_seed(0), 0)

    def test_missing_or_negative_seed_is_random(self):
        for value in (None, -1, -99):
            with patch.object(random, "randint", return_value=4242) as randint:
                self.assertEqual(resolve_seed(value), 4242)
            randint.assert_called_once_with(0, 99999)

    def test_blank_seed_validates_as_random(self):
        request = AudioRequest(
            audio_data="audio",
            model_name="model",
            prompt_duration=6,
            seed="",
        )
        self.assertIsNone(request.seed)


if __name__ == "__main__":
    unittest.main()
