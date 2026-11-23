"""Failure cases that must never become zero-valued outcomes."""

import collections
import copy
import json
from pathlib import Path
import unittest

import numpy as np

import arguana
import study


class MissingResultsTests(unittest.TestCase):
    def setUp(self):
        self.docs = [f"d{i}" for i in range(10)]
        self.queries = ["q0", "q1"]
        self.qrels = {"q0": ["d1"], "q1": ["d2"]}

    def test_missing_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cover every"):
            study.metrics_scores(np.zeros((1, 10)), self.docs, self.qrels, self.queries)

    def test_absent_or_unknown_positive_is_rejected(self):
        for broken in ({"q0": ["d1"]}, {"q0": ["absent"], "q1": ["d2"]}):
            with (
                self.subTest(broken=broken),
                self.assertRaisesRegex(ValueError, "relevance"),
            ):
                study.metrics_scores(
                    np.zeros((2, 10)),
                    self.docs,
                    collections.defaultdict(list, broken),
                    self.queries,
                )

    def test_actual_zero_after_valid_ranking_is_accepted(self):
        docs = self.docs + ["outside-top-ten"]
        scores = np.array([list(range(10)) + [-1]])
        metrics, top = study.metrics_scores(
            scores, docs, {"q": ["outside-top-ten"]}, ["q"]
        )
        np.testing.assert_array_equal(metrics, np.zeros((1, 3)))
        self.assertEqual(top.shape, (1, 10))

    def test_budget_applies_to_selected_numeric_channel(self):
        declared = json.loads(
            Path(study.__file__).with_name("selection.json").read_bytes()
        )
        arguana.validate_selection(declared, study.BUDGETS)
        broken = copy.deepcopy(declared)
        broken["64"]["numeric"]["config"]["bits"] = 8
        broken["64"]["numeric_key"] = "raw_r384_b8"
        with self.assertRaisesRegex(ValueError, "Invalid numerical selection"):
            arguana.validate_selection(broken, [64])

    def test_decoder_identity_must_match_winning_key(self):
        declared = json.loads(
            Path(study.__file__).with_name("selection.json").read_bytes()
        )
        declared["64"]["text"]["decoder"] = "dense"
        with self.assertRaisesRegex(ValueError, "Invalid text selection"):
            arguana.validate_selection(declared, [64])


if __name__ == "__main__":
    unittest.main()
