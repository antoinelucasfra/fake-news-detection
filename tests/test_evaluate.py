"""Unit tests for the zero-shot TRUE/FALSE answer parser — pure logic, no model needed.

Runs in CI (which installs the full dependency set); locally just compile-check.
Note: importing evaluate pulls torch, so these tests only run where deps exist.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import parse_true_false


def test_plain_true_false():
    assert parse_true_false("TRUE") == 1
    assert parse_true_false("FALSE") == 0


def test_qualified_answers():
    assert parse_true_false("True.") == 1
    assert parse_true_false('"True"') == 1
    assert parse_true_false("Probably TRUE, the data supports it.") == 1
    assert parse_true_false("FALSE. The claim is exaggerated.") == 0


def test_negations_do_not_count_as_true():
    # Substring matching would misread these as TRUE
    assert parse_true_false("NOT TRUE") == 0
    assert parse_true_false("UNTRUE") == 0
    assert parse_true_false("It is not true.") == 0


def test_empty_or_ambiguous_defaults_to_false():
    assert parse_true_false("") == 0
    assert parse_true_false("   ") == 0
    assert parse_true_false("I cannot say.") == 0
