"""Multi-Doc QA scoring + answer normalization. Dataset loading test is integration-only."""
from budgetfair.tasks.base import Episode
from budgetfair.tasks.multidoc_qa import MultiDocQA, normalize_answer


def test_normalize_answer_lowercase():
    assert normalize_answer("Paris") == "paris"


def test_normalize_answer_strips_articles():
    assert normalize_answer("the eiffel tower") == "eiffel tower"
    assert normalize_answer("an apple") == "apple"
    assert normalize_answer("a book") == "book"


def test_normalize_answer_strips_punctuation():
    assert normalize_answer("Hello, World!") == "hello world"


def test_normalize_answer_collapses_whitespace():
    assert normalize_answer("hello   world") == "hello world"


def test_normalize_answer_arabic_passthrough():
    """Non-Latin scripts are not punctuation; should pass through."""
    # No articles to strip in Arabic; punctuation stripping leaves words intact.
    assert normalize_answer("مرحبا") == "مرحبا"


def test_score_exact_match():
    task = MultiDocQA()
    ep = Episode(episode_id="e1", initial_state={}, ground_truth="Paris", metadata={})
    assert task.score(ep, "Paris") == (True, 1.0)
    assert task.score(ep, "paris") == (True, 1.0)
    assert task.score(ep, "the Paris") == (True, 1.0)
    assert task.score(ep, "Paris.") == (True, 1.0)


def test_score_mismatch():
    task = MultiDocQA()
    ep = Episode(episode_id="e1", initial_state={}, ground_truth="Paris", metadata={})
    assert task.score(ep, "London") == (False, 0.0)
    assert task.score(ep, "") == (False, 0.0)


def test_score_none_answer():
    task = MultiDocQA()
    ep = Episode(episode_id="e1", initial_state={}, ground_truth="Paris", metadata={})
    assert task.score(ep, None) == (False, 0.0)
