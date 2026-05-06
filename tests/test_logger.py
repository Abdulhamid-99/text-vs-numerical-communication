"""EpisodeLogger correctness. JSONL roundtrip + CSV header behavior."""
import csv
import json

from budgetfair.logging_utils.logger import EpisodeLogger


def test_message_log_writes_jsonl(tmp_path):
    logger = EpisodeLogger(tmp_path, "run1")
    logger.log_message(episode_id="e1", sender="a", receiver="b",
                       channel="text", bit_cost=80, payload_text="hello")
    p = tmp_path / "run1" / EpisodeLogger.MESSAGES_FILE
    assert p.exists()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["episode_id"] == "e1"
    assert rec["bit_cost"] == 80
    assert rec["payload_text"] == "hello"


def test_message_log_handles_arabic(tmp_path):
    """Don't ensure_ascii — Arabic must be readable in the JSONL."""
    logger = EpisodeLogger(tmp_path, "run1")
    logger.log_message(episode_id="e1", payload_text="مرحبا")
    p = tmp_path / "run1" / EpisodeLogger.MESSAGES_FILE
    raw = p.read_text(encoding="utf-8").strip()
    assert "مرحبا" in raw  # not escaped to \uXXXX
    rec = json.loads(raw)
    assert rec["payload_text"] == "مرحبا"


def test_episode_csv_header_then_rows(tmp_path):
    logger = EpisodeLogger(tmp_path, "run1")
    logger.finalize_episode(episode_id="e1", success=True, score=1.0, bits_used=240)
    logger.finalize_episode(episode_id="e2", success=False, score=0.0, bits_used=180)

    p = tmp_path / "run1" / EpisodeLogger.EPISODES_FILE
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["episode_id"] == "e1"
    assert rows[0]["success"] == "True"
    assert rows[1]["score"] == "0.0"


def test_multiple_runs_separate_dirs(tmp_path):
    a = EpisodeLogger(tmp_path, "runA")
    b = EpisodeLogger(tmp_path, "runB")
    a.log_message(episode_id="x")
    b.log_message(episode_id="y")
    assert (tmp_path / "runA" / EpisodeLogger.MESSAGES_FILE).exists()
    assert (tmp_path / "runB" / EpisodeLogger.MESSAGES_FILE).exists()
