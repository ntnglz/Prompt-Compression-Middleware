"""Tests para check_dataset_leakage."""

from pathlib import Path

from pcm.training.dataset import (
    build_chat_example,
    check_leakage,
    normalize_user_text,
    user_text_hash,
)

ROOT = Path(__file__).resolve().parent.parent


def test_normalize_strips_prompt_prefix():
    assert normalize_user_text("Prompt a comprimir:\nhola mundo") == "hola mundo"
    assert normalize_user_text("  hola   mundo  ") == "hola mundo"


def test_user_text_hash_stable():
    h1 = user_text_hash("Prompt a comprimir:\ntest")
    h2 = user_text_hash("test")
    assert h1 == h2


def test_detects_exact_overlap():
    excluded = {user_text_hash("hola")}
    train = [build_chat_example("hola", "TASK=explain TOPIC=test")]
    leaks = check_leakage(train, excluded)
    assert len(leaks) == 1


def test_no_leak_when_disjoint():
    excluded = {user_text_hash("eval only")}
    train = [build_chat_example("train only", "TASK=explain TOPIC=test")]
    leaks = check_leakage(train, excluded)
    assert leaks == []


def test_load_excluded_hashes_includes_gold():
    from pcm.training.dataset import load_excluded_hashes

    hashes = load_excluded_hashes(ROOT)
    assert len(hashes) >= 10
