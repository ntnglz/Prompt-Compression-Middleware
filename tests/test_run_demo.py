"""Tests for visitor-facing run.py demos."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run.py"


def _run_demo_stub() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN), "--demo-stub"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_demo_stub_exits_zero():
    result = _run_demo_stub()
    assert result.returncode == 0, result.stderr


def test_demo_stub_shows_canonical_pcm():
    from pcm.canonical import CANONICAL_INSTRUCTION, CANONICAL_PCM

    result = _run_demo_stub()
    assert CANONICAL_INSTRUCTION in result.stdout
    assert CANONICAL_PCM in result.stdout
    assert "BEFORE" in result.stdout
    assert "AFTER" in result.stdout


def test_demo_stub_mode_label():
    result = _run_demo_stub()
    assert "stub" in result.stdout.lower()


def test_canonical_examples_match_module():
    import json

    from pcm.canonical import (
        CANONICAL_INSTRUCTION,
        canonical_compress_request,
        canonical_proxy_chat_request,
    )

    compress_path = ROOT / "data" / "examples" / "canonical_compress.json"
    proxy_path = ROOT / "data" / "examples" / "proxy_chat.json"

    assert json.loads(compress_path.read_text()) == canonical_compress_request()
    body = json.loads(proxy_path.read_text())
    expected = canonical_proxy_chat_request()
    assert body["model"] == expected["model"]
    assert body["messages"][0]["content"] == expected["messages"][0]["content"]
    assert CANONICAL_INSTRUCTION in body["messages"][0]["content"]

def test_demo_stub_shows_token_metrics():
    result = _run_demo_stub()
    assert "Token savings" in result.stdout
    assert "81%" in result.stdout or "51%" in result.stdout


def test_cursor_dev_examples_match_module():
    import json

    from pcm.canonical import (
        CURSOR_DEV_INSTRUCTION,
        cursor_dev_compress_request,
        cursor_dev_proxy_chat_request,
    )

    compress_path = ROOT / "data" / "examples" / "cursor_dev_compress.json"
    proxy_path = ROOT / "data" / "examples" / "cursor_dev_triage.json"

    assert json.loads(compress_path.read_text()) == cursor_dev_compress_request()
    body = json.loads(proxy_path.read_text())
    expected = cursor_dev_proxy_chat_request()
    assert body["messages"][0]["content"] == expected["messages"][0]["content"]
    assert CURSOR_DEV_INSTRUCTION in body["messages"][0]["content"]


def test_cursor_dev_metrics_order_of_magnitude():
    from pcm.canonical import cursor_dev_metrics

    m = cursor_dev_metrics()
    assert m.instruction_saved >= 100
    assert m.instruction_ratio >= 0.75
    assert m.message_ratio >= 0.45


def test_pcm_importable_without_pythonpath():
    from pcm import PromptCompressor, __version__

    assert __version__ == "1.0.0"
    assert PromptCompressor is not None
