from pcm.message_assembly import build_proxy_system_prompt, build_system_prompt


def test_build_system_prompt_order():
    system = build_system_prompt(
        compressed_instruction="TASK=review INPUT=python CHECK=race",
        response_lang="en",
        output_style="concise",
        pcm_interpretation_hint="Interpret PCM key=value lines.",
    )
    parts = system.split("\n\n")
    assert parts[0] == "TASK=review INPUT=python CHECK=race"
    assert parts[1].startswith("RESPONSE:")
    assert parts[2] == "Interpret PCM key=value lines."


def test_build_system_prompt_without_hint():
    system = build_system_prompt(
        compressed_instruction="TASK=review INPUT=python",
        response_lang="en",
        output_style="normal",
    )
    assert "TASK=review" in system
    assert "RESPONSE:" in system
    assert system.count("\n\n") == 1


def test_build_proxy_system_prompt_no_instruction():
    system = build_proxy_system_prompt(
        response_lang="en",
        output_style="concise",
        pcm_interpretation_hint="PCM hint here.",
    )
    assert not system.startswith("TASK=")
    assert system.startswith("RESPONSE:")
    assert system.endswith("PCM hint here.")
