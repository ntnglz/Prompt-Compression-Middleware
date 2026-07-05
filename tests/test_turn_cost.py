from pcm.turn_cost import TurnCostMetrics, compute_turn_cost, count_message_tokens


def test_count_message_tokens_sums_roles():
    messages = [
        {"role": "system", "content": "a b c"},
        {"role": "user", "content": "d e"},
    ]
    # tiktoken gpt-4: exact count checked against canonical helper
    from pcm.canonical import count_tokens

    expected = count_tokens("a b c") + count_tokens("d e")
    assert count_message_tokens(messages) == expected


def test_compute_turn_cost_splits_prices():
    metrics = compute_turn_cost(
        messages=[{"role": "user", "content": "hello world"}],
        output_text="short answer",
        output_tokens=3,
        input_price_per_m=1.5,
        output_price_per_m=7.5,
    )
    assert metrics.input_tokens > 0
    assert metrics.output_tokens == 3
    assert metrics.cost_input == round(metrics.input_tokens * 1.5 / 1_000_000, 6)
    assert metrics.cost_output == round(3 * 7.5 / 1_000_000, 6)
    assert metrics.cost_total == round(metrics.cost_input + metrics.cost_output, 6)


def test_turn_cost_metrics_to_dict():
    m = TurnCostMetrics(
        input_tokens=100,
        input_tokens_instruction=40,
        input_tokens_context=60,
        output_tokens=20,
        input_price_per_m=1.5,
        output_price_per_m=7.5,
        cost_input=0.00015,
        cost_output=0.00015,
        cost_total=0.0003,
    )
    d = m.to_dict()
    assert d["cost_total"] == 0.0003
    assert d["output_tokens"] == 20
