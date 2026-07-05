import pytest
from pcm.proxy import ChatProxy, ProxyConfig
from pcm.compressor import PromptCompressor
from pcm.upstream import UpstreamTarget


@pytest.fixture
def mistral_upstream():
    return UpstreamTarget(
        provider="mistral",
        base_url="https://api.mistral.ai/v1",
        api_key="mistral-key",
        model="mistral-medium-3.5",
        supports_reasoning_effort=True,
        reasoning_effort="none",
    )


@pytest.mark.asyncio
async def test_transform_injects_response_block(mistral_upstream):
    config = ProxyConfig(
        inject_pcm_system=True,
        output_style="concise",
        response_lang="en",
        compress_roles=frozenset(),
    )
    proxy = ChatProxy(PromptCompressor(), config=config)
    body = {"messages": [{"role": "user", "content": "Hello"}]}
    transformed, _ = await proxy.transform_request(
        body, upstream=mistral_upstream, compress=False
    )
    system = next(m["content"] for m in transformed["messages"] if m["role"] == "system")
    assert "RESPONSE:" in system
    assert "Answer only what was asked" in system
